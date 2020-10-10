import argparse
import torch.nn.functional as F
import numpy as np
import random

from data   import *
from mir    import *
from ema    import EMA
from utils  import get_logger, get_temp_logger, logging_per_task
from buffer import Buffer
from copy   import deepcopy
from pydoc  import locate
from model  import ResNet18
from utils import SelectiveBackPropagation,naive_cross_entropy_loss, onehot, FocalLoss
import copy
# Arguments
# -----------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type=str, default='Results',
    help='directory where we save results and samples')
parser.add_argument('-u', '--unit_test', action='store_true',
    help='unit testing mode for fast debugging')
parser.add_argument('-d', '--dataset', type=str, default = 'split_cifar10',
    choices=['split_mnist', 'permuted_mnist', 'split_cifar10', 'split_cifar100', 'miniimagenet'])
parser.add_argument('--n_tasks', type=int, default=-1,
    help='total number of tasks. -1 does default amount for the dataset')
parser.add_argument('-r','--reproc', type=int, default=1,
    help='if on, no randomness in numpy and torch')
parser.add_argument('--disc_epochs', type=int, default=1)
parser.add_argument('--disc_iters', type=int, default=1,
    help='number of training iterations for the classifier')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--buffer_batch_size', type=int, default=10)
parser.add_argument('--use_conv', action='store_true')
parser.add_argument('--samples_per_task', type=int, default=-1,
    help='if negative, full dataset is used')
parser.add_argument('--mem_size', type=int, default=600, help='controls buffer size')
parser.add_argument('--n_runs', type=int, default=1,
    help='number of runs to average performance')
parser.add_argument('--suffix', type=str, default='',
    help="name for logfile")
parser.add_argument('--subsample', type=int, default=50,
    help="for subsampling in --method=replay, set to 0 to disable")
parser.add_argument('--print_every', type=int, default=500,
    help="print metrics every this iteration")
parser.add_argument('--update_buffer_hid', type=int, default=1,
    help='related to latent buffer')
# logging
parser.add_argument('-l', '--log', type=str, default='off', choices=['off', 'online'],
    help='enable WandB logging')
parser.add_argument('--wandb_project', type=str, default='mir',
    help='name of the WandB project')

#------ MIR -----#
parser.add_argument('-m','--method', type=str, default='rand_replay', choices=['no_rehearsal',
    'rand_replay', 'mir_replay'])
parser.add_argument('--compare_to_old_logits', action='store_true',help='uses old logits')
parser.add_argument('--reuse_samples', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.1)

parser.add_argument('--task_free', action='store_true')
parser.add_argument('--mask_trick', action='store_true')
parser.add_argument('--grad_trick', action='store_true')
parser.add_argument('--ema', action='store_true')

args = parser.parse_args()

# Obligatory overhead
# -----------------------------------------------------------------------------------------

if not os.path.exists(args.result_dir): os.mkdir(args.result_dir)
sample_path = os.path.join(args.result_dir,'samples/')
if not os.path.exists(sample_path): os.mkdir(sample_path)
recon_path = os.path.join(args.result_dir,'reconstructions/')
if not os.path.exists(recon_path): os.mkdir(recon_path)

args.cuda = torch.cuda.is_available()
args.device = 'cuda:0'

# argument validation
overlap = 0

fc_loss = FocalLoss(1.0)
#########################################
# TODO(Get rid of this or move to data.py)
args.ignore_mask = False
args.gen = False
args.newer = 2
#########################################

args.gen_epochs=0
args.output_loss = None

if args.reproc:
    seed=0
    torch.manual_seed(seed)
    np.random.seed(seed)

# fetch data
data = locate('data.get_%s' % args.dataset)(args)

# make dataloaders
train_loader, val_loader, test_loader  = [CLDataLoader(elem, args, train=t) \
        for elem, t in zip(data, [True, False, False])]

if args.log != 'off':
    import wandb
    wandb.init(args.wandb_project)
    wandb.config.update(args)
else:
    wandb = None

# create logging containers
LOG = get_logger(['cls_loss', 'acc'],
        n_runs=args.n_runs, n_tasks=args.n_tasks)

args.mem_size = args.mem_size*args.n_classes #convert from per class to total memory

# Eval model
# -----------------------------------------------------------------------------------------

def eval_model(model, loader, task, mode='valid'):
    model = model.eval()

    for task_t, te_loader in enumerate(loader):
        if task_t > task: break
        LOG_temp = get_temp_logger(None, ['cls_loss', 'acc'])

        # iterate over samples from task
        for i, (data, target) in enumerate(te_loader):
            if args.unit_test and i > 10: break

            if args.cuda:
                data, target = data.to(args.device), target.to(args.device)

            logits = model(data)

            if args.multiple_heads:
                logits = logits.masked_fill(te_loader.dataset.mask == 0, -1e9)

            loss = F.cross_entropy(logits, target)
            pred = logits.argmax(dim=1, keepdim=True)

            LOG_temp['acc'] += [pred.eq(target.view_as(pred)).sum().item() / pred.size(0)]
            LOG_temp['cls_loss'] += [loss.item()]

        logging_per_task(wandb, LOG, run, mode, 'acc', task, task_t,
                 np.round(np.mean(LOG_temp['acc']),2))
        logging_per_task(wandb, LOG, run, mode, 'cls_loss', task, task_t,
                 np.round(np.mean(LOG_temp['cls_loss']),2))

    print('\n{}:'.format(mode))
    print(LOG[run][mode]['acc'])


# Train the model
# -----------------------------------------------------------------------------------------

for run in range(args.n_runs):

    class_count=torch.ones(args.n_classes).cuda()
    # REPRODUCTIBILITY
    if args.reproc:
        np.random.seed(run)
        torch.manual_seed(run)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # CLASSIFIER
    model = ResNet18(args.n_classes, nf=20, input_size=args.input_size)
    if args.cuda:
        model = model.to(args.device)
    model.train()

    sb = SelectiveBackPropagation(
        args.buffer_batch_size,
        epoch_length=500,
        loss_selection_threshold=False)

    sb2 = SelectiveBackPropagation(
        args.batch_size,
        epoch_length=500,
        loss_selection_threshold=False)

    if args.ema:
        EMAS = []
        # for n_step in [1, 10, 50]:
        for decay in [0.99]:#, 0.9, 0.99, 0.999]:
            EMAS += [EMA(gamma=decay, update_freq=3)]

    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    buffer = Buffer(args)
    if run == 0:
        print("number of classifier parameters:",
                sum([np.prod(p.size()) for p in model.parameters()]))
        print("buffer parameters: ", np.prod(buffer.bx.size()))

    mask_so_far = None
    #----------
    # Task Loop
    for task, tr_loader in enumerate(train_loader):
        sample_amt = 0

        model = model.train()

       # import ipdb; ipdb.set_trace()
        if False and mask_so_far is not None:
            scaling = torch.norm(model.linear.weight,dim=1)[mask_so_far].mean().detach().item()
            curr = model.linear.weight[tr_loader.dataset.mask.byte(),:].detach()
            model.linear.weight.data[tr_loader.dataset.mask.byte(),:]= (scaling/torch.norm(curr,dim=1).unsqueeze(1))*curr
        #---------------
        # Minibatch Loop
        for i, (data, target) in enumerate(tr_loader):
            if args.unit_test and i > 10: break
            if sample_amt > args.samples_per_task > 0: break
            sample_amt += data.size(0)

            if args.cuda:
                data, target = data.to(args.device), target.to(args.device)


            #------ Train Classifier-------#
            if i==0:
                print('\n--------------------------------------')
                print('Run #{} Task #{} --> Train Classifier'.format(
                    run, task))
                print('--------------------------------------\n')

            #---------------
            # Iteration Loop
            for it in range(args.disc_iters):
                if args.method == 'no_rehearsal':
                    rehearse = False
                else:
                    rehearse = (task + i) > 0 if args.task_free else task > 0


               # logits = model(data)
              #  hidden = model.return_hidden(data)
               # present = target.unique()

                if False and  task>0:
                    hidden = model.return_hidden(data)
                    present = target.unique()
                    x1=[]
                    x2=[]
                    for t in present:
                        hid = hidden[torch.nonzero(target==t).squeeze()]
                        if len(hid.shape)==1:
                          #  import ipdb; ipdb.set_trace()
                            continue
                        x1.append(hid)
                        idx=np.arange(0,hid.shape[0])[::-1]
                        x2.append(hid[idx.copy()])
                   # import ipdb; ipdb.set_trace()
                    x1=torch.cat(x1)
                    x2=torch.cat(x2)
                    loss=torch.norm(x1-x2).mean()/1000.
                 #   import ipdb;ipdb.set_trace()
                
                if args.mask_trick:
                    logits = model(data)
                    # mask logits not present in batch
                    present = target.unique()
                    if mask_so_far is not None:
                        mask_so_far = torch.cat([mask_so_far,present]).unique()
                    else:
                        mask_so_far = present
                    mask = torch.zeros_like(logits)
                    if True:
                        mask[:,present]=1
                        logits[:,present]
                        m=torch.zeros(args.n_classes).byte()
                        m[present]=1
                        m2=torch.zeros(args.n_classes).byte()
                        m2[mask_so_far]=1
                        m[~m2]=1
                        if task>0:
                            w=1./class_count[~m]
                            w=0.3*w/w.sum()
                            logits[:,~m] = logits[:,~m]*w[None,:]
                        else:
                            logits=logits.masked_fill(mask==0,-1e9)
                        #logits=logits*mask.float()+(1.-mask.float())*logits/50.

                        loss_a = F.cross_entropy(logits, target, reduction='none')
                        loss = ((loss_a).sum() / loss_a.size(0)) * (1. / float(it + 1.))
                        for z in range(len(target)):
                            class_count[target[z]]+=1
                    else:
                        mask[:,present]=1
                        logits=logits.masked_fill(mask==0,-1e9)
                        loss = fc_loss(logits,target)





                opt.zero_grad()
                loss.backward()

                if rehearse:
                    mem_x, mem_y, bt = buffer.sample(args.buffer_batch_size,reset=it==0) # , exclude_task=task)


                    logits_buffer = model(mem_x)
                    #if task>0:
                     #   max_conf_seen_for_seen = F.softmax(logits_buffer, dim=1)[:, tr_loader.dataset.mask.byte()].max(dim=1)[0].mean()
                    if args.mask_trick:
                        present = mask_so_far
                        mask = torch.zeros_like(logits_buffer)
                        mask[:, present] = 1
                        logits_buffer = logits_buffer.masked_fill(mask == 0, -1e9)


                    loss_a = F.cross_entropy(logits_buffer, mem_y, reduction='none')
                    #loss = fc_loss(logits_buffer,mem_y)
                    loss = (loss_a).sum() / loss_a.size(0)
                    loss.backward()

                model(data)
                opt.step()
                #if task>0:
                 #   print('%.2f for incoming, confidence in old,  %.2f for old confidence in incoming'%(max_conf_seen_for_unseen,max_conf_seen_for_seen))
            buffer.add_reservoir(data, target, None, task)
            if args.ema:
                for EMA in EMAS:
                    EMA.update(model)

        # ------------------------ eval ------------------------ #
        if args.ema:
            for EMA in EMAS:
                print('EMA ', EMA.gamma)
                eval_model(EMA, val_loader, task)

        eval_model(model, val_loader, task)

    # final run results
    print('--------------------------------------')
    print('Run #{} Final Results'.format(run))
    print('--------------------------------------')
    for mode in ['valid','test']:
        final_accs = LOG[run][mode]['acc'][:,task]
        logging_per_task(wandb, LOG, run, mode, 'final_acc', task,
            value=np.round(np.mean(final_accs),2))
        best_acc = np.max(LOG[run][mode]['acc'], 1)
        final_forgets = best_acc - LOG[run][mode]['acc'][:,task]
        logging_per_task(wandb, LOG, run, mode, 'final_forget', task,
            value=np.round(np.mean(final_forgets[:-1]),2))

        print('\n{}:'.format(mode))
        print('final accuracy: {}'.format(final_accs))
        print('average: {}'.format(LOG[run][mode]['final_acc']))
        print('final forgetting: {}'.format(final_forgets))
        print('average: {}\n'.format(LOG[run][mode]['final_forget']))


# final results
print('--------------------------------------')
print('--------------------------------------')
print('FINAL Results')
print('--------------------------------------')
print('--------------------------------------')
for mode in ['valid']:

    final_accs = [LOG[x][mode]['final_acc'] for x in range(args.n_runs)]
    final_acc_avg = np.mean(final_accs)
    final_acc_se = 2*np.std(final_accs) / np.sqrt(args.n_runs)
    final_forgets = [LOG[x][mode]['final_forget'] for x in range(args.n_runs)]
    final_forget_avg = np.mean(final_forgets)
    final_forget_se = 2*np.std(final_forgets) / np.sqrt(args.n_runs)

    print('\nFinal {} Accuracy: {:.3f} +/- {:.3f}'.format(mode, final_acc_avg, final_acc_se))
    print('\nFinal {} Forget: {:.3f} +/- {:.3f}'.format(mode, final_forget_avg, final_forget_se))

    if wandb is not None:
        wandb.log({mode+'final_acc_avg':final_acc_avg})
        wandb.log({mode+'final_acc_se':final_acc_se})
        wandb.log({mode+'final_forget_avg':final_forget_avg})
        wandb.log({mode+'final_forget_se':final_forget_se})
