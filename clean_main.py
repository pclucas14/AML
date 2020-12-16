import argparse
import torch.nn.functional as F
import numpy as np
import random

from copy import deepcopy
from data   import *
from utils  import get_logger, get_temp_logger, logging_per_task, sho_
from buffer import Buffer
from copy   import deepcopy
from pydoc  import locate
from model  import ResNet18,normalize, ContrastiveLoss
from utils import naive_cross_entropy_loss, onehot, Lookahead, AverageMeter
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
parser.add_argument('--samples_per_task', type=int, default=-1,
    help='if negative, full dataset is used')
parser.add_argument('--mem_size', type=int, default=600, help='controls buffer size')
parser.add_argument('--n_runs', type=int, default=1,
    help='number of runs to average performance')
parser.add_argument('--suffix', type=str, default='',
    help="name for logfile")
parser.add_argument('--print_every', type=int, default=500,
    help="print metrics every this iteration")
# logging
parser.add_argument('-l', '--log', type=str, default='off', choices=['off', 'online'],
    help='enable WandB logging')
parser.add_argument('--wandb_project', type=str, default='er_imbalance',
    help='name of the WandB project')
parser.add_argument('--mask_vers', type=int, default=1)
#------ MIR -----#
parser.add_argument('-m','--method', type=str, default='er', choices=['er', 'mask', 'triplet'])
parser.add_argument('--lr', type=float, default=0.1)

parser.add_argument('--incoming_neg', type=float, default=2.0)
parser.add_argument('--buffer_neg', type=float, default=2.0)

parser.add_argument('--task_free', action='store_true')
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
    wandb.init(project=args.wandb_project+'_'+args.dataset, name=args.suffix)
    wandb.config.update(args)
else:
    wandb = None

# create logging containers
LOG = get_logger(['cls_loss', 'acc'],
        n_runs=args.n_runs, n_tasks=args.n_tasks)

args.mem_size = args.mem_size*args.n_classes #convert from per class to total memory

# Eval model
# -----------------------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix
def eval_model(model, loader, task, mode='valid'):
    model = model.eval()
    y_pred =[]
    y_truth=[]
    for task_t, te_loader in enumerate(loader):
        if task_t > task: break
        LOG_temp = get_temp_logger(None, ['cls_loss', 'acc'])

        # iterate over samples from task
        for i, (data, target, _) in enumerate(te_loader):
            if args.unit_test and i > 10: break

            if args.cuda:
                data, target = data.to(args.device), target.to(args.device)

            logits = model(data)

            if args.multiple_heads:
                logits = logits.masked_fill(te_loader.dataset.mask == 0, -1e9)

            loss = F.cross_entropy(logits, target)
            pred = logits.argmax(dim=1, keepdim=True)
            y_pred.append(pred.squeeze().cpu().numpy())
            y_truth.append(target.squeeze().cpu().numpy())
            LOG_temp['acc'] += [pred.eq(target.view_as(pred)).sum().item() / pred.size(0)]
            LOG_temp['cls_loss'] += [loss.item()]

        logging_per_task(wandb, LOG, run, mode, 'acc', task, task_t,
                 np.round(np.mean(LOG_temp['acc']),2))
        logging_per_task(wandb, LOG, run, mode, 'cls_loss', task, task_t,
                 np.round(np.mean(LOG_temp['cls_loss']),2))
    y_truth=np.hstack(y_truth)
    y_pred=np.hstack(y_pred)
    print(confusion_matrix(y_truth,y_pred))
    print('\n{}:'.format(mode))
    print(LOG[run][mode]['acc'])

    if wandb is not None:
        wandb.log({mode + '_anytime_acc_avg_' + str(run): LOG[run][mode]['acc'][0:task + 1, task].mean()})
        wandb.log({mode + '_anytime_last_acc_' + str(run): LOG[run][mode]['acc'][task, task]})


hid_new_change=AverageMeter()
hid_old_change=AverageMeter()

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

    opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    # new_buffer = NewBuffer(args.input_size, args.n_classes, args.mem_size).to(args.device)
    buffer = Buffer(args).to(args.device)
    if run == 0:
        print("number of classifier parameters:",
                sum([np.prod(p.size()) for p in model.parameters()]))
        print("buffer parameters: ", np.prod(buffer.bx.size()))

    mask_so_far = None
    #----------
    # Task Loop
    for task, tr_loader in enumerate(train_loader):
        sample_amt = 0

        contrastive = ContrastiveLoss(margin=0.6)

        last_mask = tr_loader.dataset.mask

        model = model.train()
        old_class_mask = deepcopy(mask_so_far)

        #---------------
        # Minibatch Loop
        for i, (data, target, idx) in enumerate(tr_loader):
            if args.unit_test and i > 10: break
            if sample_amt > args.samples_per_task > 0: break
            sample_amt += data.size(0)

            if args.cuda:
                idx = idx.to(args.device)
                data = data.to(args.device)
                target = target.to(args.device)

            buffer.add_reservoir(data, target, None, task, idx, overwrite=False)

            #------ Train Classifier-------#
            if i==0:
                print('\n--------------------------------------')
                print('Run #{} Task #{} --> Train Classifier'.format(
                    run, task))
                print('--------------------------------------\n')

            #---------------
            # Iteration Loop
            target_orig= copy.deepcopy(target)
            for it in range(args.disc_iters):
                rehearse = (task + i) > 0 if args.task_free else task > 0

                target = copy.deepcopy(target_orig)
                hidden = model.return_hidden(data)

                # mask logits not present in batch
                present = target.unique()
                if mask_so_far is not None:
                    mask_so_far = torch.cat([mask_so_far,present]).unique()
                else:
                    mask_so_far = present

                mask = torch.zeros(len(target), args.n_classes)

                if args.method == 'mask':
                    mask[:, present] = 1
                    if mask_so_far.max() < args.n_classes-1:
                        mask[:,mask_so_far.max():]=1
                    mask = mask.cuda()
                    logits = model.linear(hidden)
                    logits = logits.masked_fill(mask == 0, -1e9)
                    loss = F.cross_entropy(logits, target)
                elif task == 0 or args.method == 'er':
                    logits = model.linear(hidden)
                    loss = F.cross_entropy(logits, target)
                elif args.method == 'triplet':
                    # fetch the constrasting points
                    pos_x, neg_x_same_t, neg_x_diff_t, invalid_idx = \
                            buffer.fetch_pos_neg_samples(target, task, idx, data=data)

                    all_xs  = torch.cat((pos_x, neg_x_same_t, neg_x_diff_t))
                    all_hid = normalize(model.return_hidden(all_xs)).reshape(3, pos_x.size(0), -1)
                    pos_hid, neg_hid_same_t, neg_hid_diff_t = all_hid[:, ~invalid_idx].chunk(3)

                    hidden_norm = normalize(hidden[~invalid_idx])

                    x_show = torch.stack((data, pos_x, neg_x_same_t, neg_x_diff_t))

                    loss  = args.incoming_neg * F.triplet_margin_loss(hidden_norm, pos_hid, neg_hid_same_t, 0.2)
                    loss += args.buffer_neg * F.triplet_margin_loss(hidden_norm, pos_hid, neg_hid_diff_t, 0.2)
                else:
                    assert False

                opt.zero_grad()
                loss.backward(retain_graph=True)

                if rehearse:
                    mem_x, mem_y, bt, inds = buffer.sample(args.buffer_batch_size, aug=False, ret_ind=True)#, exclude_task=task)
                    hidden_buff = model.return_hidden(mem_x)

                    logits_buffer = model.linear(hidden_buff)

                    loss_a = F.cross_entropy(logits_buffer, mem_y, reduction='none')
                    loss = (loss_a).sum() / loss_a.size(0)
                    loss.backward()

                model(data)
                opt.step()

            # make sure we don't exceed the memory requirements
            buffer.balance_memory()

        print(f'buf {buffer.bx.size(0)}')
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

        LOG[run][mode]['last_task_acc'] = np.diag(
            LOG[run][mode]['acc']).mean()  # (LOG[run][mode]['acc'][:task+1,task][-1])
        LOG[run][mode]['allbutfirst_tasks_acc'] = np.mean(LOG[run][mode]['acc'][:task + 1, task][:-1])

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
    final_last_task_acc= np.mean( [LOG[x][mode]['last_task_acc'] for x in range(args.n_runs)])
    final_allbutfirst_tasks_acc = np.mean([LOG[x][mode]['allbutfirst_tasks_acc'] for x in range(args.n_runs)])
    print('\nFinal {} Accuracy: {:.3f} +/- {:.3f}'.format(mode, final_acc_avg, final_acc_se))
    print('\nFinal {} Forget: {:.3f} +/- {:.3f}'.format(mode, final_forget_avg, final_forget_se))
    print('\nFinal {} final_last_task_acc: {:.3f}'.format(mode, final_last_task_acc))
    print('\nFinal {} final_allbutfirst_tasks_acc: {:.3f}'.format(mode, final_allbutfirst_tasks_acc))

    if wandb is not None:
        for task in range(0,args.n_tasks):
            wandb.log({mode + '_anytime_acc_avg_all': np.mean([LOG[run][mode]['acc'][0:task + 1, task].mean() for run in range(0,args.n_runs)])})
            wandb.log({mode + '_last_acc_avg_all': np.mean([LOG[run][mode]['acc'][task,task] for run in range(0,args.n_runs)])})

        wandb.log({mode+'final_acc_avg':final_acc_avg,
                   mode+'final_acc_se':final_acc_se,
                   mode+'final_forget_avg':final_forget_avg,
                   mode+'final_forget_se':final_forget_se,
                   mode+'final_last_task_acc':final_last_task_acc,
                   mode+'final_allbutfirst_tasks_acc':final_allbutfirst_tasks_acc})
