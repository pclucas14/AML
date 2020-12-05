import argparse
import torch.nn.functional as F
import numpy as np
import random

from copy import deepcopy
from data   import *
from mir    import *
from ema    import EMA
from utils  import get_logger, get_temp_logger, logging_per_task
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
parser.add_argument('--wandb_project', type=str, default='er_imbalance',
    help='name of the WandB project')
parser.add_argument('--mask_vers', type=int, default=1)
#------ MIR -----#
parser.add_argument('-m','--method', type=str, default='rand_replay', choices=['no_rehearsal',
    'rand_replay', 'mir_replay'])
parser.add_argument('--compare_to_old_logits', action='store_true',help='uses old logits')
parser.add_argument('--reuse_samples', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.1)

parser.add_argument('--task_free', action='store_true')
parser.add_argument('--mask_trick', action='store_true')
parser.add_argument('--prot_trick', action='store_true')
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
        for i, (data, target) in enumerate(te_loader):
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
    #import ipdb; ipdb.set_trace()
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

    base_opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    opt = Lookahead(base_opt, k=args.disc_iters, alpha=0.5)  # Initialize Lookahead

    buffer = Buffer(args)
    if run == 0:
        print("number of classifier parameters:",
                sum([np.prod(p.size()) for p in model.parameters()]))
        print("buffer parameters: ", np.prod(buffer.bx.size()))

    mask_so_far = None
    buffer.dist_mat = buffer.dist_mat.cuda()
    buffer.dist_amt = buffer.dist_amt.cuda()
    #----------
    # Task Loop
    for task, tr_loader in enumerate(train_loader):
        sample_amt = 0

        contrastive = ContrastiveLoss(margin=0.6)
        if False and task>0:
            w_dat = model.linear.L.weight.data
            L_norm = torch.norm(w_dat,p=2,dim=1).unsqueeze(1).expand_as(w_dat)
            w_dat = w_dat.div(L_norm + 0.00001)
            #new_w = -w_dat[mask_so_far].mean(dim=0)
            model.linear.L.weight.data[tr_loader.dataset.mask.byte()] = -w_dat[last_mask.byte()]
        last_mask = tr_loader.dataset.mask

        model = model.train()
       # import ipdb; ipdb.set_trace()
        old_class_mask = deepcopy(mask_so_far)
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
            if False and i==0:
                if mask_so_far is not None:
                    model.eval()
                    un_t=target.unique()
                    out = normalize(model.return_hidden(data)).detach().data
                    for c in un_t:
                        model.linear.L.weight.data[c] = out[target==c].mean()
                    model.train()
                  #  model.linear.requires_grad = False
                    #scaling = torch.norm(model.linear.weight, dim=1)[mask_so_far].mean().detach().item()
                    #curr = model.linear.weight[tr_loader.dataset.mask.byte(), :].detach()
                    #model.linear.weight.data[tr_loader.dataset.mask.byte(), :] = (scaling / torch.norm(curr, dim=1).unsqueeze(1)) * curr
            #---------------
            # Iteration Loop
            target_orig= copy.deepcopy(target)
            for it in range(args.disc_iters):
                if args.method == 'no_rehearsal':
                    rehearse = False
                else:
                    rehearse = (task + i) > 0 if args.task_free else task > 0

                if rehearse:
                    if it==0:
                        mem_x, mem_y, bt, inds = buffer.sample(args.buffer_batch_size, ret_ind=True,
                                                               reset=task,aug=True) #,exclude_task=task)  # , exclude_task=task)
                    hidden_buff = model.return_hidden(mem_x)

                target = copy.deepcopy(target_orig)
                hidden = model.return_hidden(data)
                if False and it>0:
                    temp = np.arange(0,len(target))
                    np.random.shuffle(temp)
                    hidden=hidden[temp[:-2]]
                    target = target[temp[:-2]]
                # mask logits not present in batch
                present = target.unique()
                woof = target.unique()
                if mask_so_far is not None:
                    mask_so_far = torch.cat([mask_so_far,present]).unique()
                else:
                    mask_so_far = present

                mask = torch.zeros(len(target), args.n_classes)

                if args.mask_trick:
                    present = target.unique()
                    mask[:, present] = 1
                    mask = mask.cuda()
                    logits = model.linear(hidden)
                    logits = logits.masked_fill(mask == 0, -1e9)
                    loss = F.cross_entropy(logits, target)
                else:
                    if task==0:
                        logits = model.linear(hidden)
                        loss = F.cross_entropy(logits, target)
                    else:
                        ind_neg = []
                        anchor_pos = []
                        select = []
                        p = np.arange(0, len(target))


                        skip=False
                        for t in target.unique():
                            if (t == target).sum()==0: #this wont work with many classes + small batch size
                                skip=True
                                break

                        if skip:
                            break


                        for j in range(len(target)):
                            pos_target = target[j].item()

                            np.random.shuffle(p)
                            neg_updated = False
                            anc_updated = False
                            neg = None
                            anc = None
                            for k in p:
                                if k == j:
                                    continue

                                if (not neg_updated) and target[k].item() != pos_target:
                                    neg = k
                                    neg_updated = True

                                if (not anc_updated) and target[k].item() == pos_target:
                                    anc = k
                                    anc_updated = True
                            if neg_updated and anc_updated:
                                select.append(j)
                                ind_neg.append(neg)
                                anchor_pos.append(anc)

                        hidden = normalize(hidden)

                        ind2_neg = []
                        anchor2_pos = []
                        for j in range(len(target)):
                            pos_target = target[j].item()

                            np.random.shuffle(p)
                            neg_buf_updated = False
                            anc2_updated = False
                            for k in p:
                                if k == j:
                                    continue

                                if (not neg_buf_updated) and mem_y[k].item() != pos_target:
                                    ind2_neg.append(k)
                                    neg_buf_updated = True

                                if (not anc2_updated) and target[k].item() == pos_target:
                                    anchor2_pos.append(k)
                                    anc2_updated = True

                        if len(anchor_pos) != len(select) or len(ind_neg) != len(select):
                          #  logits = model.linear(hidden)
                           # loss = F.cross_entropy(logits, target)
                            print('wiif')
                            break

                        loss = 2.0*F.triplet_margin_loss(hidden[select],hidden[anchor_pos], hidden[ind_neg], 0.2)

                       # present = target.unique()
                        #mask[:, present] = 1
                        #mask = mask.cuda()
                        #logits = model.linear(hidden)
                        #logits = logits.masked_fill(mask == 0, -1e9)
                        #loss += F.cross_entropy(logits, target)



                        #loss2 = 0
                        #for k in range(len(logits)):
                         #   loss2 += logits[k, mem_y[k]]
                        #loss += -0.5 * loss2 / float(len(logits)) / 5.0

                        if len(anchor2_pos) != len(target) or len(ind2_neg) != len(target):
                            break

                        loss+= 0.0*F.triplet_margin_loss(hidden, hidden[anchor2_pos], normalize(hidden_buff[ind2_neg]).detach(), 0.1)
                               #+ torch.norm(normalize(hidden_buff[0:len(target)]) - normalize(hidden_buff[0:len(target)]).detach())

                        if it==0:
                            orig_buff=hidden_buff[0:len(target)].detach()
                        #else:
                            #loss+= 2.0*F.triplet_margin_loss(hidden[anchor_pos], hidden, normalize(hidden_buff[0:len(target)]), 0.1)
                           # loss+=torch.norm(hidden_buff-orig_buff)

                        #####Eval
                        debug=False
                        if debug:
                            model.eval()
                            all_logit=model(buffer.bx)/5.0
                            all_class=buffer.by.unique()
                            clases = target.unique()

                            dist = model.linear(hidden)/5.0
                            pred = dist.argmax(dim=1, keepdim=True).squeeze()
                            for clas in all_class:


                                if any(pred==clas):
                                    print('class:' + str(clas.item()))
                                    print('MISCLASSIFIED: max dist new misclassified to this class: ' + str(dist[pred==clas].max().item()))
                                    print('avg dist sample to its class: ' + str(all_logit[buffer.by == clas].mean().item()))
                                    print('std dist sample to its class: ' + str(all_logit[buffer.by == clas].std().item()))
                                    print('max dist sample to its class: ' + str(all_logit[buffer.by == clas].max().item()))
                            model.train()
                      #  pause(0.2)
                        #import ipdb; ipdb.set_trace()
                        #####

                opt.zero_grad()
                loss.backward(retain_graph=True)

                if rehearse:
                    logits_buffer = model.linear(hidden_buff)
                    if args.mask_trick:
                        present = mask_so_far
                        mask = torch.zeros_like(logits_buffer)
                        mask[:, present] = 1
                        logits_buffer = logits_buffer.masked_fill(mask == 0, -1e9)

                    loss_a = F.cross_entropy(logits_buffer, mem_y, reduction='none')
                    loss = (loss_a).sum() / loss_a.size(0)
                    loss.backward()

                model(data)

                if task>0 and False:
                    for param in model.linear.parameters():
                        param.grad[last_mask.byte()]=0

                opt.step()

                #####Just for debug
                if False and task>0:
                    if it==0:
                        h1_store = copy.deepcopy(hidden.data.cpu())
                        h_buff = copy.deepcopy(hidden_buff.data.cpu())
                    model.eval()
                    h1_store_after = model.return_hidden(data).data.cpu()
                    h_buff_after = model.return_hidden(mem_x).data.cpu()
                    hid_new_change.update(torch.norm(normalize(h1_store)-normalize(h1_store_after)).item())
                    hid_old_change.update(torch.norm(normalize(h_buff)- normalize(h_buff_after)).item())
                    print('task: '+str(task))
                    print('new classes:' + str(hid_new_change.avg))
                    print('old classes:' + str(hid_old_change.avg))
                    model.train()
                ######
            buffer.add_reservoir(data, target_orig, None, task)


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