import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from copy import deepcopy
from data   import *
from utils  import get_logger, get_temp_logger, logging_per_task, sho_
from buffer import Buffer
from copy   import deepcopy
from pydoc  import locate
from model  import ResNet18, normalize
from methods import get_method
from utils import naive_cross_entropy_loss, onehot, Lookahead, AverageMeter

# Arguments
# -----------------------------------------------------------------------------------------
METHODS = ['icarl', 'er', 'mask', 'triplet', 'iid', 'iid++', 'icarl_mask']

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default = 'split_cifar10',
    choices=['split_mnist', 'permuted_mnist', 'split_cifar10', 'split_cifar100', 'miniimagenet'])
parser.add_argument('--n_tasks', type=int, default=-1,
    help='total number of tasks. -1 does default amount for the dataset')
parser.add_argument('-r','--reproc', type=int, default=0,
    help='if on, no randomness in numpy and torch')
parser.add_argument('--disc_iters', type=int, default=1,
    help='number of training iterations for the classifier')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--buffer_batch_size', type=int, default=10)
parser.add_argument('--samples_per_task', type=int, default=-1,
    help='if negative, full dataset is used')
parser.add_argument('--mem_size', type=int, default=20, help='controls buffer size')
parser.add_argument('--n_runs', type=int, default=1,
    help='number of runs to average performance')
# logging
parser.add_argument('-l', '--log', type=str, default='off', choices=['off', 'online'],
    help='enable WandB logging')
parser.add_argument('--wandb_project', type=str, default='er_imbalance',
    help='name of the WandB project')
parser.add_argument('--exp_name', type=str, default='tmp')

#------ MIR -----#
parser.add_argument('-m','--method', type=str, default='er', choices=METHODS)
parser.add_argument('--lr', type=float, default=0.1)

parser.add_argument('--incoming_neg', type=float, default=2.0)
parser.add_argument('--buffer_neg', type=float, default=0)
parser.add_argument('--margin', type=float, default=0.2)
parser.add_argument('--use_augmentations', type=int, default=0)

parser.add_argument('--distill_coef', type=int, default=1)
parser.add_argument('--task_free', type=int, default=0)
args = parser.parse_args()

if args.method in ['iid', 'iid++']:
    print('overwriting args for iid setup')
    args.n_tasks = 1
    args.mem_size = 1

# ICarl Wrapper
# ----------------------------------------------------------------------------------------
class ICarl(nn.Module):
    def __init__(self, net, gamma=0.9):
        super().__init__()

        self.net = net
        dev = next(net.parameters()).device
        # we will just use the last final connected layer as class prototypes
        self.arr = torch.arange(self.net.linear.L.weight.size(1)).to(dev)
        self.dev = dev

        self.gamma = gamma

    def update_proto(self, hid, y):
        # normalize feature vectors
        hid = F.normalize(hid, p=2, dim=-1)
        D   = hid.size(-1)

        old_proto = self.net.linear.L.weight
        n_classes = old_proto.size(0)

        correct = hid.new_zeros(size=(n_classes, D))
        for i in range(10):
            correct[i] += hid[y == i].sum(0)

        # let's just do it in one-d
        out_idx = self.arr.view(1, -1) + y.view(-1, 1) * D

        buf = hid.new_zeros(size=(n_classes, hid.size(-1)))
        buf = buf.flatten().scatter_add(0, out_idx.flatten(), hid.flatten()).view_as(buf)

        # update the ones we saw in memory
        update_mask = buf.new_zeros(size=(n_classes,), dtype=torch.bool, device=self.dev)
        update_mask[y.unique()] = 1
        update_mask = update_mask.view(-1, 1)

        new_proto = old_proto * self.gamma + buf * (1 - self.gamma)
        new_proto = update_mask * new_proto + ~update_mask * old_proto

        new_proto = F.normalize(new_proto, p=2, dim=-1)

        # set new points
        self.net.linear.L.weight.data.copy_(new_proto.data)


    def get_logits(self, hid):
        # n_c x d
        proto = self.net.linear.L.weight

        #b_s x d for hid

        # b_s x n_c
        dist = (proto.unsqueeze(0) - hid.unsqueeze(1)).pow(2).sum(-1)

        return -dist

    def forward(self, x):
        hid = self.net.return_hidden(x)
        return self.get_logits(hid)


# Obligatory overhead
# -----------------------------------------------------------------------------------------
if args.method == 'triplet':
    args.task_free = 1

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
    wandb.init(project=args.wandb_project, name=args.exp_name, config=args)
    wandb.config.update(args)
else:
    wandb = None

# create logging containers
LOG = get_logger(['cls_loss', 'acc'],
        n_runs=args.n_runs, n_tasks=args.n_tasks)

args.mem_size = args.mem_size*args.n_classes #convert from per class to total memory

# Eval model
# -----------------------------------------------------------------------------------------
@torch.no_grad()
def eval_agent(agent, loader, task, mode='valid'):
    agent.eval()
    y_pred =[]
    y_truth=[]
    for task_t, te_loader in enumerate(loader):
        if task_t > task: break
        LOG_temp = get_temp_logger(None, ['cls_loss', 'acc'])

        # iterate over samples from task
        for i, (data, target, _) in enumerate(te_loader):
            if args.cuda:
                data, target = data.to(args.device), target.to(args.device)

            logits = agent.predict(data)

            if args.multiple_heads:
                logits = logits.masked_fill(te_loader.dataset.mask == 0, -1e9)

            try:
                loss = F.cross_entropy(logits, target)
            except:
                import pdb; pdb.set_trace()
                xx = 1
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

    buffer = Buffer(args).to(args.device)
    if run == 0:
        print("number of classifier parameters:",
                sum([np.prod(p.size()) for p in model.parameters()]))
        print("buffer parameters: ", np.prod(buffer.bx.size()))

    # Build Method wrapper
    agent = get_method(args.method)(model, buffer, args)

    mask_so_far = None
    #----------
    # Task Loop
    for task, tr_loader in enumerate(train_loader):
        sample_amt = 0
        agent.train()

        if args.method == 'iid++':
            other_tr_loader = iter(torch.utils.data.DataLoader(tr_loader.dataset, batch_size=args.batch_size, num_workers=8))

        #---------------
        # Minibatch Loop
        for i, (data, target, idx) in enumerate(tr_loader):
            if sample_amt > args.samples_per_task > 0: break
            sample_amt += data.size(0)

            if args.cuda:
                idx = idx.to(args.device)
                data = data.to(args.device)
                target = target.to(args.device)

            if args.method == 'triplet':
                buffer.add_reservoir(data, target, None, task, idx, overwrite=False)

            #------ Train Classifier-------#
            if i==0:
                print('\n--------------------------------------')
                print('Run #{} Task #{} --> Train Classifier'.format(
                    run, task))
                print('--------------------------------------\n')

            #---------------
            # Iteration Loop
            for it in range(args.disc_iters):
                rehearse = task > 0 #(task + i) > 0 if args.task_free else task > 0
                rehearse = rehearse and ~(args.method in ['iid', 'iid++'])

                loss, loss_re = agent.observe(data, target, task, idx, rehearse=rehearse)

                opt.zero_grad()
                (loss + loss_re).backward()
                opt.step()

                '''
                if args.method == 'iid++':
                    try:
                        x2, y2, _ = next(other_tr_loader)
                    except StopIteration:
                        other_tr_loader = iter(torch.utils.data.DataLoader(tr_loader.dataset, batch_size=args.batch_size, num_workers=8))
                        x2, y2, _ = next(other_tr_loader)

                    x2, y2 = x2.to(args.device), y2.to(args.device)
                    F.cross_entropy(model(x2), y2).backward()
                '''

                '''
                if args.method == 'icarl':
                    hids, ys = hidden, target
                    if rehearse:
                        hids = torch.cat((hids, hidden_buff))
                        ys   = torch.cat((ys, mem_y))

                    icarl.update_proto(hids, ys)
                '''

            if args.method == 'triplet':
                buffer.balance_memory()
            else:
                buffer.add_reservoir(data, target, None, task, idx)

        # eval_model(icarl if args.method == 'icarl' else model, val_loader, task, model='valid')
        eval_agent(agent, test_loader, task, mode='test')


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
