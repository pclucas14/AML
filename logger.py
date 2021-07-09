import os
import sys
import pickle
import numpy as np

from datetime import datetime

class Logger:
    def __init__(self, args, base_dir='outputs', save_every=30):

        self.step = 0
        self.args = args
        self.save_every = 30

        if args.wandb_log == 'online':
            import wandb
            wandb.init(project=args.wandb_project, name=args.exp_name, config=args)
            self.wandb = wandb
        else:
            self.wandb = None

        # create directory to store real logs
        date, time = datetime.now().strftime("%d_%m_%Y %H_%M_%S").split(' ')
        self.path  = os.path.join(base_dir, date, time + f'_{np.random.randint(1000)}')
        os.makedirs(self.path)

        # dump args
        f = open(os.path.join(self.path, "params.json"), "wt")
        f.write(str(args) + "\n")
        f.close()

        self.to_pickle  = []
        self.picklename = os.path.join(self.path,  "db.pickle")


    def register_name(self, name):
        if self.wandb is not None:
            self.wandb.config.update({'unique_name': name})



    def log_scalars(self, values):

        for k,v in values.items():
            self.to_pickle += [(k, v, self.step)]

        if self.wandb is not None:
            self.wandb.log(values, step=self.step)

        if self.step % self.save_every == 0:
            self.dump()

        self.step += 1


    def log_matrix(self, name, value):
        self.to_pickle += [(name, value, self.step)]

        self.step += 1


    def dump(self):
        f = open(self.picklename, "ab")
        pickle.dump(self.to_pickle, f)
        f.close()
        self.to_pickle = []


    def close(self):
        if self.wandb is not None:
            self.wandb.finish()

        self.dump()


