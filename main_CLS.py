from data_loader_CLS import data_generator

import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import os

import Model
import pandas as pd

import argparse
import matplotlib.pyplot as plt
import random
import wandb
from sweep_args import *
from sklearn.metrics import accuracy_score, f1_score
import gc

class Train():
    def __init__(self, args, configs, dataset):

        self.dataset = dataset
        self.args = args
        self.configs = configs


    def Train_batch(self):
        self.net.train()
        loss_ = 0
        for data, label in self.train:
            # print(data)
            data = data.cuda() if tr.cuda.is_available() else data
            label = label.cuda() if tr.cuda.is_available() else label
            self.optim.zero_grad()
            prediction = self.net(data)
            loss = self.loss_function(prediction, label)
            loss.backward()
            self.optim.step()
            loss_ = loss_ + loss.item()
        return loss_

    def sweep(self):
        # sweep configurations

        sweep_runs_count = self.configs.num_sweeps
        sweep_config = {
            'method': self.configs.hp_search_strategy,
            'metric': {'name': self.configs.metric_to_maximize, 'goal': 'maximize'},
            'parameters': {**get_sweep_hparams(self.dataset)}
        }
        sweep_id = wandb.sweep(sweep_config, project=f'{self.dataset}_fixed_seed')

        wandb.agent(sweep_id, self.Train_model, count=sweep_runs_count)  # Training with sweep
    def set_seed(self,seed=42):
        """Set random seed for reproducibility across multiple libraries"""
        # Python random module
        random.seed(seed)

        # Numpy
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        # PyTorch
        tr.manual_seed(seed)
        tr.cuda.manual_seed(seed)
        tr.cuda.manual_seed_all(seed)  # if using multi-GPU
        tr.backends.cudnn.deterministic = True
        tr.backends.cudnn.benchmark = False
        # tr.use_deterministic_algorithms(True)
        # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        # os.environ['PYTHONHASHSEED'] = '0'
        print(f"Random seed set to {seed} for all modules")
    def Train_model(self):
        def cleanup():
            if tr.cuda.is_available():
                tr.cuda.empty_cache()
            gc.collect()
        args_dict = {
            k: v for k, v in vars(self.args).items()
            if not k.startswith('_') and not isinstance(v, (list, dict))  # Optional: exclude complex types
        }
        # print(args_dict['kernel'])
        runs = wandb.init(
            project=self.dataset,  # Your project name
            config=args_dict,  # Log all hyperparameters
            name=self.dataset,  # Experiment name
            settings=wandb.Settings(start_method="thread",init_timeout=300)
        )
        epoch = self.configs.epoch
        self.args = wandb.config
        self.args.kernel = [2, 2]
        self.args.stride = [1, 2]
        self.args.pool_choice = 'mean'
        self.loss_function = nn.CrossEntropyLoss()


        best_accu = []
        best_mf1 = []
        for run_id in range(self.configs.runs):
            self.set_seed(run_id)

            self.train, self.valid, self.test = data_generator(f'./datasets/{self.dataset}', self.args, self.dataset)

            self.net = Model.GAP_STGNN_CLS(self.args.lstmhidden_dim, self.args.lstmout_dim,
                                            self.args.conv_kernel,
                                            self.args.hidden_dim, self.args.num_patch,
                                            self.args.time_length, self.args.num_sensor,
                                            self.args.kernel, self.args.stride,
                                            self.args.decay, self.args.pool_choice,
                                            self.args.n_class,
                                            self.args.tem_map, self.args.patch_dim)

            self.net = self.net.cuda() if tr.cuda.is_available() else self.net
            self.optim = optim.Adam(self.net.parameters())

            test_accu_ls = []
            test_mf1_ls = []
            best_run_accu = 0.0
            best_run_mf1 = 0.0
            epochs_no_improve = 0
            early_stop = False
            for i in range(epoch):
                if early_stop:
                    print(f"Early stopping at epoch {i}")
                    break
                loss = self.Train_batch()
                if i%1 == 0:
                    test_accu,test_mf1, prediction, real = self.Prediction()

                    test_accu_ls.append(test_accu)
                    test_mf1_ls.append(test_mf1)

                    if test_accu > best_run_accu:
                        best_run_accu = test_accu
                        best_run_mf1 = test_mf1
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve > 50:
                            early_stop = True
                    print('In the {}th epoch, TESTING accuracy and MF1 are {} and {}'.format(i, np.round(best_run_accu, 3),np.round(best_run_mf1, 3)))

            test_accu_ls = np.stack(test_accu_ls, 0)
            test_mf1_ls = np.stack(test_mf1_ls, 0)
            ind_best = np.argmax(test_accu_ls)
            test_accu_best = test_accu_ls[ind_best]
            test_mf1_best = test_mf1_ls[ind_best]
            best_accu.append(test_accu_best)
            best_mf1.append(test_mf1_best)

        best_accu = np.stack(best_accu, 0)
        best_mf1 = np.stack(best_mf1, 0)

        avg_best_accu = np.average(best_accu)
        avg_best_mf1 = np.average(best_mf1)

        std_best_accu = np.std(best_accu)
        std_best_mf1 = np.std(best_mf1)
        if best_accu.ndim == 1:
            df_accu = pd.DataFrame(best_accu, columns=["accuracy"])
        else:
            df_accu = pd.DataFrame(best_accu)

        if best_mf1.ndim == 1:
            df_mf1 = pd.DataFrame(best_mf1, columns=["mf1"])
        else:
            df_mf1 = pd.DataFrame(best_mf1)
        wandb.log({
            "test_accu": avg_best_accu,
            "test_mf1": avg_best_mf1,
            'test_accu_std': std_best_accu,
            'test_mf1_std': std_best_mf1,
        })
        wandb.log({'Accu': wandb.Table(dataframe=df_accu, allow_mixed_types=True)})
        wandb.log({'MF1': wandb.Table(dataframe=df_mf1, allow_mixed_types=True)})

        test_results = np.stack([best_accu, best_mf1], 0)
        np.save('./experiment/{}.npy'.format(self.configs.save_name), test_results)
        runs.finish()

    def cuda_(self, x):
        x = tr.Tensor(np.array(x))
        # x = tr.Tensor(x)

        if tr.cuda.is_available():
            return x.cuda()
        else:
            return x



    def Prediction(self):
        '''
        This is to predict the results for testing dataset
        :return:
        '''
        self.net.eval()
        prediction_ = []
        real_ = []
        for data, label in self.test:
            data = data.cuda() if tr.cuda.is_available() else data
            real_.append(label)
            prediction = self.net(data)
            prediction_.append(prediction.detach().cpu())
        predicted_classes = tr.cat(prediction_, 0)
        real_labels = tr.cat(real_, 0)

        predicted_classes = tr.argmax(predicted_classes, -1)

        accuracy = accuracy_score(real_labels.numpy(), predicted_classes.numpy())*100

        f1_macro = f1_score(real_labels.numpy(), predicted_classes.numpy(), average='macro')*100

        return accuracy, f1_macro, prediction_, real_


if __name__ == '__main__':
    from args import args

    args = args()
    class train_config():
        def __init__(self):
            self.num_sweeps = 400
            self.hp_search_strategy = 'random'
            self.metric_to_maximize = 'accuracy'
            self.save_name = 'sweep'

            self.epoch = 100
            self.runs = 1

    def args_config_HAR(args):

        args.k = 1
        args.pool_choice = 'mean'
        args.tem_map = False
        args.num_sensor = 9

        args.time_length = 128
        args.num_patch = 32
        args.lstmout_dim = 32
        args.hidden_dim = 48
        args.window_sample = 128
        args.kernel = [2, 2]
        args.stride = [1, 2]
        args.lr = 0.001
        args.lstmhidden_dim = 16
        args.conv_kernel = 7
        args.batch_size = 200
        args.decay = 0.78164
        args.n_class = 6
        args.patch_dim = 4

        return args

    def args_config_SSC(args):

        args.k = 1
        args.pool_choice = 'mean'
        args.tem_map = False
        args.num_sensor = 10

        args.time_length = 300
        args.num_patch = 4
        args.lstmout_dim = 16
        args.hidden_dim = 96
        args.window_sample = 300
        args.kernel = [2, 2]
        args.stride = [1, 2]
        args.lr = 0.00001
        args.lstmhidden_dim = 72
        args.conv_kernel = 26
        args.batch_size = 150
        args.decay = 0.5655
        args.n_class = 5
        args.patch_dim = 8


        return args



    def args_config_WISDM(args):

        args.k = 1
        args.pool_choice = 'mean'
        args.tem_map = False

        args.num_sensor = 3

        args.time_length = 128
        args.num_patch = 8
        args.lstmout_dim = 16
        args.hidden_dim = 8
        args.window_sample = 300
        args.kernel = [2, 2]
        args.stride = [1, 2]
        args.lr = 0.001
        args.lstmhidden_dim = 64
        args.conv_kernel = 12
        args.batch_size = 100
        args.decay = 0.48384
        args.n_class = 6
        args.patch_dim = 16

        return args


    def args_config_CharacterTrajectories(args):

        args.k = 1
        args.pool_choice = 'mean'
        args.tem_map = False

        args.num_sensor = 3

        args.time_length = 180
        args.num_patch = 64
        args.lstmout_dim = 32
        args.hidden_dim = 96
        args.window_sample = 300
        args.kernel = [2, 2]
        args.stride = [1, 2]
        args.lr = 0.03
        args.lstmhidden_dim = 32
        args.conv_kernel = 8
        args.batch_size = 100
        args.decay = 0.87968
        args.n_class = 20
        args.patch_dim = 16


        return args
    def args_config_SelfRegulationSCP1(args):

        args.k = 1
        args.pool_choice = 'mean'
        args.tem_map = True

        args.num_sensor = 6

        args.time_length = 224
        args.num_patch = 16
        args.lstmout_dim = 48
        args.hidden_dim = 64
        args.window_sample = 300
        args.kernel = [2, 2]
        args.stride = [1, 2]
        args.lr = 0.001
        args.lstmhidden_dim = 16
        args.conv_kernel = 8
        args.batch_size = 250
        args.decay = 0.5108
        args.n_class = 2
        args.patch_dim = 32


        return args
    def args_config_FingerMovements(args):

        args.k = 1
        args.pool_choice = 'mean'
        args.tem_map = False

        args.num_sensor = 28

        args.time_length = 50
        args.num_patch = 16
        args.lstmout_dim = 48
        args.hidden_dim = 48
        args.window_sample = 300
        args.kernel = [2, 2]
        args.stride = [1, 2]
        args.lr = 0.0005

        args.lstmhidden_dim = 48
        args.conv_kernel = 10
        args.batch_size = 200
        args.decay = 0.19279

        args.n_class = 2
        args.patch_dim = 4


        return args

    def args_config_Cricket(args):

        args.k = 1
        args.pool_choice = 'mean'
        args.tem_map = False

        args.num_sensor = 6

        args.time_length = 1197
        args.num_patch = 32
        args.lstmout_dim = 40
        args.hidden_dim = 48
        args.window_sample = 300
        args.kernel = [2, 2]
        args.stride = [1, 2]
        args.lr = 0.0007

        args.lstmhidden_dim = 8
        args.conv_kernel = 6

        args.batch_size = 100
        args.decay = 0.41684

        args.n_class = 12
        args.patch_dim = 32


        return args
    def args_config_Epilepsy(args):

        args.k = 1
        args.pool_choice = 'mean'
        args.tem_map = False
        args.num_sensor = 3

        args.time_length = 206
        args.num_patch = 8
        args.lstmout_dim = 48
        args.hidden_dim = 16
        args.window_sample = 300
        args.kernel = [2, 2]
        args.stride = [1, 2]
        args.lr = 0.0005
        args.lstmhidden_dim = 64
        args.conv_kernel = 10
        args.batch_size = 150
        args.decay = 0.8224
        args.n_class = 4
        args.patch_dim = 16


        return args
    def args_config_ERing(args):

        args.k = 1
        args.pool_choice = 'mean'
        args.tem_map = False

        args.num_sensor = 4

        args.time_length = 65
        args.num_patch = 64
        args.lstmout_dim = 16
        args.hidden_dim = 96
        args.window_sample = 300
        args.kernel = [2, 2]
        args.stride = [1, 2]
        args.lr = 0.01
        args.lstmhidden_dim = 16
        args.conv_kernel = 2
        args.batch_size = 150
        args.decay = 0.10794

        args.n_class = 6
        args.patch_dim = 4


        return args

    def args_config_SpokenArabicDigits(args):

        args.k = 1
        args.pool_choice = 'mean'
        args.tem_map = False

        args.num_sensor = 13

        args.time_length = 93
        args.num_patch = 32
        args.lstmout_dim = 16
        args.hidden_dim = 64
        args.window_sample = 300
        args.kernel = [2, 2]
        args.stride = [1, 2]
        args.lr = 0.0005


        args.lstmhidden_dim = 64
        args.conv_kernel = 10
        args.batch_size = 100
        args.decay = 0.67203


        args.n_class = 10
        args.patch_dim = 16


        return args
    

    configs = train_config()
    
    args = args_config_HAR(args)

    train = Train(args, configs, 'HAR')

    ## Run
    train.Train_model()

    ### Sweep
    # train.sweep()