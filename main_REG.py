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

from data_loader_REG import CMPDataIter
import matplotlib.pyplot as plt
import random
import wandb
from sweep_args import *
class Train():
    def __init__(self, args, configs, dataset):
        
        self.dataset = dataset
        self.args = args
        self.configs = configs

    def Train_batch(self):
        self.net.train()
        loss_ = 0
        batch_size = self.args.batch_size
        iter = int(self.train_data.size(0) / batch_size)
        remain = self.train_data.size(0) - iter * batch_size
        for i in range(iter):

            data = self.train_data[i * batch_size:(i + 1) * batch_size]
            label = self.train_label[i * batch_size:(i + 1) * batch_size]
            self.optim.zero_grad()
            prediction = self.net(data)
            loss = self.loss_function(prediction, label)

            loss.backward()
            self.optim.step()
            loss_ = loss_ + loss.item()

        if remain != 0:

            data = self.train_data[-remain:]
            label = self.train_label[-remain:]
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
            'metric': {'name': self.configs.metric_to_maximize, 'goal': 'minimize'},
            'parameters': {**get_sweep_hparams('C_MAPSS')[f'FD00{args.data_sub}']}
        }
        sweep_id = wandb.sweep(sweep_config, project=f'RUL_pre_FD00{args.data_sub}_fixed_seed')

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
        args_dict = {
            k: v for k, v in vars(self.args).items()
            if not k.startswith('_') and not isinstance(v, (list, dict))  # Optional: exclude complex types
        }
        runs = wandb.init(
            project="RUL-Prediction",  # Your project name
            config=args_dict,  # Log all hyperparameters
            name=f"FD00{args.data_sub}_exp",  # Experiment name
            settings=wandb.Settings(start_method="thread",init_timeout=300)
        )
        epoch = self.configs.epoch
        self.args = wandb.config
        self.args.pool_choice = 'mean'
        self.args.max_rul = 125
        self.args.kernel = [2, 2]
        self.args.stride = [1, 2]


        self.loss_function = nn.MSELoss()

        best_RMSE = []
        best_score = []
        for run_id in range(self.configs.runs):
            self.set_seed(run_id)
            data_iter = CMPDataIter('./datasets/',
                                    data_set=self.dataset,
                                    max_rul=self.args.max_rul,
                                    seq_len=self.args.time_length,
                                    net_name=1)
            self.train_data = self.cuda_(data_iter.out_x)
            self.train_ops = self.cuda_(data_iter.out_ops)
            self.train_label = self.cuda_(data_iter.out_y)

            self.val_data = self.cuda_(data_iter.cross_val_x)
            self.val_ops = self.cuda_(data_iter.cross_val_ops)
            self.val_label = self.cuda_(data_iter.cross_val_y)

            self.test_data = self.cuda_(data_iter.test_x)
            self.test_ops = self.cuda_(data_iter.test_ops)
            self.test_label = self.cuda_(data_iter.test_y)

            self.net = Model.FC_STGNN_REG(self.args.lstmhidden_dim, self.args.lstmout_dim,
                                            self.args.conv_kernel,
                                            self.args.hidden_dim, self.args.num_patch,
                                            self.args.time_length, self.args.num_sensor,
                                            self.args.kernel, self.args.stride, self.args.decay,
                                            self.args.pool_choice, 1,
                                            self.args.tem_map, self.args.patch_dim)

            self.net = self.net.cuda() if tr.cuda.is_available() else self.net
            self.optim = optim.Adam(self.net.parameters())

            test_RMSE = []
            test_score = []
            best_run_RMSE = tr.inf
            best_run_score = tr.inf
            epochs_no_improve = 0
            early_stop = False
            for i in range(epoch):
                if early_stop:
                    print(f"Early stopping at epoch {i}")
                    break
                loss = self.Train_batch()
                if i%1 == 0:
                    loss_val = self.Cross_validation()

                    test_RMSE_, test_score_, test_result_predicted, test_result_real = self.Prediction()

                    test_RMSE.append(test_RMSE_)
                    test_score.append(test_score_)

                    if test_RMSE_ < best_run_RMSE:
                        best_run_RMSE = test_RMSE_
                        best_run_score = test_score_

                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve > 20:
                            early_stop = True
                    print('In the {}th epoch, TESTING RMSE is {}, TESTING Score is {}'.format(i, best_run_RMSE, best_run_score))



            test_RMSE = np.stack(test_RMSE,0)
            test_score = np.stack(test_score,0)
            ind_best = np.argmin(test_RMSE)
            test_RMSE_best = test_RMSE[ind_best]
            test_score_best = test_score[ind_best]
            best_RMSE.append(test_RMSE_best)
            best_score.append(test_score_best)

        best_RMSE = np.stack(best_RMSE,0)
        best_score = np.stack(best_score,0)

        avg_best_RMSE = np.average(best_RMSE)
        std_best_RMSE = np.std(best_RMSE)
        avg_best_Score = np.average(best_score)
        std_best_Score = np.std(best_score)
        if best_RMSE.ndim == 1:
            df_rmse = pd.DataFrame(best_RMSE, columns=["RMSE"])
        else:
            df_rmse = pd.DataFrame(best_RMSE)

        if best_score.ndim == 1:
            df_score = pd.DataFrame(best_score, columns=["Score"])
        else:
            df_score = pd.DataFrame(best_score)
        wandb.log({
            "test_RMSE": avg_best_RMSE,
            "test_score": avg_best_Score,
            'test_RMSE_std': std_best_RMSE,
            'test_Score_std': std_best_Score,

        })

        wandb.log({'RMSE': wandb.Table(dataframe=df_rmse, allow_mixed_types=True)})
        wandb.log({'Score': wandb.Table(dataframe=df_score, allow_mixed_types=True)})
        test_results = np.stack([best_RMSE, best_score],0)
        np.save('./experiment/{}.npy'.format(self.configs.save_name),test_results)

        runs.finish()

    def cuda_(self, x):
        x = tr.Tensor(x)

        if tr.cuda.is_available():
            return x.cuda()
        else:
            return x

    def data_preprocess_transpose(self, data, ops):
        '''

        :param data: size is [bs, time_length, dimension, Num_nodes]
        :return: size is [bs, time_length, Num_nodes, dimension]
        '''

        data = tr.transpose(data,2,3)
        ops = tr.transpose(ops,2,3)

        return data, ops

    def Cross_validation(self):
        self.net.eval()
        loss_ = 0
        batch_size = self.args.batch_size
        iter = int(self.val_data.size(0) / batch_size)
        remain = self.val_data.size(0) - iter * batch_size
        for i in range(iter):
            data = self.val_data[i * batch_size:(i + 1) * batch_size]
            label = self.val_label[i * batch_size:(i + 1) * batch_size]
            prediction = self.net(data)
            loss = self.loss_function(prediction, label)
            loss_ = loss_ + loss.item()

        if remain != 0:
            data = self.val_data[-remain:]
            label = self.val_label[-remain:]
            prediction = self.net(data)
            loss = self.loss_function(prediction, label)
            loss_ = loss_ + loss.item()
        return loss_

    def Prediction(self):
        '''
        This is to predict the results for testing dataset
        :return:
        '''
        self.net.eval()
        prediction = self.net(self.test_data)
        predicted_RUL = prediction
        real_RUL = self.test_label
        MSE = self.loss_function(predicted_RUL, real_RUL)

        RMSE = tr.sqrt(MSE)*self.args.max_rul
        score = self.scoring_function(predicted_RUL, real_RUL)
        return RMSE.detach().cpu().numpy(),\
               score.detach().cpu().numpy(), \
               predicted_RUL.detach().cpu().numpy(), \
               real_RUL.detach().cpu().numpy()

    def Prediction_training(self):
        self.net.eval()
        sample_idx = random.sample(range(len(self.train_data)), 100)
        train_data_sample = self.train_data[sample_idx]
        train_label_sample = self.train_label[sample_idx]
        prediction = self.net(train_data_sample)
        MSE = self.loss_function(prediction, train_label_sample)
        RMSE = tr.sqrt(MSE) * self.args.max_rul
        score = self.scoring_function(prediction, train_label_sample)[0]
        return RMSE.detach().cpu().numpy(), \
               score.detach().cpu().numpy(), \
               prediction.detach().cpu().numpy(), \
               train_label_sample.detach().cpu().numpy()



    def scoring_function(self, predicted, real):
        score = 0
        num = predicted.size(0)
        for i in range(num):

            if real[i] > predicted[i]:
                score = score+ (tr.exp((real[i]*self.args.max_rul-predicted[i]*self.args.max_rul)/13)-1)

            elif real[i]<= predicted[i]:
                score = score + (tr.exp((predicted[i]*self.args.max_rul-real[i]*self.args.max_rul)/10)-1)

        return score
if __name__ == '__main__':
    from args import args



    args = args()
    class train_config():
        def __init__(self):
            self.num_sweeps = 200
            self.hp_search_strategy = 'random'
            self.metric_to_maximize = 'test_score'
            self.save_name = 'sweep'

            self.epoch = 40
            self.runs = 1
    
    
    
    def args_config_FD001(args):


        args.k = 1
        args.pool_choice = 'mean'
        args.tem_map = True
        args.patch_learn = True
        args.kernel = [2, 2]
        args.stride = [1, 2]
        args.decay = 0.7
        args.batch_size = 100

        args.data_sub = 1
        args.time_length = 30
        args.num_patch = 5
        args.lstmout_dim = 32
        args.hidden_dim = 32
        args.window_sample = 30

        args.lr = 0.001
        args.lstmhidden_dim = 48
        args.conv_kernel = 7
        args.tem_map = True
        args.patch_dim = 4

        return args

    def args_config_FD002(args):


        args.k = 1
        args.pool_choice = 'mean'
        args.tem_map = True
        args.patch_learn = True
        args.kernel = [2, 2]
        args.stride = [1, 2]
        args.decay = 0.7
        args.batch_size = 100


        args.data_sub = 2
        args.time_length = 50
        args.num_patch = 6
        args.lstmout_dim = 32
        args.hidden_dim = 48
        args.window_sample = 50

        args.lr = 0.01
        args.lstmhidden_dim = 64
        args.conv_kernel = 2
        args.tem_map = True
        args.patch_dim = 8


        return args
    
    def args_config_FD003(args):


        args.k = 1
        args.pool_choice = 'mean'
        args.tem_map = True
        args.patch_learn = True
        args.kernel = [2, 2]
        args.stride = [1, 2]
        args.decay = 0.7
        args.batch_size = 100

        args.data_sub = 3
        args.time_length = 50
        args.num_patch = 7
        args.lstmout_dim = 48
        args.hidden_dim = 16
        args.window_sample = 50

        args.lr = 0.001
        args.lstmhidden_dim = 64
        args.conv_kernel = 4
        args.tem_map = True
        args.patch_dim = 8


        return args
    
    def args_config_FD004(args):


        args.k = 1
        args.pool_choice = 'mean'
        args.tem_map = True
        args.patch_learn = True
        args.kernel = [2, 2]
        args.stride = [1, 2]
        args.decay = 0.7
        args.batch_size = 100


        args.data_sub = 4
        args.time_length = 50
        args.num_patch = 6
        args.lstmout_dim = 48
        args.hidden_dim = 32
        args.window_sample = 50

        args.lr = 0.0001
        args.lstmhidden_dim = 64
        args.conv_kernel = 5
        args.tem_map = True
        args.patch_dim = 4

        return args
    
    configs = train_config()
    args = args_config_FD001(args)
    train = Train(args, configs, 'FD001')

    ## Run
    train.Train_model()

    ### Sweep
    # train.sweep()
