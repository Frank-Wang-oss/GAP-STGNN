import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
def data_preparation_ISRUC(Fold_data, Fold_Label, down_sample = 10):
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    test_data = []
    test_label = []
    for i in range(len(Fold_data)):
        data_idx = Fold_data[i]
        label_idx = Fold_Label[i]
        len_idx = len(data_idx)
        num_train = int(len_idx*0.6)
        num_val = int(len_idx*0.2)
        idx = np.arange(len_idx)
        np.random.shuffle(idx)

        data_idx = data_idx[idx]
        label_idx = label_idx[idx]

        train_data.append(data_idx[:num_train])
        train_label.append(label_idx[:num_train])
        val_data.append(data_idx[num_train:(num_train+num_val)])
        val_label.append(label_idx[num_train:(num_train+num_val)])
        test_data.append(data_idx[(num_train+num_val):])
        test_label.append(label_idx[(num_train+num_val):])

    train_data = np.concatenate(train_data,0)
    train_label = np.concatenate(train_label,0)
    val_data = np.concatenate(val_data,0)
    val_label = np.concatenate(val_label,0)
    test_data = np.concatenate(test_data,0)
    test_label = np.concatenate(test_label,0)

    len_train = train_data.shape[0]
    idx = np.arange(len_train)
    np.random.shuffle(idx)
    train_data = train_data[idx]
    train_label = train_label[idx]

    train_data = train_data[:,:,::down_sample]
    train_label = np.argmax(train_label,-1)
    val_data = val_data[:,:,::down_sample]
    val_label = np.argmax(val_label,-1)
    test_data = test_data[:,:,::down_sample]
    test_label = np.argmax(test_label,-1)

    train_dataset = {"samples":train_data, "labels":train_label}
    valid_dataset = {"samples":val_data, "labels":val_label}
    test_dataset = {"samples":test_data, "labels":test_label}
    return train_dataset, valid_dataset, test_dataset

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, args):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]
        if torch.is_tensor(X_train):
            X_train = X_train.numpy()
        if torch.is_tensor(y_train):
            y_train = y_train.numpy()

        self.len = X_train.shape[0]
        self.time_length = X_train.shape[1]  # Time steps (assuming shape [samples, timesteps, features])
        self.num_sensor = X_train.shape[2]  # Number of features/dimensions
        self.n_class = len(np.unique(y_train))  # Number of classes

        X_train = self._handle_nan(X_train)


        self.x_data = torch.from_numpy(X_train).float()
        self.y_data = torch.from_numpy(y_train).long()


        
        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if self.x_data.shape[1] != args.num_sensor:
            self.x_data = torch.transpose(self.x_data, -1, -2)


        self.len = X_train.shape[0]
        shape = self.x_data.size()
        # self.x_data = self.x_data.reshape(shape[0],shape[1],args.time_denpen_len, args.patch_size)
        # self.x_data = torch.transpose(self.x_data, 1,2)

    def _handle_nan(self, data):
        """Replace NaN values with channel means"""
        if np.isnan(data).any():
            print("Warning: NaN values detected - replacing with channel means")
            # Calculate mean ignoring NaNs
            channel_means = np.nanmean(data, axis=(0, 2), keepdims=True)
            # Replace NaNs with channel means
            nan_mask = np.isnan(data)
            data[nan_mask] = np.take(channel_means, np.where(nan_mask)[1])
        return data
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, args, dataset):

    if dataset == 'ISRUC':
        path = path+'/ISRUC_S3.npz'
        ReadList = np.load(path, allow_pickle=True)
        Fold_Num = ReadList['Fold_len']  # Num of samples of each fold
        Fold_Data = ReadList['Fold_data']  # Data of each fold
        Fold_Label = ReadList['Fold_label']  # Labels of each fold

        train_dataset, valid_dataset, test_dataset = data_preparation_ISRUC(Fold_Data,Fold_Label)

    else:
        train_dataset = torch.load(os.path.join(data_path, "train.pt"))
        valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
        test_dataset = torch.load(os.path.join(data_path, "test.pt"))

    train_dataset = Load_Dataset(train_dataset, args)
    valid_dataset = Load_Dataset(valid_dataset, args)
    test_dataset = Load_Dataset(test_dataset, args)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                               shuffle=True, drop_last=args.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size,
                                               shuffle=False, drop_last=args.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader
