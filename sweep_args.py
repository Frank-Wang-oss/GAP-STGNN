def get_sweep_hparams(dataset_name):
    """Return the dataset object (class, dict, etc.) with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]
C_MAPSS = {
    'FD001': {
        'num_patch': {'values': [3, 4, 5, 6, 7, 8, 9, 10, 11]},
        'lstmout_dim': {'values': [8,16,32,48,64, 72, 96]},
        'hidden_dim': {'values': [16,32,48,64, 72]},
        'lr': {'values': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]},
        'lstmhidden_dim': {'values': [16,32,48,64, 72]},
        'conv_kernel': {'values': [2,3,4,5,6,7,8]},
        'decay': {'distribution': 'uniform', 'min': 0, 'max': 1},
        'tem_map': {'values': [True, False]},
        'batch_size': {'values': [150,100,200]},
        'time_length':{'values': [30]},
        'num_sensor': {'values': [14]},
        'kernel': {'values': [[2, 2]]},
        'stride': {'values': [[1, 2]]},
        'data_sub': {'values': [1]},
        'patch_dim': {'values': [2, 4, 6, 8, 16, 32]},

    },
    'FD002': {
        'num_patch': {'values': [3, 4, 5, 6, 7, 8, 9]},
        'lstmout_dim': {'values': [8, 16, 32, 48, 64]},
        'hidden_dim': {'values': [32, 48, 64, 72]},
        'lr': {'values': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]},
        'lstmhidden_dim': {'values': [16, 32, 48, 64]},
        'conv_kernel': {'values': [2, 3, 4, 5, 6, 7, 8]},
        'decay': {'distribution': 'uniform', 'min': 0, 'max': 1},
        'tem_map': {'values': [True, False]},
        'batch_size': {'values': [150, 100, 200]},
        'time_length': {'values': [50]},
        'num_sensor': {'values': [14]},
        'kernel': {'values': [[2, 2]]},
        'stride': {'values': [[1, 2]]},
        'data_sub': {'values': [2]},
        'patch_dim': {'values': [4, 8, 16, 32]},

    },
    'FD003': {
        'num_patch': {'values': [3, 4, 5, 6, 7, 8, 9]},
        'lstmout_dim': {'values': [16, 32, 48, 64]},
        'hidden_dim': {'values': [16, 32, 48, 64]},
        'lr': {'values': [5e-3, 1e-3, 5e-4, 1e-4]},
        'lstmhidden_dim': {'values': [8, 16, 32, 48, 64]},
        'conv_kernel': {'values': [2, 3, 4, 5, 6, 7, 8]},
        'decay': {'distribution': 'uniform', 'min': 0, 'max': 1},
        'tem_map': {'values': [True, False]},
        'batch_size': {'values': [150, 100, 200]},
        'time_length': {'values': [50]},
        'num_sensor': {'values': [14]},
        'kernel': {'values': [[2, 2]]},
        'stride': {'values': [[1, 2]]},
        'data_sub': {'values': [3]},
        'patch_dim': {'values': [4, 8, 16, 32]},

    },
    'FD004': {
        'num_patch': {'values': [3, 4, 5, 6, 7, 8, 9]},
        'lstmout_dim': {'values': [8, 16, 32, 48, 64]},
        'hidden_dim': {'values': [8, 16, 32, 48, 64]},
        'lr': {'values': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 7e-5]},
        'lstmhidden_dim': {'values': [16, 32, 48, 64, 72]},
        'conv_kernel': {'values': [2, 3, 4, 5, 6, 7, 8]},
        'decay': {'distribution': 'uniform', 'min': 0, 'max': 1},
        'tem_map': {'values': [True, False]},
        'batch_size': {'values': [150, 100, 200, 250]},
        'time_length': {'values': [50]},
        'num_sensor': {'values': [14]},
        'kernel': {'values': [[2, 2]]},
        'stride': {'values': [[1, 2]]},
        'data_sub': {'values': [4]},
        'patch_dim': {'values': [2, 4, 6, 8, 16, 32]},

    },
}


HAR = {
        'num_patch': {'values': [3, 4, 6, 8, 10, 16, 32, 64]},
        'lstmout_dim': {'values': [8,16,32,48,64]},
        'hidden_dim': {'values': [8,16,32,48,64]},
        'lr': {'values': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]},
        'lstmhidden_dim': {'values': [8,16,32,48,64]},
        'conv_kernel': {'values': [2,3,4,5,6,7,8, 10, 12, 16]},
        'decay': {'distribution': 'uniform', 'min': 0, 'max': 1},
        'tem_map': {'values': [True, False]},
        'batch_size': {'values': [150,100,200]},
        'time_length':{'values': [128]},
        'num_sensor': {'values': [9]},
        'kernel': {'values': [[2, 2]]},
        'stride': {'values': [[1, 2]]},
        'patch_dim': {'values': [4, 8, 16, 32]},
        'n_class':{'values': [6]}

    }

SSC = {
        'num_patch': {'values': [3, 4, 5, 6, 8, 10, 12, 14, 16]},
        'lstmout_dim': {'values': [16,32,64,96, 128]},
        'hidden_dim': {'values': [64,96,128, 148, 196]},
        'lstmhidden_dim': {'values': [48, 64, 72, 96, 128]},
        'lr': {'values': [5e-3, 1e-3, 5e-4, 1e-4,3e-5,1e-5,7e-6]},
        'conv_kernel': {'values': [14, 16, 18, 20, 22, 24, 26]},
        'decay': {'distribution': 'uniform', 'min': 0, 'max': 1},
        'tem_map': {'values': [True, False]},
        'batch_size': {'values': [150]},
        'time_length':{'values': [300]},
        'num_sensor': {'values': [10]},
        'kernel': {'values': [[2, 2]]},
        'stride': {'values': [[1, 2]]},
        'patch_dim': {'values': [4, 8, 16, 32]},
        'n_class': {'values': [5]}

}

WISDM = {
        'num_patch': {'values': [3, 4, 6, 8, 10, 16, 32, 64]},
        'lstmout_dim': {'values': [8,16,32,48,64]},
        'hidden_dim': {'values': [8,16,32,48,64]},
        'lr': {'values': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]},
        'lstmhidden_dim': {'values': [8,16,32,48,64]},
        'conv_kernel': {'values': [2,3,4,5,6,7,8, 10, 12, 16]},
        'decay': {'distribution': 'uniform', 'min': 0, 'max': 1},
        'tem_map': {'values': [True, False]},
        'batch_size': {'values': [150,100,200]},
        'time_length':{'values': [128]},
        'num_sensor': {'values': [3]},
        'kernel': {'values': [[2, 2]]},
        'stride': {'values': [[1, 2]]},
        'patch_dim': {'values': [4, 8, 16, 32]},
        'n_class':{'values': [6]}

    }

CharacterTrajectories = {
        'num_patch': {'values': [6, 8, 10, 12, 14, 16, 18, 20, 32, 48,64]},
        'lstmout_dim': {'values': [8,16,20,32,48,64,72]},
        'hidden_dim': {'values': [32,48,64, 72, 84, 96, 112, 128]},
        'lr': {'values': [3e-2, 1e-2, 8e-3, 5e-3, 2e-3, 1e-3, 5e-4, 3e-4, 1e-4]},
        'lstmhidden_dim': {'values': [8,16,32,40,48,64,78,96]},
        'conv_kernel': {'values': [4,5,6,7,8, 10, 12, 14, 16,18]},
        'decay': {'distribution': 'uniform', 'min': 0, 'max': 1},
        'tem_map': {'values': [True, False]},
        'batch_size': {'values': [150,100,200]},
        'time_length':{'values': [180]},
        'num_sensor': {'values': [3]},
        'kernel': {'values': [[2, 2]]},
        'stride': {'values': [[1, 2]]},
        'patch_dim': {'values': [4, 8, 16, 32]},
        'n_class':{'values': [20]}

    }

SelfRegulationSCP1 = {
        'num_patch': {'values': [3, 4, 6, 8, 10, 16, 32]},
        'lstmout_dim': {'values': [8,16,32,40, 48,64,72, 96]},
        'hidden_dim': {'values': [16,32,48,64, 72, 96]},
        'lr': {'values': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]},
        'lstmhidden_dim': {'values': [8,16,32,48,64, 72, 96]},
        'conv_kernel': {'values': [3,4,5,6,7,8, 9,10,11, 12, 16]},
        'decay': {'distribution': 'uniform', 'min': 0, 'max': 1},
        'tem_map': {'values': [True, False]},
        'batch_size': {'values': [150,100,200, 250]},
        'time_length':{'values': [224]},
        'num_sensor': {'values': [6]},
        'kernel': {'values': [[2, 2]]},
        'stride': {'values': [[1, 2]]},
        'patch_dim': {'values': [4, 8, 16, 32]},
        'n_class':{'values': [2]}

    }

FingerMovements = {
        'num_patch': {'values': [3, 4, 6, 8, 10, 16, 32, 64]},
        'lstmout_dim': {'values': [8,16,32,48,64]},
        'hidden_dim': {'values': [8,16,32,48,64]},
        'lr': {'values': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]},
        'lstmhidden_dim': {'values': [8,16,32,48,64]},
        'conv_kernel': {'values': [2,3,4,5,6,7,8, 10, 12, 16]},
        'decay': {'distribution': 'uniform', 'min': 0, 'max': 1},
        'tem_map': {'values': [True, False]},
        'batch_size': {'values': [150,100,200]},
        'time_length':{'values': [50]},
        'num_sensor': {'values': [28]},
        'kernel': {'values': [[2, 2]]},
        'stride': {'values': [[1, 2]]},
        'patch_dim': {'values': [4, 8, 16, 32]},
        'n_class':{'values': [2]}

    }


Cricket = {
        'num_patch': {'values': [4, 6, 7, 8, 9, 10, 12, 16, 32]},
        'lstmout_dim': {'values': [8,16,32,40, 48]},
        'hidden_dim': {'values': [8,16,32,48]},
        'lr': {'values': [1e-2, 5e-3, 1e-3,7e-4, 5e-4, 1e-4]},
        'lstmhidden_dim': {'values': [8,16,24,32,48,64]},
        'conv_kernel': {'values': [2,3,4,5,6,7,8,9, 10, 12, 14, 16]},
        'decay': {'distribution': 'uniform', 'min': 0, 'max': 1},
        'tem_map': {'values': [True, False]},
        'batch_size': {'values': [100]},
        'time_length':{'values': [1197]},
        'num_sensor': {'values': [6]},
        'kernel': {'values': [[2, 2]]},
        'stride': {'values': [[1, 2]]},
        'patch_dim': {'values': [4, 8, 16, 32]},
        'n_class':{'values': [12]}

    }

Epilepsy = {
        'num_patch': {'values': [3, 4, 6, 8, 10, 16, 32, 64]},
        'lstmout_dim': {'values': [8,16,32,48,64]},
        'hidden_dim': {'values': [8,16,32,48,64]},
        'lr': {'values': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]},
        'lstmhidden_dim': {'values': [8,16,32,48,64]},
        'conv_kernel': {'values': [2,3,4,5,6,7,8, 10, 12, 16]},
        'decay': {'distribution': 'uniform', 'min': 0, 'max': 1},
        'tem_map': {'values': [True, False]},
        'batch_size': {'values': [150,100,200]},
        'time_length':{'values': [206]},
        'num_sensor': {'values': [3]},
        'kernel': {'values': [[2, 2]]},
        'stride': {'values': [[1, 2]]},
        'patch_dim': {'values': [4, 8, 16, 32]},
        'n_class':{'values': [4]}

    }

ERing = {
        'num_patch': {'values': [4, 6, 8, 10, 16, 32, 48,64,72]},
        'lstmout_dim': {'values': [8,16,24,32,48,64]},
        'hidden_dim': {'values': [16,32,48,64,72,96]},
        'lr': {'values': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]},
        'lstmhidden_dim': {'values': [8,16,24,32,48]},
        'conv_kernel': {'values': [2,3,4,5,6,7,8, 10, 12]},
        'decay': {'distribution': 'uniform', 'min': 0, 'max': 1},
        'tem_map': {'values': [True, False]},
        'batch_size': {'values': [150,125,200,250]},
        'time_length':{'values': [65]},
        'num_sensor': {'values': [4]},
        'kernel': {'values': [[2, 2]]},
        'stride': {'values': [[1, 2]]},
        'patch_dim': {'values': [4, 8, 16, 32]},
        'n_class':{'values': [6]}

    }


SpokenArabicDigits = {
        'num_patch': {'values': [3, 4, 6, 8, 10, 16, 32, 64]},
        'lstmout_dim': {'values': [8,16,32,48,64]},
        'hidden_dim': {'values': [8,16,32,48,64]},
        'lr': {'values': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]},
        'lstmhidden_dim': {'values': [8,16,32,48,64]},
        'conv_kernel': {'values': [2,3,4,5,6,7,8, 10, 12, 16]},
        'decay': {'distribution': 'uniform', 'min': 0, 'max': 1},
        'tem_map': {'values': [True, False]},
        'batch_size': {'values': [150,100,200]},
        'time_length':{'values': [93]},
        'num_sensor': {'values': [13]},
        'kernel': {'values': [[2, 2]]},
        'stride': {'values': [[1, 2]]},
        'patch_dim': {'values': [4, 8, 16, 32]},
        'n_class':{'values': [10]}

    }

Opportunity = {
        'num_patch': {'values': [3, 4, 6, 8, 10, 16, 32, 64]},
        'lstmout_dim': {'values': [2,4, 8,16,32,48,64]},
        'hidden_dim': {'values': [2, 4, 8,16,32,48,64]},
        'lr': {'values': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]},
        'lstmhidden_dim': {'values': [2, 4, 8,16,32,48,64]},
        'conv_kernel': {'values': [2,3,4,5,6,7,8, 10, 12, 16]},
        'decay': {'distribution': 'uniform', 'min': 0, 'max': 1},
        'tem_map': {'values': [True, False]},
        'batch_size': {'values': [150,100,200, 50]},
        'time_length':{'values': [128]},
        'num_sensor': {'values': [110]},
        'kernel': {'values': [[2, 2]]},
        'stride': {'values': [[1, 2]]},
        'patch_dim': {'values': [4, 8, 16, 32]},
        'n_class':{'values': [4]}

    }

EigenWorms = {
        'num_patch': {'values': [3, 4, 6, 8, 10, 16, 32, 64]},
        'lstmout_dim': {'values': [2,4, 8,16,32,48]},
        'hidden_dim': {'values': [2, 4, 8,16,32,48]},
        'lr': {'values': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]},
        'lstmhidden_dim': {'values': [2, 4, 8,16,32,48]},
        'conv_kernel': {'values': [2,3,4,5,6,7,8, 10, 12, 16]},
        'decay': {'distribution': 'uniform', 'min': 0, 'max': 1},
        'tem_map': {'values': [True, False]},
        'batch_size': {'values': [150,100, 50,25]},
        'time_length':{'values': [2248]},
        'num_sensor': {'values': [6]},
        'kernel': {'values': [[2, 2]]},
        'stride': {'values': [[1, 2]]},
        'patch_dim': {'values': [4, 8, 16, 32]},
        'n_class':{'values': [5]}

    }
