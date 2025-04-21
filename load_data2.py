import os, sys
import torch
import numpy as np
import scipy.io as scio

def get_data_path(file_path):
    data_path = []
    for f in os.listdir(file_path):
        if f.startswith("."):
            continue
        else:
            data_path.append(os.path.join(file_path, f))

    return data_path
# def get_data_path(file_path):
#     path = []
#     data_path = os.listdir(file_path)  # 得到文件夹下的所有文件名称
#     for i in data_path:
#         path.append(file_path+i)
#     return path

def load_trained_data(samples_path_list, labels_path_list):
    # load the label data
    # label = scio.loadmat(labels_path_list)
    train_sample = None
    train_label = None
    for sample_path in samples_path_list:
        # load the sample data
        sample = scio.loadmat(sample_path, verify_compressed_data_integrity=False)
        # print(sample.keys())
        val = sample['de_LDS']
        if train_sample is None:
            train_sample = val
            # print(val.shape)
        else:
            train_sample = np.concatenate((train_sample, val), axis=1)
            # print(val.shape)

    for label_path in labels_path_list:
        # load the sample data
        label = scio.loadmat(label_path, verify_compressed_data_integrity=False)
        # print(label['perclos'].shape)
        label = np.array(label['perclos'])
        # print(label.shape)
        label = (label>0.35).astype(int).flatten()
        # print(label.shape)
        if train_label is None:
            train_label = label
            # print(train_label.shape,train_label[:10])
        else:
            train_label = np.concatenate((train_label, label), axis=0)
            # print(train_label.shape,train_label[:10])
    return train_sample, train_label

def normalize(features, select_dim=0):
    features_min, _ = torch.min(features, dim=select_dim)
    features_max, _ = torch.max(features, dim=select_dim)
    features_min = features_min.unsqueeze(select_dim)
    features_max = features_max.unsqueeze(select_dim)
    return (features - features_min)/(features_max - features_min)

def reshape_feature(fts_sample):
    fts = None
    if torch.is_tensor(fts_sample):
        elec, number, freq = fts_sample.shape
        fts = fts_sample.permute(1, 0, 2).reshape(number, elec * freq)
    else:
        print("Something Wrong!!!")
    return fts


def load4train(samples_path_list, labels_path):
    """
    load the SEED data set for TL
    """
    train_sample, train_label = load_trained_data(samples_path_list, labels_path)
    # transfer from ndarray to tensor
    train_sample = torch.from_numpy(train_sample).type(torch.FloatTensor)
    train_label = torch.from_numpy(train_label).type(torch.LongTensor)
    # normalize tensor
    train_sample = normalize(train_sample)
    # reshape the tensor
    train_sample = reshape_feature(train_sample)
    
    return train_sample, train_label

