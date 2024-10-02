import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from load_data2 import get_data_path, load4train
from network import MLDA,U,V
from idcd import idcd
import numpy as np
from sklearn.metrics import f1_score
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr

def set_seed(seed=1024):
    torch.manual_seed(seed+1)
    torch.cuda.manual_seed(seed+2)
    torch.cuda.manual_seed_all(seed+3)

def p_cof(x,y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    std_x = torch.std(x)
    std_y = torch.std(y)

    return torch.sum((x-mean_x)*(y-mean_y)) / (std_x*std_y)

def gauss_(v1, v2, sigma):
    norm_ = torch.norm(v1 - v2, p=2, dim=0)
    return torch.exp(-0.5 * norm_ / sigma ** 2)

def calculate_js_divergence(source_features, target_features):

    source_prob = source_features / np.sum(source_features, axis=1, keepdims=True)
    target_prob = target_features / np.sum(target_features, axis=1, keepdims=True)
    js_divergence = np.zeros(source_features.shape[0])
    for i in range(source_features.shape[0]):
        js_divergence[i] = jensenshannon(source_prob[i], target_prob[i])**2

    return np.mean(js_divergence)

def calculate_kl_divergence(source_features, target_features):

    source_prob = source_features / np.sum(source_features, axis=1, keepdims=True)
    target_prob = target_features / np.sum(target_features, axis=1, keepdims=True)
    kl_divergence = np.zeros(source_features.shape[0])
    for i in range(source_features.shape[0]):
        kl_divergence[i] = np.sum(rel_entr(source_prob[i], target_prob[i]))

    return np.mean(kl_divergence)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net_config = {"EPOCH": 600, "fts": 750, "cls": 2, "lr": 5e-3, "weight_decay": 0.0005,"BS":64}
config_path = {"file_path": "data path",
               "label_path": "label path"}
files = [0,'01sub_8.mat','02sub.mat','03sub.mat','04sub.mat','05sub.mat','06sub.mat','07sub.mat',\
              '08sub.mat','09sub.mat','10sub.mat','11sub.mat']

for i in range(1,12):
    tar_file=files[i]
    set_seed(1024)

    path_list = get_data_path(config_path["file_path"])
    path_list.remove(config_path["file_path"] + tar_file)
    source_path_list = path_list

    source_label_list = get_data_path(config_path['label_path'])
    source_label_list.remove(config_path["label_path"] + tar_file)

    target_path_list = [config_path["file_path"] + tar_file]
    target_label_list = [config_path["label_path"] + tar_file]

    source_sample, source_label = load4train(source_path_list, source_label_list)
    target_sample, target_label = load4train(target_path_list, target_label_list)

    source_dset = torch.utils.data.TensorDataset(source_sample, source_label)
    target_dset = torch.utils.data.TensorDataset(target_sample, target_label)
    test_dset = torch.utils.data.TensorDataset(target_sample, target_label)

    source_loader = torch.utils.data.DataLoader(source_dset, batch_size=net_config["BS"], shuffle=True)
    target_loader = torch.utils.data.DataLoader(target_dset, batch_size=net_config["BS"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=net_config["BS"], shuffle=True)

    data_loader = {"source_loader": source_loader, "target_loader": target_loader, "test_loader": test_loader}

    model = MLDA(net_config["fts"],net_config["cls"])
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=net_config["lr"], weight_decay=net_config["weight_decay"])
    u = U(32,0.05).to(device)
    v = V(32,0.05).to(device)
    optimizer_u = torch.optim.SGD(u.parameters(), lr=net_config["lr"], weight_decay=net_config["weight_decay"])
    optimizer_v = torch.optim.SGD(v.parameters(), lr=net_config["lr"], weight_decay=net_config["weight_decay"])

    best_acc = 0
    best_f1 = 0

    writer = SummaryWriter("output path" + tar_file)

    for epoch in range(1,net_config["EPOCH"]+1):
        model.train()
        correct = 0
        count = 0
        total_time=0
        for idx, (src_examples, tar_examples) in enumerate(zip(data_loader["source_loader"], data_loader["target_loader"])):
            src_data, src_label_cls = src_examples
            tar_data, true_label = tar_examples
            if src_data.shape[0] != tar_data.shape[0]:
                continue
            src_data, src_label_cls = src_data.to(device), src_label_cls.to(device)
            tar_data, true_label = tar_data.to(device), true_label.to(device)

            src_feature, tar_feature, src_output_cls, tar_output_cls = model(src_data,tar_data)

            cls_loss = criterion(src_output_cls, src_label_cls)

            src_feature1 = u(src_feature)
            tar_feature1 = v(tar_feature)

            gdd_loss = calculate_js_divergence(src_feature1.detach().cpu().numpy(), \
                                                tar_feature1.detach().cpu().numpy())

            max_prob, pseudo_label1 = torch.max(tar_output_cls, dim=1)

            confident_example = tar_feature
            confident_label = pseudo_label1

            lsd_loss = idcd(src_feature, confident_example, src_label_cls, confident_label)

            LOSS_WEIGHT = 0.5
            # LOSS_WEIGHT = 1.0 / (1.0 + torch.exp(torch.tensor(100.0 - epoch)))
            train_loss = cls_loss + 2 * ((1 - LOSS_WEIGHT) * gdd_loss + LOSS_WEIGHT * lsd_loss)

            optimizer.zero_grad()
            optimizer_u.zero_grad()
            optimizer_v.zero_grad()
            train_loss.backward()
            optimizer.step()
            optimizer_u.step()
            optimizer_v.step()

            _, pred = torch.max(src_output_cls, dim=1)
            correct += pred.eq(src_label_cls.data.view_as(pred)).sum()
            count += pred.size(0)

        train_acc = float(correct) / count
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("train/class-loss", cls_loss, epoch)

        model.eval()
        correct = 0
        y_true = torch.tensor([]).cuda()
        y_pred = torch.tensor([]).cuda()
        with torch.no_grad():

            for idx, (test_input, label) in enumerate(data_loader["test_loader"]):
                test_input, label = test_input.to(device), label.to(device)

                _,output = model(test_input,None)
                test_loss = criterion(output, label)
                _, pred = torch.max(output, dim=1)
                correct += pred.eq(label.data.view_as(pred)).sum()
                y_pred = torch.cat((y_pred, pred), dim=0)
                y_true = torch.cat((y_true, label), dim=0)

            test_acc = float(correct) / len(data_loader["test_loader"].dataset)

            f1 = f1_score(y_true.cpu(),y_pred.cpu(),average='weighted')

            if test_acc > best_acc:
                best_acc = test_acc
                best_f1 = f1
        writer.add_scalar("test/loss", test_loss, epoch)
        writer.add_scalar("test/Accuracy", test_acc, epoch)
        writer.add_scalar("test/Best_Acc", best_acc, epoch)
        writer.add_scalar("test/f1_score", best_f1, epoch)

    writer.close()
