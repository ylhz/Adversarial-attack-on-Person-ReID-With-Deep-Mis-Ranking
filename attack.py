import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable as V

import os
from tqdm import tqdm

from util.utils import fliplr
from opts import get_opts, Imagenet_mean, Imagenet_stddev

# def get_adv_loss(ls):
#     # 靠近目标聚类中心
#     # clf_criterion = adv_CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu) if args.loss in ['xent', 'xent_htri'] else adv_CrossEntropyLoss(use_gpu=use_gpu)
#     # return adv_loss               
#     pass

def get_rank_loss(ls, target_pid):
    """排序误差"""
    pass
    

class adv_TripletLoss(nn.Module):
    """拉近和目标类的距离"""
    def __init__(self, margin=0.3):
        super(adv_TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    def distance(self, f1, f2):
        """只考虑f1和f2的对应样本的距离,不考虑不同样本之间的关系

        Args:
            f1 (tensor): 特征向量 [N*2048]
            f2 (tensor): 特征向量 [N*2048]
        Returns:
            tensor: 对应距离 [N]
        """
        # 欧式距离
        d = ((f1-f2)**2).sum(1)  # [N]
        return d

    def forward(self, ori_features, adv_features, pids, center_features, tar_pids):
        """计算Loss

        Args:
            adv_features (tensor): 当前对抗样本的特征向量 [N, 2048]
            ori_features (tensor): 当前样本的原始特征向量 [N, 2048]
            pids (tensor int): 当前样本的原本PID [N]
            center_features (dict): PID和聚类中心 
            tar_pids (dict, numpy): PID和目标PID

        Returns:
            [type]: [description]
        """
        n = ori_features.size(0)  # batch size
        c_features = torch.stack([center_features[p.item()] for p in pids])  # 当前batch对应的PID聚类中心
        t_pids = [tar_pids[p.item()] for p in pids]  # 当前batch对应的PID攻击目标
        tar_features = torch.stack([center_features[p.item()] for p in t_pids])  # 当前batch对应的目标PID聚类中心
        
        adv_dist = self.distance(adv_features, ori_features).sum()/n  # max
        center_dist = self.distance(adv_features, c_features).sum()/n  # max
        tar_dist = self.distance(adv_features, tar_features).sum()/n  # min
        # print("adv_dist:", adv_dist, "center_dist", center_dist, "tar_dist", tar_dist)
        # loss = (tar_dist - 0.5*center_dist - 0.5*adv_dist) / 2  # loss最小化
        loss = tar_dist  # loss最小化
        return loss

    # def forward_old(self, adv_features, ori_features, pids, target_pids):
    #     """
    #     Args:
    #         features: feature matrix with shape (batch_size, feat_dim)[5,2048,8]
    #         pids: ground truth labels with shape (num_classes)
    #         target_pids: pids with targets (batch_size)
    #     """
    #     n = adv_features.size(0)  # batch size

    #     # L-2 norm dist
    #     # print(torch.pow(ori_features, 2).shape)
    #     ori_f = torch.pow(ori_features, 2).sum(dim=1, keepdim=True).expand(n, n)
    #     adv_f = torch.pow(adv_features, 2).sum(dim=1, keepdim=True).expand(n, n)
    #     dist = ori_f + adv_f.t()
    #     # dist.addmm_(ori_features, adv_features.t(), 1, -2)
    #     dist.addmm_(1, -2, ori_features, adv_features.t())
    #     dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

    #     # ori_mask = pids.expand(n, n).eq(pids.expand(n, n).t())
    #     target_mask = target_pids.expand(n, n).t().eq(pids.expand(n, n))
    #     # print('dist:', torch.argmax(dist, 1))
    #     # print('pids:', pids)
    #     # print('target_pids:',target_pids)
    #     # print('tar:', target_mask)

    #     dist_ap, dist_an = [], []
    #     for i in range(n):
            
    #         # 若当前batch中无target，则无法unsqueeze，会报错
    #         a=dist[i][target_mask[i]].max().unsqueeze(0)
            
    #         dist_ap.append(a) # Close the distance to the target feature
    #         dist_an.append(dist[i][target_mask[i] == 0].min().unsqueeze(0)) # 

    #     dist_ap = torch.cat(dist_ap)
    #     dist_an = torch.cat(dist_an)

    #     y = torch.ones_like(dist_an)
        
    #     # print("adv_features:", adv_features.requires_grad)
    #     # print("dist_an:", dist_an.requires_grad)
    #     # print("dist_ap:", dist_ap.requires_grad)
    #     # 和目标类别的距离要小于和其他类的距离
    #     # 期望：dist_an > dist_ap 否则更新
    #     loss = self.ranking_loss(dist_an, dist_ap, y)
    #     return torch.log(loss)

def get_target_pids(features, pids):
    """
    Args:
        features: feature matrix with shape (batch_size, feat_dim)[5,2048,8]
        pids: ground truth labels with shape (num_classes)
    """
    n = features.size(0)  # batch size

    # L-2 norm dist
    # print(torch.pow(ori_features, 2).shape)
    ori_f = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = ori_f + ori_f.t()
    # dist.addmm_(ori_features, adv_features.t(), 1, -2)
    dist.addmm_(1, -2, features, features.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

    target_idx = torch.argmax(dist, 1)
    target_pids = torch.zeros_like(pids)
    for i in range(n):
        target_pids[i] = pids[target_idx[i]]
    # set target = pid+1, 若设为
    # target_pids = pids + 1
    # target_pids[target_pids>self.pids_max] = 1
    return target_pids.detach()


def attack(imgs, pids, target_net, center, tar_center, net_name, eps, iter_num):
    """攻击函数

    Args:
        imgs (tensor): [description]
        pids (tensor): [description]
        target_net ([type]): [description]
        center (dict): 记录了所有PID的聚类中心特征
        tar_center (dict): PID以及PID的攻击目标
        net_name ([type]): [description]
        eps ([type]): [description]
        iter_num ([type]): [description]   

    Returns:
        [type]: [description]
    """
    eps = eps/255.
    alpha = eps/iter_num
    is_training = False
    momentum = 0.9
    get_adv_loss = adv_TripletLoss()
    noise = torch.zeros_like(imgs, requires_grad=True)
    
    old_grad =0.0
    for i in range(iter_num):
        adv_imgs = imgs + noise
        # ls = target_net(adv_imgs, is_training)
        ls = extract(normalize(adv_imgs), target_net, net_name, is_training)
        # test
        if len(ls) == 1: new_features = ls[0]
        if len(ls) == 2: new_features, new_local_features = ls    # []
        # # train
        # if len(ls) == 1: new_outputs = ls[0]
        # if len(ls) == 2: new_outputs, new_features = ls
        # if len(ls) == 3: new_outputs, new_features, new_local_features = ls
        if i == 0 and len(ls) != 1:
            ori_features = new_features.clone().detach()
            # ori_outputs = new_outputs.detach().clone()  # train
            # target_pids = get_target_pids(ori_outputs, pids)  # train
        

        # maximize the distance of the matched pair, minimize the distance of the target pair
        assert len(ls) >= 2 
        adv_loss = get_adv_loss(ori_features, new_features, pids, center, tar_center)
        # adv_loss = get_adv_loss(new_outputs, ori_outputs, pids, target_pids=target_pids)  # train
        loss = adv_loss
        loss.backward()
        grad = noise.grad.data

        # MI-FGSM
        grad = grad / torch.abs(grad).mean([1,2,3], keepdim=True)
        grad = momentum * old_grad + grad
        old_grad = grad

        noise = noise - alpha * torch.sign(grad)

        # avoid of bound
        # 先对noise标准化
        noise = torch.clamp(noise, -eps, eps)
        adv_imgs = imgs + noise
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
        noise = adv_imgs - imgs
        noise = V(noise, requires_grad = True)

    return adv_imgs.detach()

    distmat, hits, ignore_list = make_results(new_qf, gf, new_lqf, lgf, q_pids, g_pids, q_camids, g_camids, args.targetmodel, args.ak_type, attr_matrix, args.dataset, attr_list)


    
def extract(imgs, target_net, net_name, is_training):
    if net_name in ['pcb', 'lsro']:
        ls = [target_net(imgs, is_training)[0] + target_net(fliplr(imgs), is_training)[0]]
    else: 
        ls = target_net(imgs, is_training)
    # for i in range(len(ls)): ls[i] = ls[i].data
    return ls

def save_imgs(imgs, pids, batch_idx, output_dir):
    # for c in range(3):
    #     imgs.data[:,c,:,:] = (imgs.data[:,c,:,:] * Imagenet_stddev[c]) + Imagenet_mean[c]

    torchvision.utils.save_image(imgs.data, os.path.join(output_dir, 'adv_batch{}.png'.format(batch_idx)))

# data
class Normalize(nn.Module):
    """
    mode:
        'tensorflow':convert data from [0,1] to [-1,1]
        'torch':(input - mean) / std
    """
    def __init__(self, mean=0, std=0, mode='tensorflow'):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
        self.mode = mode

    def forward(self, input, is_training):
        size = input.size()
        x = input.clone()

        if self.mode == 'tensorflow':
            x = x * 2.0 -1.0  # convert data from [0,1] to [-1,1]
        elif self.mode == 'torch':
            for i in range(size[1]):
                x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x
def normalize(x):
    # size = input.size()
    mean = Imagenet_mean
    std = Imagenet_stddev
    for i in range(x.shape[1]):
        x[:, i] = (x[:, i] - mean[i]) / std[i]
    return x

# 
def get_all_feature(queryloader, test_target_net, net_name):
    """获取所有query的特征向量
    Return:
        features: [N, 2048] N表示所有图片数目，device:gpu
        pids: [N]，device:cpu
    """
    print("get features of all Query images:")
    for batch_idx, (imgs, pid, camids, pid_raw) in enumerate(tqdm(queryloader)):
        imgs = imgs.cuda()
        with torch.no_grad():
            ls = extract(normalize(imgs), test_target_net, net_name, is_training=False)
            if len(ls) == 1: f = ls[0]  
            if len(ls) == 2: f, l_f = ls  # features, local_features
            # print("test, ls[0]:", f.shape, "ls[1]:", l_f.shape)
            # input()
            if batch_idx == 0:
                features = f  # [batch, feature]
                pids = pid  # [batch]
            else:
                features = torch.cat((features, f))
                pids = torch.cat((pids, pid))
    return features, pids

def get_feature_center(ori_features, ori_pids):
    """得到每个PIDS的中心
    Args:
        ori_features: 所有Query的特征向量tensor,gpu
        ori_pids: 特征向量对应的PID,cpu
    Return:
        pid_center: 字典，key=pid, value=feature
    """
    
    def distance(f1,f2):
        d = torch.sum((f1-f2)**2)
        return d
    tmp = set(ori_pids.numpy())
    pid_center = {}
    device = ori_features.device
    for i in tmp:
        idx = ori_pids==i
        i_feature = ori_features[idx]
        i_len = i_feature.shape[0]
        dist = torch.zeros((i_len, i_len),device=device)
        # 计算当前PID中距离所有点最近的中心点
        for j in range(i_len):
            for k in range(j):
                dist[j,k] = distance(i_feature[j], i_feature[k])
                dist[k,j] = dist[j,k]
        center_idx = torch.argmin(dist.sum(0))
        pid_center[i] = i_feature[center_idx]
        
    return pid_center

def get_target(pid_center):
    """为每个pid找到攻击目标pid_tar
    Args:
        pid_center: 字典，key=pid, value=feature
    Return:
        tar_pids: 字典，key=pid, value=target pid
    """
    def distance(f1,f2):
        d = torch.sum((f1-f2)**2)
        return d    
    tar_pids = {}
    len_pid = len(pid_center.keys())
    # pid_dist = torch.full([len_pid, len_pid], torch.tensor(float('inf')))  # 找最近的错误类
    pid_dist = torch.full([len_pid, len_pid], torch.tensor(0))  # 找最远的错误类
    pid_keys = list(pid_center.keys())
    pid_values = list(pid_center.values())
    for i in range(len_pid):
        for j in range(i):
            pid_dist[i,j] = distance(pid_values[i], pid_values[j])
            pid_dist[j,i] = pid_dist[i,j]
    for i in range(len_pid):
        # idx = torch.argmin(pid_dist[i])  # 找最近的错误类
        idx = torch.argmax(pid_dist[i])  # 找最远的错误类
        tar_pids[pid_keys[i]] = pid_keys[idx]
    return tar_pids