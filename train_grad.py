from __future__ import absolute_import
from __future__ import print_function, division
import sys
import time
import datetime
import argparse
import os
import numpy as np
import os.path as osp
import math
from random import sample
from numpy.lib.function_base import append 
from scipy import io
from tqdm import tqdm

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import models
from models.PCB import PCB_test
# from ReID_attr import get_target_withattr # Need Attribute file
from opts_our import get_opts, Imagenet_mean, Imagenet_stddev
from GD import Generator, MS_Discriminator, Pat_Discriminator, GANLoss, weights_init
from advloss import DeepSupervision, adv_CrossEntropyLoss, adv_CrossEntropyLabelSmooth, adv_TripletLoss
from util import data_manager
from util.dataset_loader import ImageDataset
from util.utils import fliplr, Logger, save_checkpoint, visualize_ranked_results
from util.eval_metrics import make_results
from util.samplers import RandomIdentitySampler, AttrPool

# from utils.cluster import KMEANS
from attack import attack, save_imgs, normalize, get_all_feature, get_feature_center, get_target

# Training settings
parser = argparse.ArgumentParser(description='adversarial attack')
parser.add_argument('--root', type=str, default='data', help="root path to data directory")
parser.add_argument('--targetmodel', type=str, default='aligned', choices=models.get_names())
parser.add_argument('--blackmodel', type=str, default='cam', choices=models.get_names())
parser.add_argument('--dataset', type=str, default='market1501', choices=data_manager.get_names())
parser.add_argument('--gpu', type=str, default='0', help="gpu idx")

parser.add_argument('--eps', type=int, default=16, help='epslion')
parser.add_argument('--iter_num', type=int, default=10, help='迭代次数')
# PATH
parser.add_argument('--G_resume_dir', type=str, default='', metavar='path to resume G')
parser.add_argument('--pre_dir', type=str, default='models', help='path to be attacked model')

parser.add_argument('--attr_dir', type=str, default='', help='path to attribute file')
parser.add_argument('--save_dir', type=str, default='logs', help='path to save model')
parser.add_argument('--vis_dir', type=str, default='vis', help='path to save visualization result')
parser.add_argument('--ablation', type=str, default='', help='for ablation study')
# var
parser.add_argument('--mode', type=str, default='train', help='train/test')
parser.add_argument('--D', type=str, default='MSGAN', help='Type of discriminator: PatchGAN or Multi-stage GAN')
parser.add_argument('--normalization', type=str, default='bn', help='bn or in')
parser.add_argument('--loss', type=str, default='xent_htri', choices=['cent', 'xent', 'htri', 'xent_htri'])
parser.add_argument('--ak_type', type=int, default=-1, help='-1 if non-targeted, 1 if attribute attack')
parser.add_argument('--attr_key', type=str, default='upwhite', help='[attribute, value]')
parser.add_argument('--attr_value', type=int, default=2, help='[attribute, value]')
parser.add_argument('--mag_in', type=float, default=16.0, help='l_inf magnitude of perturbation')
parser.add_argument('--temperature', type=float, default=-1, help="tau in paper")
parser.add_argument('--usegumbel', action='store_true', default=False, help='whether to use gumbel softmax')
parser.add_argument('--use_SSIM', type=int, default=2, help="0: None, 1: SSIM, 2: MS-SSIM ")
# Base
parser.add_argument('--train_batch', default=32, type=int,help="train batch size")
parser.add_argument('--test_batch', default=32, type=int, help="test batch size")
parser.add_argument('--epoch', type=int, default=50, help='number of epochs to train for')

parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num_ker', type=int, default=32, help='generator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--print_freq', type=int, default=20, help="print frequency")
parser.add_argument('--eval_freq', type=int, default=1, help="eval frequency")
parser.add_argument('--usevis', action='store_true', default=True, help='whether to save vis')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

is_training = args.mode == 'train'
attr_list = [args.attr_key, args.attr_value]
attr_matrix = None
if args.attr_dir: 
    assert args.dataset in ['dukemtmcreid', 'market1501']
    attr_matrix = io.loadmat(args.attr_dir)
    args.ablation = osp.join('attr', args.attr_key + '=' + str(args.attr_value))

pre_dir = osp.join(args.pre_dir, args.targetmodel, args.dataset+'.pth.tar')
black_dir = osp.join(args.pre_dir, args.blackmodel, args.dataset+'.pth.tar')
save_dir = osp.join(args.save_dir, args.targetmodel, args.dataset, args.ablation)
vis_dir = osp.join(args.vis_dir, args.targetmodel, args.dataset, 'adv_eps{}'.format(args.eps),args.ablation)
# vis_dir = osp.join(args.vis_dir, args.targetmodel, args.dataset, 'ori', args.ablation)


def main(opt):
    if not osp.exists(save_dir): os.makedirs(save_dir)
    if not osp.exists(vis_dir): os.makedirs(vis_dir)

    use_gpu = torch.cuda.is_available()
    pin_memory = True if use_gpu else False

    if args.mode == 'train': 
        sys.stdout = Logger(osp.join(save_dir, 'log_train.txt'))
    else: 
        sys.stdout = Logger(osp.join(save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("GPU mode")
        device = torch.device("cuda:0")
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device("cpu")
        print("CPU mode")

    ### Setup dataset loader ###
    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_img_dataset(root=args.root, name=args.dataset, split_id=opt['split_id'], cuhk03_labeled=opt['cuhk03_labeled'], cuhk03_classic_split=opt['cuhk03_classic_split'])
    if args.ak_type < 0:    # non-targeted
        trainloader = DataLoader(ImageDataset(dataset.train, transform=opt['transform_train']), sampler=RandomIdentitySampler(dataset.train, num_instances=opt['num_instances']), batch_size=args.train_batch, num_workers=opt['workers'], pin_memory=pin_memory, drop_last=True)
    elif args.ak_type > 0:    # attribute attack
        trainloader = DataLoader(ImageDataset(dataset.train, transform=opt['transform_train']), sampler=AttrPool(dataset.train, args.dataset, attr_matrix, attr_list, sample_num=16), batch_size=args.train_batch, num_workers=opt['workers'], pin_memory=pin_memory, drop_last=True)
    queryloader = DataLoader(ImageDataset(dataset.query, transform=opt['transform_test']), batch_size=args.test_batch, shuffle=False, num_workers=opt['workers'], pin_memory=pin_memory, drop_last=False)
    galleryloader = DataLoader(ImageDataset(dataset.gallery, transform=opt['transform_test']), batch_size=args.test_batch, shuffle=False, num_workers=opt['workers'], pin_memory=pin_memory, drop_last=False)
    
    ### Prepare criterion ###
    if args.ak_type<0:
        clf_criterion = adv_CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu) if args.loss in ['xent', 'xent_htri'] else adv_CrossEntropyLoss(use_gpu=use_gpu)
    else:
        clf_criterion = nn.MultiLabelSoftMarginLoss()
    metric_criterion = adv_TripletLoss(margin=args.margin, ak_type=args.ak_type)
    criterionGAN = GANLoss()     

    ### Prepare pretrained model ###
    target_net = models.init_model(name=args.targetmodel, pre_dir=pre_dir, num_classes=dataset.num_train_pids)
    
    black_net = models.init_model(name=args.blackmodel, pre_dir=black_dir, num_classes=dataset.num_train_pids)
    # target_net = nn.Sequential(
    #     Normalize(mean=Imagenet_mean, std=Imagenet_stddev, mode='torch'),
    #     models.init_model(name=args.targetmodel, pre_dir=pre_dir, num_classes=dataset.num_train_pids),
    # )
    check_freezen(target_net, need_modified=True, after_modified=False)
    check_freezen(black_net, need_modified=True, after_modified=False)

    ### Prepare main net ###

    # setup optimizer
    
    if use_gpu: 
        test_target_net = nn.DataParallel(target_net).cuda() if not args.targetmodel == 'pcb' else nn.DataParallel(PCB_test(target_net)).cuda()
        target_net = nn.DataParallel(target_net).cuda()

        test_black_net = nn.DataParallel(black_net).cuda() if not args.targetmodel == 'pcb' else nn.DataParallel(PCB_test(black_net)).cuda()       

    # ToDO:
    # 0. 把所有Query的特征向量提取出来
    ori_features, ori_pids = get_all_feature(queryloader, test_target_net, args.targetmodel)
    # 1. 先把Query所有图片进行聚类，每个PID找到各自聚类中心
    # kmeans = KMEANS(max_iter=20,verbose=False,device=device)
    # tmp = kmeans.fit(queryloader)
    center = get_feature_center(ori_features, ori_pids)  # dict: key=PID, value=feature
    # 2. 找到距离最近的错误类作为攻击目标
    tar_center = get_target(center)  # dict:key=pid, value=target pid  # 这个有点耗时
    ########################
    ###start attack
    ########################

    start_time = time.time()
    train_time = 0
    ranks = [1, 5, 10, 20]
    with torch.no_grad():
        print("get features of all gallery images:")
        gf, lgf, g_pids, g_camids = extract_and_perturb(galleryloader, target_net, args.targetmodel, use_gpu, query_or_gallery='gallery') # cpu
        black_gf, black_lgf, g_pids, g_camids = extract_and_perturb(galleryloader, test_black_net,args.blackmodel ,use_gpu, query_or_gallery='gallery') # cpu


    qf, lqf, new_qf, new_lqf, q_pids, tar_pids, q_camids = [], [], [], [], [], [], []
    black_qf, black_lqf = [], []
    new_black_qf, new_black_lqf = [], []
    print("Attack:")
    for batch_idx, (imgs, pids, camids, pids_raw) in enumerate(tqdm(queryloader)):
        
        imgs = imgs.cuda()

        # attack
        adv_imgs = attack(imgs, pids, test_target_net, center, tar_center, args.targetmodel, eps=args.eps, iter_num=args.iter_num)
        
        # save
        save_imgs(adv_imgs, pids, batch_idx, vis_dir)
        # adv_imgs = imgs.clone()
        
        # test
        with torch.no_grad():
            
            ls = extract(imgs.clone(), test_target_net)
            new_ls = extract(adv_imgs.clone(), test_target_net)
            black_ls = [extract(imgs.clone(), test_black_net)[1]]  # 黑盒
            new_black_ls = [extract(adv_imgs.clone(), test_black_net)[1]]  # 黑盒
            if len(ls) == 1: 
                features = ls[0]
                new_features = new_ls[0]
            if len(ls) == 2: 
                features, local_features = ls    # []  
                new_features, new_local_features = new_ls    # []
                lqf.append(local_features.clone().detach().data)
                new_lqf.append(new_local_features.clone().detach().data)
            
            # 黑盒
            if len(black_ls) == 1: 
                black_features = black_ls[0]
                new_black_features = new_black_ls[0]
            if len(black_ls) == 2: 
                black_features, black_local_features = black_ls    # [] 
                black_lqf.append(black_local_features.clone().detach().data)

            qf.append(features.clone().detach().data)
            new_qf.append(new_features.clone().detach().data)
            black_qf.append(black_features.clone().detach().data)
            new_black_qf.append(new_black_features.clone().detach().data)

            q_pids.extend(pids)
            q_camids.extend(camids)

    qf = torch.cat(qf, 0)
    new_qf = torch.cat(new_qf, 0)
    black_qf = torch.cat(black_qf, 0)
    new_black_qf = torch.cat(new_black_qf, 0)
    if not lqf == []: lqf = torch.cat(lqf, 0)
    if not new_lqf == []: new_lqf = torch.cat(new_lqf, 0)
    if not black_lqf == []: black_lqf = torch.cat(black_lqf, 0)
    q_pids, q_camids = np.asarray(q_pids), np.asarray(q_camids)

    # result
    print("#"*20+"\n","查询干净图片:")
    distmat, cmc, mAP, t_cmc, t_mAP = make_results(qf, gf, lqf, lgf, q_pids, g_pids, q_camids, g_camids, args.targetmodel, args.ak_type, tar_center, mode='adv')
    print("#"*20+"\n","查询对抗样本图片:")
    new_distmat, new_cmc, new_mAP, t_new_cmc, t_new_mAP = make_results(new_qf, gf, new_lqf, lgf, q_pids, g_pids, q_camids, g_camids, args.targetmodel, args.ak_type, tar_center, mode='adv')
    print("#"*20+"\n","黑盒查询:")
    black_distmat, black_cmc, black_mAP, t_black_cmc, t_black_mAP = make_results(black_qf, black_gf, black_lqf, black_lgf, q_pids, g_pids, q_camids, g_camids, args.blackmodel, args.ak_type, tar_center, mode='adv')

    new_black_distmat, new_black_cmc, new_black_mAP, t_new_black_cmc, t_new_black_mAP = make_results(new_black_qf, black_gf, new_black_lqf, black_lgf, q_pids, g_pids, q_camids, g_camids, args.blackmodel, args.ak_type, tar_center, mode='adv')

    # print("查询对抗样本目标攻击图片")
    print("Results ----------")
    print("Before  , mAP: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}".format(mAP, ranks[0], cmc[ranks[0]-1], ranks[1], cmc[ranks[1]-1], ranks[2], cmc[ranks[2]-1], ranks[3], cmc[ranks[3]-1]))
    print("t_Before, mAP: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}".format(t_mAP, ranks[0], t_cmc[ranks[0]-1], ranks[1], t_cmc[ranks[1]-1], ranks[2], t_cmc[ranks[2]-1], ranks[3], t_cmc[ranks[3]-1]))
    print("白盒测试")
    print("After  , mAP: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}".format(new_mAP, ranks[0], new_cmc[ranks[0]-1], ranks[1], new_cmc[ranks[1]-1], ranks[2], new_cmc[ranks[2]-1], ranks[3], new_cmc[ranks[3]-1]))
    print("t_After, mAP: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}".format(t_new_mAP, ranks[0], t_new_cmc[ranks[0]-1], ranks[1], t_new_cmc[ranks[1]-1], ranks[2], t_new_cmc[ranks[2]-1], ranks[3], t_new_cmc[ranks[3]-1]))
    
    # 黑盒
    print("黑盒测试")
    print("Black  , mAP: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}".format(black_mAP, ranks[0], black_cmc[ranks[0]-1], ranks[1], black_cmc[ranks[1]-1], ranks[2], black_cmc[ranks[2]-1], ranks[3], black_cmc[ranks[3]-1]))
    print("t_Black, mAP: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}".format(t_black_mAP, ranks[0], t_black_cmc[ranks[0]-1], ranks[1], t_black_cmc[ranks[1]-1], ranks[2], t_black_cmc[ranks[2]-1], ranks[3], t_black_cmc[ranks[3]-1]))    

    print("Black  , mAP: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}".format(new_black_mAP, ranks[0], new_black_cmc[ranks[0]-1], ranks[1], new_black_cmc[ranks[1]-1], ranks[2], new_black_cmc[ranks[2]-1], ranks[3], new_black_cmc[ranks[3]-1]))
    print("t_Black, mAP: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}".format(t_new_black_mAP, ranks[0], t_new_black_cmc[ranks[0]-1], ranks[1], t_new_black_cmc[ranks[1]-1], ranks[2], t_new_black_cmc[ranks[2]-1], ranks[3], t_new_black_cmc[ranks[3]-1]))    

    

    # elapsed = round(time.time() - start_time)
    # elapsed = str(datetime.timedelta(seconds=elapsed))
    # train_time = str(datetime.timedelta(seconds=train_time))
    # print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


    # if (batch_idx+1) % args.print_freq == 0:
    #     print("===> Epoch[{}]({}/{}) loss_D: {:.4f} loss_G_GAN: {:.4f} loss_G_ReID: {:.4f} loss_G_SSIM: {:.4f}".format(epoch, batch_idx, len(trainloader), loss_D.item(), loss_G_GAN.item(), loss_G_ReID.item(), loss_G_ssim))

def extract_features(imgs, target_net):
    ls = extract(imgs, target_net)
    if len(ls) == 1: features = ls[0]
    if len(ls) == 2: features, local_features = ls    # []
        # lf.append(local_features.detach().data.cpu())     

    # f.append(features.detach().data.cpu())
    return [features, local_features]



def extract_and_perturb(loader, target_net, net_name, use_gpu, query_or_gallery):
    f, lf, new_f, new_lf, l_pids, l_camids = [], [], [], [], [], []
    ave_mask, num = 0, 0
    for batch_idx, (imgs, pids, camids, pids_raw) in enumerate(tqdm(loader)):
        if use_gpu: 
            imgs = imgs.cuda()
        if net_name == 'cam':
            ls = [extract(imgs, target_net)[1]]
        else:
            ls = extract(imgs, target_net)
        
        if len(ls) == 1: features = ls[0]
        if len(ls) == 2: 
            features, local_features = ls    # []
            lf.append(local_features.detach().data)

        f.append(features.detach().data)
        l_pids.extend(pids)
        l_camids.extend(camids)

    f = torch.cat(f, 0)
    if not lf == []: lf = torch.cat(lf, 0)
    l_pids, l_camids = np.asarray(l_pids), np.asarray(l_camids)
    
    print("Extracted features for {} set, obtained {}-by-{} matrix".format(query_or_gallery, f.size(0), f.size(1)))
    if query_or_gallery == 'gallery':
        return [f, lf, l_pids, l_camids]
    elif query_or_gallery == 'query':
        new_f = torch.cat(new_f, 0)
        if not new_lf == []: 
            new_lf = torch.cat(new_lf, 0)
        return [f, lf, new_f, new_lf, l_pids, l_camids]

def extract(imgs, target_net):
    if args.targetmodel in ['pcb', 'lsro']:
        ls = [target_net(normalize(imgs), is_training)[0] + target_net(normalize(fliplr(imgs)), is_training)[0]]
    else: 
        ls = target_net(normalize(imgs), is_training)
    for i in range(len(ls)): ls[i] = ls[i].data.cpu()
    return ls

def perturb(imgs, G, D, train_or_test='test'):
    n,c,h,w = imgs.size()
    delta = G(imgs)
    delta = L_norm(delta, train_or_test)
    new_imgs = torch.add(imgs.cuda(), delta[0:imgs.size(0)].cuda())

    _, mask = D(torch.cat((imgs, new_imgs.detach()), 1))
    delta = delta * mask
    new_imgs = torch.add(imgs.cuda(), delta[0:imgs.size(0)].cuda())

    for c in range(3):
        new_imgs.data[:,c,:,:] = new_imgs.data[:,c,:,:].clamp(new_imgs.data[:,c,:,:].min(), new_imgs.data[:,c,:,:].max()) # do clamping per channel
    if train_or_test == 'train':
        return new_imgs, mask
    elif train_or_test == 'test':
        return new_imgs, delta, mask

def L_norm(delta, mode='train'):
    delta.data += 1 
    delta.data *= 0.5

    for c in range(3):
        delta.data[:,c,:,:] = (delta.data[:,c,:,:] - Imagenet_mean[c]) / Imagenet_stddev[c]

    bs = args.train_batch if (mode == 'train') else args.test_batch
    for i in range(bs):
        # do per channel l_inf normalization
        for ci in range(3):
            try:
                l_inf_channel = delta[i,ci,:,:].data.abs().max()
                # l_inf_channel = torch.norm(delta[i,ci,:,:]).data
                mag_in_scaled_c = args.mag_in/(255.0*Imagenet_stddev[ci])
                delta[i,ci,:,:].data *= np.minimum(1.0, mag_in_scaled_c / l_inf_channel.cpu()).float().cuda()
            except IndexError:
                break
    return delta

def save_img_del(ls, pids, camids, epoch, batch_idx):
    image, new_image, delta, mask = ls
    # undo normalize image color channels
    delta_tmp = torch.zeros(delta.size())
    for c in range(3):
        image.data[:,c,:,:] = (image.data[:,c,:,:] * Imagenet_stddev[c]) + Imagenet_mean[c]
        new_image.data[:,c,:,:] = (new_image.data[:,c,:,:] * Imagenet_stddev[c]) + Imagenet_mean[c]
        delta_tmp.data[:,c,:,:] = (delta.data[:,c,:,:] * Imagenet_stddev[c]) + Imagenet_mean[c]
 
    if args.usevis: 
        torchvision.utils.save_image(image.data, osp.join(vis_dir, 'original_epoch{}_batch{}.png'.format(epoch, batch_idx)))
        torchvision.utils.save_image(new_image.data, osp.join(vis_dir, 'polluted_epoch{}_batch{}.png'.format(epoch, batch_idx)))
        torchvision.utils.save_image(delta_tmp.data, osp.join(vis_dir, 'delta_epoch{}_batch{}.png'.format(epoch, batch_idx)))
        torchvision.utils.save_image(mask.data*255, osp.join(vis_dir, 'mask_epoch{}_batch{}.png'.format(epoch, batch_idx)))

def check_freezen(net, need_modified=False, after_modified=None):
    # print(net)
    cc = 0
    for child in net.children():
        for param in child.parameters():
            if need_modified: param.requires_grad = after_modified
            # if param.requires_grad: print('child', cc , 'was active')
            # else: print('child', cc , 'was forzen')
        cc += 1

if __name__ == '__main__':
    opt = get_opts(args.targetmodel)
    main(opt)
