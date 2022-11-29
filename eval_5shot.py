from __future__ import print_function
import os
import argparse
import socket
import time
import sys
from tqdm import tqdm
import mkl
import torch
import random
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
from models import model_pool
from models.util import create_model, get_teacher_name
from dataset.mini_imagenet import ImageNet, ImageNetRank, MetaImageNet
from dataset.cifar import CIFAR100, MetaCIFAR100
from dataset.CUB import CUB, MetaCUB, CUBRank
from dataset.transform_cfg import transforms_options, transforms_list
from util import adjust_learning_rate, accuracy, AverageMeter, rotrate_concat, Logger, generate_final_report
import numpy as np
import wandb
from dataloader import get_dataloaders
import copy
import h5py
from torch.autograd import Variable

##############################center##################
from models.center_module import LinearModule

########################################rank#################
from rank_module import LogisticModule, Linear_rankModule
from rank_data import Set_rankdata, Set_rankdata_dot, Set_rankdata_cat_mul, Set_rankdata_0, Set_rankdata_1, Set_rankdata_2
from meta_rank_eval import meta_test_dot, meta_test_mul, meta_test_mul_5, meta_test_mixways_5, meta_test_mixways_5_1, meta_test_mixways_2,meta_test_mixways_3,meta_test_nshot

os.environ["CUDA_VISIBLE_DEVICES"] = '4,2'


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    # dataset and model
    parser.add_argument('--model_s', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--model_t', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet','CIFAR-FS', 'FC100', 'CUB'])
    parser.add_argument('--batch_size', type=int, default=200, help='batch_size')
    # path to teacher model
    parser.add_argument('--path_t', type=str, default="", help='teacher model snapshot')

    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    # specify folder
    parser.add_argument('--model_path', type=str, default='save/', help='path to save model')
    parser.add_argument('--data_root', type=str, default='/raid/data/IncrementLearn/imagenet/Datasets/MiniImagenet/', help='path to data root')
    parser.add_argument('--trans', type=int, default=16, help='number of transformations')
    parser.add_argument('--memfeature_size', type=int, default=64, help='temperature for contrastive ssl loss')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N', help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N', help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=5, metavar='N', help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N', help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=1, type=int, help='The number of augmented samples for each meta test sample')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size', help='Size of test batch)')
    parser.add_argument('-t', '--trial', type=str, default='1', help='the experiment id')

    opt = parser.parse_args()
    return opt


class Wrapper(nn.Module):

    def __init__(self, model):
        super(Wrapper, self).__init__()
    
        self.model = model
        self.feat = torch.nn.Sequential(*list(self.model.module.children())[:-3])
           
    def forward(self, images):
        feat = self.feat(images)
        feat = feat.view(images.size(0), -1)

        return feat


def load_model(model_path, model_name, n_cls, dataset='miniImageNet', trans=16, embd_size=64):
    """load the final model"""
    print('==> loading model')
    print(model_name)
    model = create_model(model_name, n_cls, dataset, n_trans=trans, embd_sz=embd_size)
    if torch.cuda.device_count() > 1:
        print("gpu count:", torch.cuda.device_count())
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model


def my_get_dataloaders(opt, partition, pretrain):

    if opt.dataset == 'miniImageNet':

        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test.pickle'),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        meta_valloader  = DataLoader(MetaImageNet(args=opt, partition='val.pickle'),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        meta_trainloader = DataLoader(MetaImageNet(args=opt, partition='train.pickle'),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers) 


    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':

        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test.pickle'),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val.pickle'),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)  
        meta_trainloader = DataLoader(MetaCIFAR100(args=opt, partition='train.pickle'),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers) 

    elif opt.dataset == 'tieredImageNet':

        meta_trainloader = DataLoader(tiered_imagenet(args=opt, partition='train', transform=None),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
        meta_testloader = DataLoader(MetaTieredImageNet(args=opt, partition='test',
                                                        train_transform=None,
                                                        test_transform=None),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaTieredImageNet(args=opt, partition='val',
                                                       train_transform=None,
                                                       test_transform=None),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)

    elif opt.dataset == 'CUB':
        meta_testloader = DataLoader(MetaCUB(args=opt, partition='test'),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCUB(args=opt, partition='val'),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)  
        meta_trainloader = DataLoader(MetaCUB(args=opt, partition='train'),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers) 

    return meta_trainloader, meta_valloader, meta_testloader



def eval():
    #trian_acc
    start = time.time()
    meta_train_acc, meta_train_std = meta_test_nshot(IE_model, center_model, rank_model, meta_testloader)
    test_time = time.time() - start
    print('Meta Train Acc : {:.4f}, Meta Train std: {:.4f}, Time: {:.1f}'.format(meta_train_acc, meta_train_std, test_time))



if __name__ == '__main__':
    opt = parse_option()

    #load image
    partition = 'train.pickle'
    pretrain = True
    opt = parse_option()
    meta_trainloader, meta_valloader, meta_testloader = my_get_dataloaders(opt, partition, pretrain)
    print("=====>loaderdata success")

    # IEmodel
    feature_model = load_model(opt.path_t, opt.model_t, 64, opt.dataset, opt.trans, opt.memfeature_size)
    #feature_model = load_model(opt.path_t, opt.model_t, 100, opt.dataset, opt.trans, opt.memfeature_size)
    IE_model = Wrapper(feature_model).cuda()
    IE_model.eval()

    #center_model
    center_model = LinearModule(640,64).cuda()
    test_weights_center = '/home/ght/2022CVPR/Rankingnet_IE/center_loss_premodel/CIFAR/checkpoint.pth.tar'
    #test_weights_center = '/home/ght/2022CVPR/Rankingnet_IE/center_loss_premodel/tiered/checkpoint.pth.tar'
    test_weights_center = '/home/ght/2022CVPR/Rankingnet_IE/center_loss_premodel/miniImagenet/checkpoint.pth.tar'
    #center_model = LinearModule(640,100).cuda()
    #test_weights_center = '/home/ght/2022CVPR/Rankingnet_IE/center_loss_premodel/CUB/checkpoint.pth.tar'
    checkpoint = torch.load(test_weights_center)
    center_model.load_state_dict(checkpoint['net'])
    center_model.eval()

    #train rank_net
    #rank_model = Linear_rankModule(16384,1).cuda()
    rank_model = Linear_rankModule(16384,1).cuda()
    #test_weights = '/home/ght/2022CVPR/Rankingnet_IE/asset/CIFAR_KR_best_model/checkpoint.pth.tar'
    #test_weights = '/home/ght/2022CVPR/Rankingnet_IE/asset/tiered_KR_best_model/checkpoint.pth.tar'
    test_weights = '/home/ght/2022CVPR/Rankingnet_IE/asset/miniImagenet_KR_best_model/checkpoint.pth.tar'
    #test_weights = '/home/ght/2022CVPR/Rankingnet_IE/asset/CUB_KR_best_model/checkpoint.pth.tar'
    checkpoint = torch.load(test_weights)
    rank_model.load_state_dict(checkpoint['net'])
    rank_model.eval()
    print("eval")
    eval()


#python3 eval_5shot.py --model_t resnet12 --path_t /home/ght/2022CVPR/Rankingnet_IE/IE_premodel/miniImagenet/mini_premodel.pth --dataset miniImageNet --data_root /home/ght/2022CVPR/Rankingnet_IE/data/miniImageNet
#python3 eval_5shot.py --model_t resnet12 --path_t /home/ght/2022CVPR/Rankingnet_IE/IE_premodel/CUB/CUB_premodel.pth --dataset CUB --data_root /home/ght/2022CVPR/Rankingnet_IE/data/CUB
#python3 eval_5shot.py --model_t resnet12 --path_t /home/ght/2022CVPR/Rankingnet_IE/IE_premodel/tiered/tiered_premodel.pth --dataset tieredImageNet --data_root /home/ght/2022CVPR/Rankingnet_IE/data/tiered/tieredImageNet
#python3 eval_5shot.py --model_t resnet12 --path_t /home/ght/2022CVPR/Rankingnet_IE/IE_premodel/CIFAR/CIFAR_premodel.pth --dataset CIFAR-FS --data_root /home/ght/2022CVPR/Rankingnet_IE/data/CIFAR
