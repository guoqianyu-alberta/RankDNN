from __future__ import print_function
import os
import argparse
import socket
import time
import sys
from torch._C import CudaBFloat16StorageBase
from tqdm import tqdm
#import mkl
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
from dataset.mini_imagenet import ImageNet, ImageNetRank, MetaImageNet, ImageNet_v1
from dataset.cifar import CIFAR100, MetaCIFAR100,CIFAR100_v1, CIFAR_FSRank
from dataset.CUB import CUB, MetaCUB, CUBRank,CUB_v1
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet,tiered_imagenet_v1
from dataset.transform_cfg import transforms_options, transforms_list
from util import adjust_learning_rate, accuracy, AverageMeter, rotrate_concat, Logger, generate_final_report
import numpy as np
#import wandb
from dataloader import get_dataloaders
import copy
import h5py
from torch.autograd import Variable

##############################center##################
from models.center_module import LinearModule

########################################rank#################
from rank_module import Linear_rankModule, Ranknet, Ranknet_v1, GroupLinear
from rank_data import Set_rankdata, Set_rankdata_v1
from meta_rank_eval import meta_test_mul, meta_test_nshot

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    # dataset and model
    parser.add_argument('--model_s', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--model_t', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet','CIFAR-FS', 'FC100','CUB'])
    parser.add_argument('--batch_size', type=int, default=100, help='batch_size')
    # path to teacher model
    parser.add_argument('--path_t', type=str, default="", help='teacher model snapshot')

    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    # specify folder
    parser.add_argument('--model_path', type=str, default='save/', help='path to save model')
    parser.add_argument('--data_root', type=str, default='/raid/data/IncrementLearn/imagenet/Datasets/MiniImagenet/', help='path to data root')
    parser.add_argument('--trans', type=int, default=16, help='number of transformations')
    parser.add_argument('--memfeature_size', type=int, default=64, help='temperature for contrastive ssl loss')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=100, metavar='N', help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N', help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N', help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N', help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=1, type=int, help='The number of augmented samples for each meta test sample')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size', help='Size of test batch)')
    parser.add_argument('-t', '--trial', type=str, default='1', help='the experiment id')

    opt = parser.parse_args()
    return opt


class BalanceSample(Sampler):
    def __init__(self, dataset, args):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.batch_size = args.batch_size

    def __iter__(self):
        indices = []
        origin = np.array(range(len(self.dataset)))
        every = self.batch_size // self.dataset.num_classes
        label = np.array(self.dataset.labels)



        parts = 600 // every
        store = np.zeros([parts, 64, every])
        for n in range(64):
            select_idx = np.array(np.where((label) == n))
            start = int(select_idx[0][0])
            for m in range(parts):
                select_result = list(range(start + every * m, start + every * (m + 1)))
                # print(select_result)
                store[m, n, :] = np.array(select_result)
                # print(store[n, :, m])
        store_t = torch.from_numpy(store)
        store = store_t.view(parts * 64, every).numpy()
        sort_r = random.sample(range(0, parts * 64), parts * 64)
        store = store[sort_r]
        for i in range(parts * 64):
            indices.extend(list(map(int, store[i, :])))
        # print(indices)
        '''
        every_class_num = (self.batch_size // self.dataset.num_classes)
        result = random.sample(range(0, 64), self.dataset.num_classes)
        for n in range(self.dataset.num_classes):
            #print(self.dataset.labels)
            select_ind = np.where(np.array(self.dataset.labels) == int(result[n]))
            select_ind = np.array(select_ind)
            start = int(select_ind[0][0])
            print(start)
            select_result = random.sample(range(start, (start+600)), every_class_num)
            print(select_result)
            indices.extend(select_result)
        print(indices)
        '''
        # indices = torch.cat(indices, dim=0)

        return iter(indices)

    def __len__(self):
        return len(self.dataset)


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


def my_get_dataloaders(opt, partition, pretrain, num_classes):
    # dataloader
    the_sample = BalanceSample(ImageNetRank(args=opt, partition=partition, transform=None, num_classes=num_classes), args=opt)

    if opt.dataset == 'miniImageNet':
        loader = DataLoader(ImageNetRank(args=opt, partition=partition, transform=None, pretrain = pretrain),
                                  batch_size=opt.batch_size, shuffle=False, drop_last=False, sampler=the_sample,
                                  num_workers=opt.num_workers)
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
        loader = DataLoader(CIFAR100_v1(args=opt, partition=partition, transform=None),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=False,
                                  num_workers=opt.num_workers)
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
        loader = DataLoader(tiered_imagenet_v1(args=opt, partition=partition, transform=None),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=False,
                                  num_workers=opt.num_workers)
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
        loader = DataLoader(CUB_v1(args=opt, partition='train', transform=None),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=False,
                                  num_workers=opt.num_workers)
        meta_testloader = DataLoader(MetaCUB(args=opt, partition='test'),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCUB(args=opt, partition='val'),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)  
        meta_trainloader = DataLoader(MetaCUB(args=opt, partition='train'),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers) 

    return loader, meta_trainloader, meta_valloader, meta_testloader


def save_checkpoint(state, best_model):
    torch.save(state, 'asset/checkpoints/{}/group=8'.format(experiment) + 'checkpoint.pth.tar')
    if best_model:
        shutil.copyfile('asset/checkpoints/{}/group=8'.format(experiment) + 'checkpoint.pth.tar',
                        'asset/checkpoints/{}/group=8'.format(experiment) + 'model_best.pth.tar')


def set_exp_name(dataset,train_batch_size,train_epochs):
    exp_name = 'D-{}'.format(dataset)
    exp_name += '_L-{}_B-{}-group=8'.format(train_batch_size, train_epochs)
    return exp_name


def eval():
    #trian_acc
    start = time.time()
    meta_train_acc, meta_train_std = meta_test_mul(IE_model, center_model, rank_model, meta_testloader)
    test_time = time.time() - start
    print('Meta Train Acc : {:.4f}, Meta Train std: {:.4f}, Time: {:.1f}'.format(meta_train_acc, meta_train_std, test_time))


def train():
    best_perf = 0.0

    for epoch in range(rank_train_epochs):
        loss_sigma = 0.0 #每个epoch的总loss
        #train center_loss:Wrapper_model(IEpretrain_model,loader=imgloader,batchsize=opt.batchsize)

        for i, data in enumerate(loader):
            # data-premodel-[batch.size*640]
            img, target, item = data
            #print("iterator {0}: {1}\n".format(i, target))
            target = target.cuda()
            img = img.cuda()
            img_var = Variable(img)
            center_data = IE_model(img_var)  #center_data.shape=([batch.size,640])


            #feature640- center_model-128feature
            center_features, class_output = center_model(center_data)

            #train rank_net
            pair_num = 10000
            #max_item = target.size()[0]
            rank_train, rank_labels = Set_rankdata(target, pair_num, center_features)
            #print(rank_train.shape)
            #rank_train = rank_train.unsqueeze(-1).unsqueeze(-1)
            #print(rank_train.shape)
            rank_labels = torch.Tensor(rank_labels).cuda()
            rank_train = torch.Tensor([item.cpu().detach().numpy() for item in rank_train]).cuda()

            optimizer.zero_grad()
            outputs = rank_model(rank_train).cuda()
            outputs = outputs.squeeze(-1)
            loss = criterion(outputs, rank_labels)
            loss.backward()
            optimizer.step()
            loss_sigma += loss.item()

            # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
            if i % 10 == 9:
                loss_avg = loss_sigma / 10
                loss_sigma = 0.0
                print("Training : Epoch [{}/{}] Iteration [{}/{}], rank_Loss: {:.4f}".format(epoch+1,rank_train_epochs, i+1, len(loader), loss_avg))


        state = {'net':rank_model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch+155}

        dataset = 'miniImageNet_withoutcenter'
        if not os.path.exists('asset/{}/pre_model/{}/'.format(dataset,epoch+155)):
            os.makedirs('asset/{}/pre_model/{}/'.format(dataset,epoch+155))
        torch.save(state, 'asset/{}/pre_model/{}/'.format(dataset,epoch+155) + 'checkpoint.pth.tar')

        #trian_acc
        start = time.time()
        meta_train_acc, meta_train_std = meta_test_mul(IE_model, center_model, rank_model, meta_trainloader)
        test_time = time.time() - start
        print('Meta Train Acc : {:.4f}, Meta Train std: {:.4f}, Time: {:.1f}'.format(meta_train_acc, meta_train_std, test_time))

        #val_acc
        start = time.time()
        meta_val_acc, meta_val_std = meta_test_mul(IE_model, center_model, rank_model, meta_valloader)
        test_time = time.time() - start
        print('Meta Val Acc : {:.4f}, Meta Val std: {:.4f}, Time: {:.1f}'.format(meta_val_acc, meta_val_std, test_time))

        #evaluate
        start = time.time()
        meta_test_acc, meta_test_std = meta_test_mul(IE_model, center_model, rank_model, meta_testloader)
        test_time = time.time() - start
        print('Meta Test Acc: {:.4f}, Meta Test std: {:.4f}, Time: {:.1f}'.format(meta_test_acc, meta_test_std, test_time)) 


if __name__ == '__main__':
    opt = parse_option()

    #rank_parameters
    rank_train_epochs =1000
    input_size = 128*128
    num_classes = 1
    model_learning_rate = 0.001
    experiment = set_exp_name(opt.dataset,opt.batch_size,rank_train_epochs)
    if not os.path.exists('asset/checkpoints'):
        os.makedirs('asset/checkpoints')
    if not os.path.exists('asset/checkpoints/' + experiment):
        os.makedirs('asset/checkpoints/' + experiment)

    model_state_file = 'asset/checkpoints/{}/'.format(experiment) + 'checkpoint.pth.tar'


    #load image
    partition = 'train.pickle'
    pretrain = True
    opt = parse_option()
    num_class = 10
    loader, meta_trainloader, meta_valloader, meta_testloader = my_get_dataloaders(opt, partition, pretrain, num_class)
    print("=====>loaderdata success")

    # IEmodel
    feature_model = load_model(opt.path_t, opt.model_t, 64, opt.dataset, opt.trans, opt.memfeature_size)
    IE_model = Wrapper(feature_model).cuda()
    IE_model.eval()

    #center_model
    center_model = LinearModule(640,64).cuda()
    test_weights_center = '/home/cs20-guoqy/ECCV2022/Rankingnet_IE/center_loss_premodel/miniImagenet/checkpoint.pth.tar'
    checkpoint = torch.load(test_weights_center)
    center_model.load_state_dict(checkpoint['net'])
    center_model.eval()

    #train rank_net
    #rank_model = Linear_rankModule(input_size,num_classes).cuda()
    rank_model = GroupLinear().cuda()
    #test_weights = '/home/ght/2022CVPR/Rankingnet_IE/asset/miniImageNet_IE/pre_model/152/checkpoint.pth.tar'
    #checkpoint = torch.load(test_weights)
    #rank_model.load_state_dict(checkpoint['net'])
    criterion = nn.BCELoss().cuda()
    #optimizer = torch.optim.Adam(rank_model.parameters(),lr=learning_rate,weight_decay=1e-5)
    optimizer = torch.optim.SGD(rank_model.parameters(), lr=model_learning_rate, momentum=0.9, weight_decay=1e-6)
    rank_model.train()

    train()
    #rank_model.eval()

    #eval()

#python3 train_rank.py --model_t resnet12 --path_t /home/cs20-guoqy/ECCV2022/Rankingnet_IE/IE_premodel/CUB/CUB_premodel.pth --dataset CUB --data_root /home/cs20-guoqy/ECCV2022/Rankingnet_IE/data/CUB/
#python3 train_rank.py --model_t resnet12 --path_t /home/cs20-guoqy/ECCV2022/Rankingnet_IE/IE_premodel/miniImagenet/mini_premodel.pth --dataset miniImageNet --data_root /home/cs20-guoqy/ECCV2022/Rankingnet_IE/data/miniImageNet/
#python3 train_rank.py --model_t resnet12 --path_t /home/cs20-guoqy/ECCV2022/Rankingnet_IE/IE_premodel/CIFAR/CIFAR_premodel.pth --dataset CIFAR-FS --data_root /home/cs20-guoqy/ECCV2022/Rankingnet_IE/data/CIFAR/