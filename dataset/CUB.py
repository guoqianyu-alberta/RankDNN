from __future__ import print_function

import os
import pickle
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import cv2


class CUB(Dataset):

    def __init__(self, args, partition='train', pretrain=True, is_sample=False, k=4096,
                 transform=None):

        super(CUB, self).__init__()
        self.data_root = args.data_root
        self.partition = partition
        self.data_aug = args.data_aug
        self.mean = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        self.std = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        self.pretrain = pretrain
        self.file_pattern = 'cub_%s_path.pickle' 

        self.data = {}

        self.imgs = []
        self.labels = []
        with open(os.path.join(self.data_root, self.file_pattern % partition), 'rb') as f:
            data = pickle.load(f)
            labels_class = -1
            # for each class
            for c_idx in data:
                #if labels_class >= 9:
                    #break
                labels_class += 1
                #num_of_pics = 0
                # for each image
                for i_idx in range(len(data[c_idx])):
                    #if num_of_pics >= 10:
                        #break

                    # resize
                    image_data = os.path.join(self.data_root, self.partition, data[c_idx][i_idx])                   

                    if labels_class in self.data:
                        self.data[labels_class].append(image_data)
                    else:
                        empty_dict = []
                        empty_dict.append(image_data)
                        self.data[labels_class] = empty_dict

                    # save data path
                    self.imgs.append(image_data)
                    self.labels.append(labels_class)
                    
                    #num_of_pics += 1
            print(labels_class)        
        #print(self.imgs)            
        
        # pre-process for contrastive sampling
        self.k = k
        self.is_sample = is_sample        

    def transform_sample(self, img, indx=None):
        if indx is not None:
            out = transforms.functional.resized_crop(img, indx[0], indx[1], indx[2], indx[3], (84,84))
        else:
            out = img
        out = self.color_transform(out)
        out = transforms.RandomHorizontalFlip()(out)
        out = transforms.functional.to_tensor(out)
        out = self.normalize(out)
        return out

    def __getitem__(self, item):

        #open the picture
        img = cv2.imread(self.imgs[item])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

        img = transforms.Resize([126,126])(Image.fromarray(img))
        img = transforms.CenterCrop(84)(img)

        #transform
        if self.partition == 'train':
            img = transforms.RandomResizedCrop(84)(Image.fromarray(img))
        else:
            img = transforms.RandomResizedCrop(84)(Image.fromarray(img))

        img2 = self.transform_sample(img, [np.random.randint(28), 0, 56, 84])
        img3 = self.transform_sample(img, [0, np.random.randint(28), 84, 56])
        img4 = self.transform_sample(img, [np.random.randint(28), np.random.randint(28), 56, 56])

        if self.partition == 'train':
            img = self.transform_sample(img)
        else:
            img = transforms.functional.to_tensor(img)
            img = self.normalize(img)

        target = self.labels[item] 
        
        return img, img2, img3, img4, target, item

    def __len__(self):
        return len(self.labels)




class CUB_v1(Dataset):

    def __init__(self, args, partition='train', pretrain=True, is_sample=False, k=4096,
                 transform=None):

        super(CUB, self).__init__()
        self.data_root = args.data_root
        self.partition = partition
        self.mean = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        self.std = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        self.pretrain = pretrain
        self.simclr = args.simclr
        self.file_pattern = 'cub_%s_path.pickle' 

        self.data = {}

        self.imgs = []
        self.labels = []
        with open(os.path.join(self.data_root, self.file_pattern % partition), 'rb') as f:
            data = pickle.load(f)
            labels_class = -1
            # for each class
            for c_idx in data:
                labels_class += 1
                # for each image
                for i_idx in range(len(data[c_idx])):
                    # resize
                    image_data = os.path.join(self.data_root, self.partition, data[c_idx][i_idx])

                    if labels_class in self.data:
                        self.data[labels_class].append(image_data)
                    else:
                        empty_dict = []
                        empty_dict.append(image_data)
                        self.data[labels_class] = empty_dict

                    # save data path
                    self.imgs.append(image_data)
                    self.labels.append(labels_class)

        # pre-process for contrastive sampling
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            self.labels = np.asarray(self.labels)
            self.labels = self.labels - np.min(self.labels)
            num_classes = np.max(self.labels) + 1

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(len(self.imgs)):
                self.cls_positive[self.labels[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)        

    def transform_sample(self, img, indx=None):
        if indx is not None:
            out = transforms.functional.resized_crop(img, indx[0], indx[1], indx[2], indx[3], (84,84))
        else:
            out = img
        out = self.color_transform(out)
        out = transforms.RandomHorizontalFlip()(out)
        out = transforms.functional.to_tensor(out)
        out = self.normalize(out)
        return out

    def __getitem__(self, item):

        #open the picture
        img = cv2.imread(self.imgs[item])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       
        #transform
        if self.partition == 'train':
            img = transforms.RandomCrop(84, padding=8)(Image.fromarray(img))
        else:
            img = transforms.RandomCrop(84, padding=8)(Image.fromarray(img))

        img2 = self.transform_sample(img, [np.random.randint(28), 0, 56, 84])
        img3 = self.transform_sample(img, [0, np.random.randint(28), 84, 56])
        img4 = self.transform_sample(img, [np.random.randint(28), np.random.randint(28), 56, 56])

        if self.partition == 'train':
            img = self.transform_sample(img)
        else:
            img = transforms.functional.to_tensor(img)
            img = self.normalize(img)

        target = self.labels[item] - min(self.labels)
        
        if not self.is_sample:
            return img, img2, img3, img4, target, item
        else:
            pos_idx = item
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, item, sample_idx

    def __len__(self):
        return len(self.labels)


class CUBRank(Dataset):

    def __init__(self, args, partition='train', pretrain=True, is_sample=False, k=4096,
                 transform=None):

        super(CUBRank, self).__init__()
        self.data_root = args.data_root
        self.partition = partition
        self.mean = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        self.std = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        self.pretrain = pretrain
        self.file_pattern = 'cub_%s_path.pickle' 

        self.data = {}

        self.imgs = []
        self.labels = []
        with open(os.path.join(self.data_root, self.file_pattern % partition), 'rb') as f:
            data = pickle.load(f)
            labels_class = -1
            # for each class
            for c_idx in data:
                labels_class += 1
                # for each image
                for i_idx in range(len(data[c_idx])):
                    # resize
                    image_data = os.path.join(self.data_root, self.partition, data[c_idx][i_idx])

                    if labels_class in self.data:
                        self.data[labels_class].append(image_data)
                    else:
                        empty_dict = []
                        empty_dict.append(image_data)
                        self.data[labels_class] = empty_dict

                    # save data path
                    self.imgs.append(image_data)
                    self.labels.append(labels_class)

        self.k = k
        self.is_sample = is_sample     

    def __getitem__(self, item):

        #open the picture
        img = cv2.imread(self.imgs[item])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       
        #transform
        if self.partition == 'train':
            img = transforms.RandomResizedCrop(84)(Image.fromarray(img))
        else:
            img = transforms.RandomResizedCrop(84)(Image.fromarray(img))

        img = transforms.functional.to_tensor(img)
        img = self.normalize(img)
        target = self.labels[item] - min(self.labels)
        
        return img, target, item

    def __len__(self):
        return len(self.labels)


class MetaCUB(CUB_v1):

    def __init__(self, args, partition='train', train_transform=None, test_transform=None, fix_seed=True):
        super(MetaCUB, self).__init__(args, partition, False)
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.classes = list(self.data.keys())
        self.n_test_runs = args.n_test_runs
        self.n_aug_support_samples = args.n_aug_support_samples  

        if train_transform is None:
            self.train_transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.RandomResizedCrop(84),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.train_transform = train_transform

        if test_transform is None:
            self.test_transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.test_transform = test_transform

    def get_imgs(self, path):
        #open the picture
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transforms.RandomResizedCrop(84)(Image.fromarray(img))
        img = np.array(img) 
        return img 

    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)

        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls])
            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
            for it in range(self.n_shots):
                support_xs.append(self.get_imgs(imgs[support_xs_ids_sampled][it]))
            support_ys.append([idx] * self.n_shots)

            query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            for it in range(self.n_queries):
                query_xs.append(self.get_imgs(imgs[query_xs_ids][it]))
            query_ys.append([idx] * query_xs_ids.shape[0])

        support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(query_xs), np.array(query_ys)

        _, height, width, channel = query_xs.shape    
        query_ys = query_ys.reshape((_, ))          
        support_xs = support_xs.reshape((-1, height, width, channel))


        if self.n_aug_support_samples > 1:
            support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1, 1, 1))
            support_ys = np.tile(support_ys.reshape((-1, )), (self.n_aug_support_samples))
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = query_xs.reshape((-1, height, width, channel))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)
        
        support_xs = torch.stack(list(map(lambda x: self.train_transform(x.squeeze()), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(x.squeeze()), query_xs)))
      
        return support_xs, support_ys, query_xs, query_ys 

    def __len__(self):
        return self.n_test_runs


if __name__ == '__main__':
    args = lambda x: None
    args.n_ways = 5
    args.n_shots = 5
    args.n_queries = 12
    args.data_root = '/home/guoqy/data/Fewshot_Learning/cub/CUB'
    args.data_aug = True
    args.n_test_runs = 5
    args.n_aug_support_samples = 1
    cub = CUB(args, 'val')
    cub.__getitem__(500)

    print(len(cub))
    print(cub.__getitem__(500)[0].shape)

    metacub = MetaCUB(args, 'train')
    print("Call MetaCub")
    metacub.__getitem__(500)
    os._exit(1)

    print(metacub.__getitem__(500)[0].size())
    print(metacub.__getitem__(500)[1].shape)
    print(metacub.__getitem__(500)[2].size())
    print(metacub.__getitem__(500)[3].shape)
   
   
        
                
        