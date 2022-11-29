import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in',nonlinearity='relu')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)



class Linear_rankModule(nn.Module):
    def __init__(self,input_size,num_classes):
        super(Linear_rankModule, self).__init__()
        self.linear1 = nn.Linear(input_size,8000)
        #self.bn1 = nn.BatchNorm1d(5000)
        self.linear2 = nn.Linear(8000,4000)
        #self.bn2 = nn.BatchNorm1d(1000)
        self.linear3 = nn.Linear(4000,2000)
        #self.bn3 = nn.BatchNorm1d(300)
        self.linear4 = nn.Linear(2000,1000)
        self.linear5 = nn.Linear(1000,num_classes)
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.linear1(x)
        #x = self.bn1(x)
        x = F.relu(x)
        #x = self.dropout(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.relu(x)
        x = self.linear5(x)
        x = self.sigmoid(x)
        return x


def channel_shuffle(x, groups):

    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)

    return x



class Ranknet(torch.nn.Module):
    def __init__(self,input_size,groups=4):
        super(Ranknet,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_size,
                            out_channels=4096,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=groups),
            torch.nn.BatchNorm2d(4096),
            torch.nn.ReLU()
        )
        self.conv2= torch.nn.Sequential(
            torch.nn.Conv2d(4096,2048,1,1,0),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU()
        )
        '''self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=2000,
                            out_channels=2000,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=groups),
            torch.nn.BatchNorm2d(2000),
            torch.nn.ReLU()
        )'''
        self.conv3= torch.nn.Sequential(
            torch.nn.Conv2d(2048,1024,1,1,0),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(1024,1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x = channel_shuffle(x, groups=2)
        x = self.conv1(x)
        #x = channel_shuffle(x, groups=4)
        x = self.conv2(x)
        #x = channel_shuffle(x, groups=100)
        x = self.conv3(x)
        #x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0),-1))
        x = self.sigmoid(x)
        return x

class Ranknet_v1(torch.nn.Module):
    def __init__(self,input_size,groups=8):
        super(Ranknet_v1,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_size,
                            out_channels=4096,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=groups),
            torch.nn.BatchNorm2d(4096),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=4096,
                            out_channels=2048,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=groups),
            torch.nn.BatchNorm2d(2048),
            torch.nn.ReLU()
        )
        '''self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=2000,
                            out_channels=2000,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=groups),
            torch.nn.BatchNorm2d(2000),
            torch.nn.ReLU()
        )'''
        self.conv3= torch.nn.Sequential(
            torch.nn.Conv2d(2048,1024,1,1,0),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU()
        )
        self.conv4= torch.nn.Sequential(
            torch.nn.Conv2d(1024,512,1,1,0),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(512,1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x = channel_shuffle(x, groups=2)
        x = self.conv1(x)
        #x = channel_shuffle(x, groups=4)
        x = self.conv2(x)
        #x = channel_shuffle(x, groups=100)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0),-1))
        x = self.sigmoid(x)
        return x





class ChannelShuffle(nn.Module):
    def __init__(self, dim):
        super(ChannelShuffle, self).__init__()
        self.shuffle_index = torch.randperm(dim).cuda()

    def forward(self, tensor):
        return tensor[:,self.shuffle_index]



class Group_MLP(nn.Module):
    def __init__(self, row, in_dim ,out_dim):
        super(Group_MLP, self).__init__()        
        self.mlp_list = nn.ModuleList([nn.Linear(in_dim,out_dim) for i in range(row)])

    def forward(self, tensor):
        out_list = []
        for i in range(tensor.shape[1]):
            out_list.append(self.mlp_list[i](tensor[:,i,:].unsqueeze(dim=1)))
        out = torch.cat(out_list, dim=1)
        #out = torch.Tensor([item.cpu().detach().numpy() for item in out]).cuda()
        return out


class GroupLinear(nn.Module):
    def __init__(self):
        super(GroupLinear, self).__init__()
        self.shuffle_1 = ChannelShuffle(128*128)
        self.shuffle_2 = ChannelShuffle(128*64)
        self.linear1 = Group_MLP(128,128,64)
        self.linear2 = Group_MLP(128,64,32)
        self.linear3 = nn.Linear(128*32,2048)
        self.linear4 = nn.Linear(2048,1024)
        self.linear5 = nn.Linear(1024,1)
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.shuffle_1(x)
        x = x.reshape(-1,128,128)
        x= F.relu(self.linear1(x))
        x= x.reshape(-1,128*64)
        x = self.shuffle_2(x)
        x = x.reshape(-1,128,64)
        x= F.relu(self.linear2(x))
        x = x.reshape(-1,128*32)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.sigmoid(x)

        return x
