from torch.autograd import Variable
import os
import numpy as np
import torch
import random




def Set_rankdata(labels, pair_num, sampled_data):

    X = []
    Y = []
    out, num = torch.unique(labels, return_counts=True)
    rep = torch.nonzero(num > 1).squeeze()
    list_a = out[rep]     # label数大于一次的label集合
    list_a = list_a[torch.randperm(len(list_a))]  # 打乱


    for lab in list_a:
        if len(X) >= pair_num:
            break
        # 选取一个label
        it = torch.nonzero(labels == lab).squeeze()
        jt = torch.nonzero(labels != lab).squeeze()
        a = it[0]
        for i in it[1:]:              
            for j in jt:
                if len(X) >= pair_num:
                    break
                a_feat = torch.unsqueeze(sampled_data[a],dim=1)
                i_feat = torch.unsqueeze(sampled_data[i],dim=1)
                j_feat = torch.unsqueeze(sampled_data[j],dim=1)

                x = [a_feat, i_feat, j_feat]
                x = torch.cat(x, dim = 1)
                feat = torch.matmul(x.T,x) / np.sqrt(128)
                sfeat = torch.nn.functional.softmax(feat, dim = 0)

                new_x = torch.matmul(x,sfeat)
                a_feat, i_feat, j_feat = torch.split(new_x, [1,1,1], dim = 1)

                ai_feat= torch.matmul(a_feat,i_feat.T).flatten()
                aj_feat= torch.matmul(a_feat,j_feat.T).flatten()
                positive = True
                if np.random.randint(0,2) == 1:
                    positive = False
                if positive == True:
                    feat = ai_feat - aj_feat
                    label= 0
                else:
                    feat = aj_feat - ai_feat
                    label= 1
                X.append(feat)
                Y.append(label)
                                          
    Y = torch.Tensor(Y)
    X = torch.Tensor([item.cpu().detach().numpy() for item in X]) 

    return X, Y




def Set_rankdata_v1(max_item, labels, pair_num, sampled_data):
    # kro x×y.flatten
    X = []
    Y = []
    while len(X) < pair_num:
        a = random.randint(0, max_item-1)
        for i in range(max_item):
            for j in range(max_item):
                if labels[a] == labels[i]:
                    if labels[a] != labels[j]:
                        
                        a_feat = torch.unsqueeze(sampled_data[a],dim=1)
                        i_feat = torch.unsqueeze(sampled_data[i],dim=1)
                        j_feat = torch.unsqueeze(sampled_data[j],dim=1)
                        ai_feat= torch.matmul(a_feat,i_feat.T).flatten()
                        aj_feat= torch.matmul(a_feat,j_feat.T).flatten()
                        positive = True
                        if np.random.randint(0,2) == 1:
                            positive = False
                        if positive == True:
                            feat = ai_feat - aj_feat
                            label= 0
                        else:
                            feat = aj_feat - ai_feat
                            label= 1
                        X.append(feat)
                        Y.append(label)

    return X, Y



def kronecker(A, B):
    A_B = torch.einsum("ab,cd->acbd", A, B)
    A_B = A_B.view(A.size(0)*B.size(0), A.size(1)*B.size(1))
    return A_B

