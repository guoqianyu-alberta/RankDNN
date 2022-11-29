import torch
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import scipy
from scipy.stats import t
from rank_data import Set_rankdata
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import torch.nn as nn
import sys, os
from collections import Counter


sys.path.append(os.path.abspath('..'))

from util import accuracy

def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out


def Cosine(support, support_ys, query):
    """Cosine classifier"""
    support_norm = np.linalg.norm(support, axis=1, keepdims=True)
    support = support / support_norm
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    query = query / query_norm

    cosine_distance = query @ support.transpose()
    max_idx = np.argmax(cosine_distance, axis=1)
    pred = [support_ys[idx] for idx in max_idx]
    return pred  

 


def mean_confidence_interval(data, confidence=0.95):
    a = 100.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h


def meta_test_mul(premodel, center_model, rank_model, testloader, use_logit=False, is_norm=True):
    premodel = premodel.eval()
    center_model = center_model.eval()
    rank_model = rank_model.eval()
    acc = []

    with torch.no_grad():
        with tqdm(testloader, total=len(testloader)) as pbar:
            for idx, data in enumerate(pbar):
                support_xs, support_ys, query_xs, query_ys = data
                support_xs = support_xs.cuda()
                query_xs = query_xs.cuda()
                batch_size, _, height, width, channel = support_xs.size()
                support_xs = support_xs.view(-1, height, width, channel)
                query_xs = query_xs.view(-1, height, width, channel)


                support_features = premodel(support_xs).view(support_xs.size(0), -1)
                query_features = premodel(query_xs).view(query_xs.size(0), -1)
                support_features, _ = center_model(support_features)
                query_features, _ = center_model(query_features)

                if is_norm:
                    support_features = normalize(support_features)
                    query_features = normalize(query_features)

                support_features = support_features.detach().cpu().numpy()
                query_features = query_features.detach().cpu().numpy()                
                support_ys = support_ys.view(-1).numpy()
                query_ys = query_ys.view(-1).numpy()


                #### IE Model Test ####
                # clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, penalty='l2',
                #                              multi_class='multinomial')
                # clf.fit(support_features, support_ys)
                # query_ys_pred = clf.predict(query_features)                
                # acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
                # pbar.set_postfix({"FSL_Acc":'{0:.2f}'.format(metrics.accuracy_score(query_ys, query_ys_pred))})


                ################rank_eval###################
   
                support_features = torch.tensor(support_features, dtype=torch.float32).cuda()
                query_features = torch.tensor(query_features, dtype=torch.float32).cuda()
                predicted_label = []
                n_lsamples = 5 #n_way*n_shot
                for k in range(0, query_features.shape[0]):
                    a_feat = torch.unsqueeze(query_features[k],dim=1)
                    pairwise_feat = []
                    for i in range(n_lsamples):
                        for j in range(n_lsamples):
                            p_feat = torch.unsqueeze(support_features[i],dim=1)
                            n_feat = torch.unsqueeze(support_features[j],dim=1)
                            #ap_feat_1 = torch.cat((a_feat,p_feat),0).squeeze(-1)
                            #ap_feat_1 = cat1_model(ap_feat_1).squeeze(-1)                      
                            ap_feat= torch.matmul(a_feat,p_feat.T).flatten()
                            #ap_feat = torch.cat((ap_feat_1,ap_feat_2),0)
                            #an_feat_1 = torch.cat((a_feat,n_feat),0).squeeze(-1)
                            #an_feat_1 = cat2_model(an_feat_1).squeeze(-1)                       
                            an_feat= torch.matmul(a_feat,n_feat.T).flatten()
                            #an_feat = torch.cat((an_feat_1,an_feat_2),0)
                            feat = ap_feat - an_feat   
                            feat = torch.squeeze(feat)
                            #feat = feat.unsqueeze(-1).unsqueeze(-1)                        
                            pairwise_feat.append(feat)
                    pairwise_feat = torch.stack(pairwise_feat,dim=0) 

                    predicts = rank_model(pairwise_feat)
                    predicts = predicts.squeeze(-1)
                    predicts = predicts.cpu()

                    for i in range(25):
                        if predicts[i] >= 0.5:
                            predicts[i] = 1
                        else:
                            predicts[i] = 0

                    predicts = predicts.reshape(n_lsamples, n_lsamples)
                    predicts=predicts.detach().numpy()
                    for i in range(n_lsamples):
                        predicts[i,i] = 0
                    predicts = np.sum(predicts,axis=1)
                    prediction = predicts.argmin()
                    predicted_label.append(prediction)
                predicted_label = np.array(predicted_label)
                acc_n = np.mean(predicted_label == query_ys)
                acc.append(acc_n)
          
                #query_ys_pred = Cosine(support_features, support_ys, query_features)

                #acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
                
                pbar.set_postfix({"FSL_Acc":'{0:.2f}'.format(acc_n)}) 
    
    return mean_confidence_interval(acc) 




def meta_test_cat_mul(premodel, center_model, cat1_model,cat2_model, rank_model, testloader, use_logit=False, is_norm=True):
    premodel = premodel.eval()
    center_model = center_model.eval()
    cat1_model = cat1_model.eval()
    cat2_model = cat2_model.eval()
    rank_model = rank_model.eval()
    acc = []

    with torch.no_grad():
        with tqdm(testloader, total=len(testloader)) as pbar:
            for idx, data in enumerate(pbar):
                support_xs, support_ys, query_xs, query_ys = data
                support_xs = support_xs.cuda()
                query_xs = query_xs.cuda()
                batch_size, _, height, width, channel = support_xs.size()
                support_xs = support_xs.view(-1, height, width, channel)
                query_xs = query_xs.view(-1, height, width, channel)


                support_features = premodel(support_xs).view(support_xs.size(0), -1)
                query_features = premodel(query_xs).view(query_xs.size(0), -1)
                support_features, _ = center_model(support_features)
                query_features, _ = center_model(query_features)

                if is_norm:
                    support_features = normalize(support_features)
                    query_features = normalize(query_features)

                support_features = support_features.detach().cpu().numpy()
                query_features = query_features.detach().cpu().numpy()                
                support_ys = support_ys.view(-1).numpy()
                query_ys = query_ys.view(-1).numpy()

                ################rank_eval###################

                support_features = torch.tensor(support_features, dtype=torch.float32).cuda()
                query_features = torch.tensor(query_features, dtype=torch.float32).cuda()
                predicted_label = []
                n_lsamples = 5 #n_way*n_shot
                for k in range(0, query_features.shape[0]):
                    a_feat = torch.unsqueeze(query_features[k],dim=1)
                    pairwise_feat = []
                    for i in range(n_lsamples):
                        for j in range(n_lsamples):
                            p_feat = torch.unsqueeze(support_features[i],dim=1)
                            n_feat = torch.unsqueeze(support_features[j],dim=1)
                            ap_feat_1 = torch.cat((a_feat,p_feat),0).squeeze(-1)
                            ap_feat_1 = cat1_model(ap_feat_1).squeeze(-1)                      
                            ap_feat_2= torch.mul(a_feat,p_feat).flatten()
                            ap_feat = torch.cat((ap_feat_1,ap_feat_2),0)
                            an_feat_1 = torch.cat((a_feat,n_feat),0).squeeze(-1)
                            an_feat_1 = cat2_model(an_feat_1).squeeze(-1)                       
                            an_feat_2= torch.mul(a_feat,n_feat).flatten()
                            an_feat = torch.cat((an_feat_1,an_feat_2),0)
                            feat = ap_feat - an_feat   
                            feat = torch.squeeze(feat)                        
                            pairwise_feat.append(feat)
                    pairwise_feat = torch.stack(pairwise_feat,dim=0) 

                    predicts = rank_model(pairwise_feat)
                    predicts = predicts.squeeze(-1)
                    predicts = predicts.cpu()

                    for i in range(25):
                        if predicts[i] >= 0.5:
                            predicts[i] = 1
                        else:
                            predicts[i] = 0

                    predicts = predicts.reshape(n_lsamples, n_lsamples)
                    predicts=predicts.detach().numpy()
                    for i in range(n_lsamples):
                        predicts[i,i] = 0
                    predicts = np.sum(predicts,axis=1)
                    prediction = predicts.argmin()
                    predicted_label.append(prediction)
                predicted_label = np.array(predicted_label)
                acc_n = np.mean(predicted_label == query_ys)
                acc.append(acc_n)
          
                #query_ys_pred = Cosine(support_features, support_ys, query_features)

                #acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
                
                pbar.set_postfix({"FSL_Acc":'{0:.2f}'.format(acc_n)})
    
    return mean_confidence_interval(acc) 




def meta_test_mul_5(premodel, center_model, rank_model, testloader, use_logit=False, is_norm=True):
    premodel = premodel.eval()
    center_model = center_model.eval()
    rank_model = rank_model.eval()
    acc = []

    with torch.no_grad():
        with tqdm(testloader, total=len(testloader)) as pbar:
            for idx, data in enumerate(pbar):
                support_xs, support_ys, query_xs, query_ys = data
                support_xs = support_xs.cuda()
                query_xs = query_xs.cuda()
                batch_size, _, height, width, channel = support_xs.size()
                support_xs = support_xs.view(-1, height, width, channel)
                query_xs = query_xs.view(-1, height, width, channel)


                support_features = premodel(support_xs).view(support_xs.size(0), -1)
                query_features = premodel(query_xs).view(query_xs.size(0), -1)
                support_features, _ = center_model(support_features)
                query_features, _ = center_model(query_features)

                if is_norm:
                    support_features = normalize(support_features)
                    query_features = normalize(query_features)

                support_features = support_features.detach().cpu().numpy()
                query_features = query_features.detach().cpu().numpy()                
                support_ys = support_ys.view(-1).numpy()
                query_ys = query_ys.view(-1).numpy()

                ################rank_eval###################

                support_features = torch.tensor(support_features, dtype=torch.float32).cuda()
                query_features = torch.tensor(query_features, dtype=torch.float32).cuda()
                predicted_label = []
                n_lsamples = 25
                for k in range(0, query_features.shape[0]):
                    score_support = []
                    a_feat = torch.unsqueeze(query_features[k],dim=1)
                    for i in range(n_lsamples):
                        pairwise_feat = []
                        for j in range(n_lsamples):
                            if i == j:
                                continue
                            p_feat = torch.unsqueeze(support_features[i],dim=1)
                            n_feat = torch.unsqueeze(support_features[j],dim=1)
                            ap_feat= torch.matmul(a_feat,p_feat.T).flatten()
                            an_feat= torch.matmul(a_feat,n_feat.T).flatten()
                            feat = ap_feat - an_feat
                            pairwise_feat.append(feat)
                        pairwise_feat = torch.stack(pairwise_feat,dim=0) 

                        predicts_sp = rank_model(pairwise_feat)
                        predicts_sp = predicts_sp.cpu()

                        for n in range(n_lsamples-1):
                            if predicts_sp[n] > 0.5:
                                predicts_sp[n] = 1
                            else:
                                predicts_sp[n] = 0

                        predicts_sp = predicts_sp.reshape(1, n_lsamples-1)
                        predicts_sp = predicts_sp.detach().numpy()
                        predicts_sp = np.sum(predicts_sp,axis=1)#predicts_sp为单个support的rank值
                        score_support.append(int(predicts_sp[0]))#score_support存所有support的rank值
                    arr_support = np.argsort(score_support).tolist()#将所有support的rank值从小到大排序的原下标输出为arr_support
                    predicted_label.append(support_ys[arr_support[0]])
                    '''if(arr_support[0])== max(arr_support,key=arr_support.count):
                        predicted_label.append(support_label[arr_support[0]])#如果排名最高的和前五中出现次数最多的一致，则输出
                    else:
                        predicted_arr=support_label[arr_support[:5]]
                        new_predicted_arr = []
                        for i in predicted_arr:
                            if i not in new_predicted_arr:
                                new_predicted_arr.append(i)
                        for num in new_predicted_arr:
                            label_weights = 0.0
                            best_weights = 0
                            for index, nums in enumerate(predicted_arr):
                                if nums == num:
                                    label_weights = label_weights + (5-index)
                                    if best_weights < label_weights:
                                        best_weights = label_weights
                                        pre_label = num
                        predicted_label.append(pre_label)#如果不一致，将每个位次乘以权重，输出权重最高的类别'''
                predicted_label = np.array(predicted_label)
                acc_n = np.mean(predicted_label == query_ys)
                acc.append(acc_n)
          
                #query_ys_pred = Cosine(support_features, support_ys, query_features)

                #acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
                
                pbar.set_postfix({"FSL_Acc":'{0:.2f}'.format(acc_n)})
    
    return mean_confidence_interval(acc) 









def FasterMakeTestPair(support_xs, query_xs, support_ys, Ways, Shots, is_label = False):

    A, P, N = [], [] ,[]
    PairLabel = []
    for k in range(0, query_xs.shape[0]):                       
        a_feat = torch.unsqueeze(query_xs[k], dim=1)
        for x in range(Ways):                                   
            for i in torch.arange(Shots * x, Shots * x + Shots, 1):         
                p_feat = torch.unsqueeze(support_xs[i], dim=1)
                for j in torch.arange(0, Ways * Shots, 1):
                    if support_ys[i] != support_ys[j]:
                        n_feat = torch.unsqueeze(support_xs[j], dim=1)
                        A.append(a_feat)
                        if not is_label:
                            P.append(p_feat.T)
                            N.append(n_feat.T)
                        else:
                            state = torch.randint(low=0, high=2, size=(1,))
                            PairLabel.append(state)
                            if state[0] == 1:
                                # Pair state = (a, p, n)
                                P.append(p_feat.T)
                                N.append(n_feat.T)
                            else:
                                # Pair state = (a, n, p)  
                                N.append(p_feat.T)
                                P.append(n_feat.T) 

    A, P, N = torch.stack(A), torch.stack(P), torch.stack(N)
    PairFeat = torch.einsum('ijk,ikl->ijl', [A, P]) - torch.einsum('ijk,ikl->ijl', [A, N])
    PairFeat = torch.reshape(PairFeat, (PairFeat.shape[0], -1))

    if not is_label:
        return PairFeat
    else:
        PairLabel = torch.stack(PairLabel).squeeze()
        return PairLabel, PairFeat     



def meta_test_nshot(opt, center_model, rank_model, testloader, is_norm=True):

    rank_model = rank_model.eval()

    acc = []
    with torch.no_grad():
        with tqdm(testloader, total=len(testloader)) as pbar:
            for idx, data in enumerate(pbar):
                
                support_xs, support_ys, query_xs, query_ys = data
                support_xs = torch.squeeze(support_xs)
                support_ys = torch.squeeze(support_ys)
                query_xs = torch.squeeze(query_xs)
                query_ys = torch.squeeze(query_ys)

                # pca
                support_xs = center_model.transform(support_xs)
                query_xs = center_model.transform(query_xs)
                support_xs = torch.from_numpy(support_xs)
                query_xs = torch.from_numpy(query_xs)


                if torch.cuda.is_available():
                    support_xs = support_xs.cuda()
                    support_ys = support_ys.cuda()
                    query_xs = query_xs.cuda()
                    query_ys = query_ys.cuda()

                support_features = support_xs
                query_features = query_xs

                # use norm
                support_features = support_features.reshape((5,-1,100))
                support_features = torch.sum(support_features, dim=1)
                support_ys = torch.tensor([0,1,2,3,4])

                if is_norm:
                    support_features = normalize(support_features)  
                    query_features = normalize(query_features)      # torch.Size([75, 640])

                support_features = support_features.cuda()
                query_features = query_features.cuda()

                pairwise_feat = FasterMakeTestPair(support_features, query_features, support_ys, opt.n_ways, 1)
                pairwise_feat = pairwise_feat.to(torch.float32)

                predicts_sp = rank_model(pairwise_feat)
                predicts_sp = predicts_sp.squeeze().reshape((75,5,-1))

                predicts_sp = torch.sum(predicts_sp, dim=2)
                predicted_label = torch.argmax(predicts_sp, dim=1)

                predicted_label = predicted_label.cpu().numpy()
                query_ys = query_ys.cpu().numpy()

                acc_n = np.mean(predicted_label == query_ys) 
                acc.append(acc_n)
                
                pbar.set_postfix({"FSL_Acc":'{0:.2f}'.format(acc_n)})


    return mean_confidence_interval(acc) 