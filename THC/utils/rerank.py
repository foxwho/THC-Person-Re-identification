#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:46:56 2017
@author: luohao
"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API
probFea: all feature vectors of the query set, shape = (image_size, feature_dim)
galFea: all feature vectors of the gallery set, shape = (image_size, feature_dim)
k1,k2,lambda: parameters, the original paper is (k1=20,k2=6,lambda=0.3)
MemorySave: set to 'True' when using MemorySave mode
Minibatch: avaliable when 'MemorySave' is 'True'
"""


import numpy as np
from scipy.spatial.distance import cdist

def re_ranking(input_feature,k1=20,k2=6,lambda_value=0.2, MemorySave = False, Minibatch = 2000, no_rerank=False):

    # all_num_source  = input_feature_source.shape[0]
    #query_num = probFea.shape[0]
    all_num = input_feature.shape[0]    
    #feat = np.append(probFea,galFea,axis = 0)
    feat = input_feature.astype(np.float16)

    # print('computing source distance...')
    # sour_tar_dist = np.power(
    #     cdist(input_feature, input_feature_source), 2).astype(np.float16)
    # sour_tar_dist = 1-np.exp(-sour_tar_dist)
    # source_dist_vec = np.min(sour_tar_dist, axis = 1)
    # source_dist_vec = source_dist_vec / np.max(source_dist_vec)
    # source_dist = np.zeros([all_num, all_num])
    # for i in range(all_num):
    #     source_dist[i, :] = source_dist_vec + source_dist_vec[i]
    # del sour_tar_dist
    # del source_dist_vec


    print('computing original distance...')
    if MemorySave:
        original_dist = np.zeros(shape = [all_num,all_num],dtype = np.float16)
        i = 0
        while True:
            it = i + Minibatch
            if it < np.shape(feat)[0]:
                original_dist[i:it,] = np.power(cdist(feat[i:it,],feat),2).astype(np.float16)
            else:
                original_dist[i:,:] = np.power(cdist(feat[i:,],feat),2).astype(np.float16)
                break
            i = it
    else:
        original_dist = cdist(feat,feat).astype(np.float16)  
        original_dist = np.power(original_dist,2).astype(np.float16)
    del feat    
    euclidean_dist = original_dist
    if no_rerank:
        return euclidean_dist, None
    gallery_num = original_dist.shape[0] #gallery_num=all_num
    original_dist = np.transpose(original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)  ## default axis=-1.  

    
    print('starting re_ranking...')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]  ## k1+1 because self always ranks first. forward_k_neigh_index.shape=[k1+1].  forward_k_neigh_index[0] == i.
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]  ##backward.shape = [k1+1, k1+1]. For each ele in forward_k_neigh_index, find its rank k1 neighbors
        fi = np.where(backward_k_neigh_index==i)[0]  
        k_reciprocal_index = forward_k_neigh_index[fi]   ## get R(p,k) in the paper
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2/3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)
            
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])  
        V[i,k_reciprocal_expansion_index] = weight/np.sum(weight)
    #original_dist = original_dist[:query_num,]    
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float16)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = [] 
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])  #len(invIndex)=all_num
    
    jaccard_dist = np.zeros_like(original_dist,dtype = np.float16)

    
    for i in range(all_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float16)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2-temp_min)

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    del pos_bool

    # final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    # final_dist = jaccard_dist*(1-lambda_value) + source_dist*lambda_value
    final_dist = jaccard_dist
    del original_dist
    del V
    del jaccard_dist
    #final_dist = final_dist[:query_num,query_num:]
    return euclidean_dist, final_dist






#!/usr/bin/env python2/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:46:56 2017
@author: luohao
Modified by Houjing Huang, 2017-12-22. 
- This version accepts distance matrix instead of raw features. 
- The difference of `/` division between python 2 and 3 is handled.
- numpy.float16 is replaced by numpy.float32 for numerical precision.

Modified by Zhedong Zheng, 2018-1-12.
- replace sort with topK, which save about 30s.
"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API
q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]
k1, k2, lambda_value: parameters, the original paper is (k1=20, k2=6, lambda_value=0.3)
Returns:
  final_dist: re-ranked distance, numpy array, shape [num_query, num_gallery]
"""


def k_reciprocal_neigh( initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]

def re_ranking_init(query_feature, gallery_feature, k1=20, k2=6, lambda_value=0.3):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
    q_q_dist = np.dot(query_feature, np.transpose(query_feature))
    g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))

    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = 2. - 2 * original_dist   #np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    #initial_rank = np.argsort(original_dist).astype(np.int32)
    # top K1+1
    initial_rank = np.argpartition( original_dist, range(1,k1+1) )

    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh( initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh( initial_rank, candidate, int(np.around(k1/2)))
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)

    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1,all_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist


# def re_ranking_haus(input_feature_source, input_feature, k=20, lambda_value=0.1, MemorySave=False, Minibatch=2000):

#     all_num = input_feature.shape[0]
#     feat = input_feature.astype(np.float16)

#     print('computing original distance...')
#     if MemorySave:
#         original_dist = np.zeros(shape=[all_num, all_num], dtype=np.float16)
#         i = 0
#         while True:
#             it = i + Minibatch
#             if it < np.shape(feat)[0]:
#                 original_dist[i:it, ] = np.power(
#                     cdist(feat[i:it, ], feat), 2).astype(np.float16)
#             else:
#                 original_dist[i:, :] = np.power(
#                     cdist(feat[i:, ], feat), 2).astype(np.float16)
#                 break
#             i = it
#     else:
#         original_dist = cdist(feat, feat).astype(np.float16)
#         original_dist = np.power(original_dist, 2).astype(np.float16)
#     del feat

#     knn_bool = np.zeros([all_num, all_num], dtype=bool)
#     for i in range(all_num):
#         tem_vec = original_dist[i, :]
#         kThreshold = np.partition(tem_vec, k-1,)[k-1]
#         knn_bool[i, :] = (tem_vec < kThreshold)
#         knn_bool[i, np.nonzero(tem_vec == kThreshold)[0][0]] = True
#     del tem_vec

#     haus_dist = np.zeros(shape=[all_num, all_num], dtype=np.float16)
#     tem_min = np.zeros(k+1, dtype=np.float16)
#     for i in range(all_num-1):
#         for j in range(i+1, all_num):

#             for s_index, s in enumerate(np.nonzero(knn_bool[i, :])[0]):
#                 tem_min[s_index] = np.min(original_dist[s, knn_bool[j, :]])
#             ij_max = np.max(tem_min)
#             for s_index, s in enumerate(np.nonzero(knn_bool[j, :])[0]):
#                 tem_min[s_index] = np.min(original_dist[s, knn_bool[i, :]])
#             ji_max = np.max(tem_min)
#             haus_dist[i, j] = np.max([ij_max, ji_max])

#     haus_dist += haus_dist.T

#     del original_dist

#     return haus_dist
