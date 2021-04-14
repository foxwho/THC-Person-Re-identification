from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import copy
import time
from datetime import timedelta
import os

from sklearn.cluster import AgglomerativeClustering

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from spcl import datasets
from spcl import models
from spcl.models.dsbn import convert_dsbn, convert_bn
from spcl.models.hm import HybridMemory
from spcl.trainers import SpCLTrainer_UDA
from spcl.evaluators import Evaluator, extract_features
from spcl.utils.data import IterLoader
from spcl.utils.data import transforms as T
from spcl.utils.data.sampler import RandomMultipleGallerySampler
from spcl.utils.data.preprocessor import Preprocessor
from spcl.utils.logging import Logger
from spcl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
# from spcl.utils.faiss_rerank import compute_jaccard_distance
from spcl.utils.rerank import *


start_epoch = best_mAP = 0

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader(args, dataset, height, width, batch_size, workers,
                    num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
	         T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0)
    # adopt domain-specific BN
    convert_dsbn(model)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model

def compute_dist(target_features, k1,k2,lambda_value,no_rerank):

    euclidean_dist, rerank_dist = re_ranking(
        target_features.cpu().numpy(),
        k1 =k1,
        k2=k2,
        lambda_value=lambda_value, no_rerank=no_rerank
    )

    return euclidean_dist, rerank_dist

def calDis(qFeature, gFeature):#246s
    x, y = F.normalize(qFeature), F.normalize(gFeature)
    # x, y = qFeature, gFeature
    m, n = x.shape[0], y.shape[0]
    disMat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    disMat.addmm_(1, -2, x, y.t())
    return disMat.clamp_(min=1e-5)

def splitLowconfi(feature, labels, centers, ratio=0.2):
    # set bot 20% imsimilar samples to -1
    # center VS feature
    centerDis = calDis(torch.from_numpy(feature), torch.from_numpy(centers)).numpy() # center VS samples
    noiseLoc = []
    for ii, pid in enumerate(set(labels)):
        curDis = centerDis[:,ii]
        curDis[labels!=pid] = 100
        smallLossIdx = curDis.argsort()
        smallLossIdx = smallLossIdx[curDis[smallLossIdx]!=100]
        # bot 20% removed
        partSize = int(ratio*smallLossIdx.shape[0])
        if partSize!=0:
            noiseLoc.extend(smallLossIdx[-partSize:])
    labels[noiseLoc] = -1
    return labels

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices


    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters>0) else None
    print("==> Load source-domain dataset")
    dataset_source = get_data(args.dataset_source, args.data_dir)
    print("==> Load target-domain dataset")
    dataset_target = get_data(args.dataset_target, args.data_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)
    train_loader_source = get_train_loader(args, dataset_source, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters)
    source_classes = dataset_source.num_train_pids

    # Create model
    model = create_model(args)

    # Create hybrid memory
    memory = HybridMemory(model.module.num_features, source_classes+len(dataset_target.train),
                            temp=args.temp, momentum=args.momentum).cuda()

    # Initialize source-domain class centroids
    print("==> Initialize source-domain class centroids in the hybrid memory")
    sour_cluster_loader = get_test_loader(dataset_source, args.height, args.width,
                                    args.batch_size, args.workers, testset=sorted(dataset_source.train))
    source_features, _ = extract_features(model, sour_cluster_loader, print_freq=50)
    sour_fea_dict = collections.defaultdict(list)
    for f, pid, _ in sorted(dataset_source.train):
        sour_fea_dict[pid].append(source_features[f].unsqueeze(0))
    source_centers = [torch.cat(sour_fea_dict[pid],0).mean(0) for pid in sorted(sour_fea_dict.keys())]
    source_centers = torch.stack(source_centers,0)
    source_centers = F.normalize(source_centers, dim=1)

    # Initialize target-domain instance features
    print("==> Initialize target-domain instance features in the hybrid memory")
    tgt_cluster_loader = get_test_loader(dataset_target, args.height, args.width,
                                    args.batch_size, args.workers, testset=sorted(dataset_target.train))
    target_features, _ = extract_features(model, tgt_cluster_loader, print_freq=50)
    target_features = torch.cat([target_features[f].unsqueeze(0) for f, _, _ in sorted(dataset_target.train)], 0)
    source_features = torch.cat([source_features[f].unsqueeze(0) for f, _, _ in sorted(dataset_source.train)], 0)
    memory.features = torch.cat((source_centers, F.normalize(target_features, dim=1)), dim=0).cuda()
    del tgt_cluster_loader, source_centers, target_features, sour_cluster_loader, sour_fea_dict

    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = SpCLTrainer_UDA(model, memory, source_classes)

    for epoch in range(args.epochs):
        # Calculate distance
        print('==> Create pseudo labels for unlabeled target domain with self-paced policy')
        target_features = memory.features[source_classes:].clone()
        euclidean_dist, rerank_dist = compute_dist(target_features, k1=args.k1, k2=args.k2,lambda_value=args.lambda_value,no_rerank=args.no_rerank)
        # print('euclidean_dist',euclidean_dist)

        # del target_features
        target_features = target_features.cpu().numpy()

        eps = args.eps
        print('eps in cluster: {:.3f}'.format(eps))
        # cluster = DBSCAN(eps=eps, min_samples=args.min_samples+1, metric='precomputed', n_jobs=-1)
        # sample_number = len(target_features)
        # n_clusters = sample_number - int(sample_number * 0.018 * (epoch + 1))-600
        cluster = AgglomerativeClustering(n_clusters= None,affinity = 'precomputed',linkage='average',distance_threshold=eps)
        print('cluster', cluster)
        # select & cluster images as training set of this epochs
        print('Clustering and labeling...')
        pseudo_labels = cluster.fit_predict(rerank_dist)
        num_ids = len(set(pseudo_labels))  ##for DBSCAN cluster
        # num_ids = len(set(labels)) ##for affinity_propagation cluster
        # print('list(pseudo_labels)',list(pseudo_labels))
        print('Iteration {} have {} training ids'.format(epoch + 1, num_ids))
        imagelist = []
        for i in range(len(pseudo_labels)):
            if pseudo_labels[i] == -1:
                continue
            imagelist.append(pseudo_labels[i])
        print('Iteration {} have {} training images'.format(epoch + 1, len(imagelist)))
        pseudo_labels_ori = pseudo_labels

        # generate new dataset and calculate cluster centers
        def generate_pseudo_labels(cluster_id, num):
            labels = []
            for i, ((fname, _, cid), id) in enumerate(zip(sorted(dataset_target.train), cluster_id)):
                if id!=-1:
                    labels.append(source_classes+id)
                else:
                    print('smx')
            return torch.Tensor(labels).long()
        # pseudo_labels 无-1
        pseudo_labels = generate_pseudo_labels(pseudo_labels, num_ids)
        # print('list(pseudo_labels)',list(pseudo_labels))
        num_ids3 = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
        print('Iteration {} have {} training ids'.format(epoch + 1, num_ids3))
        imagelist3= []
        for i in range(len(pseudo_labels)):
            if pseudo_labels[i] == -1:
                continue
            imagelist3.append(pseudo_labels[i])
        print('Iteration {} have {} training images'.format(epoch + 1, len(imagelist3)))
        pseudo_labeled_dataset = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_target.train), pseudo_labels)):
            if pseudo_labels_ori[i]!=-1:
                pseudo_labeled_dataset.append((fname,label.item(),cid))
            else:
                print('zkw')
        # statistics of clusters and un-clustered instances
        index2label = collections.defaultdict(int)
        for label in pseudo_labels:
            index2label[label.item()]+=1
        index2label = np.fromiter(index2label.values(), dtype=float)
        print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances'
                    .format(epoch, (index2label>1).sum(), (index2label==1).sum()))
        memory.labels = torch.cat((torch.arange(source_classes), pseudo_labels)).cuda()
        train_loader_target = get_train_loader(args, dataset_target, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset)

        train_loader_source.new_epoch()
        train_loader_target.new_epoch()

        trainer.train(epoch, train_loader_source, train_loader_target, optimizer,
                    print_freq=args.print_freq, train_iters=len(train_loader_target))

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            mAP = evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=False)
            is_best = (mAP>best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    print ('==> Test with the best model on the target domain:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on UDA re-ID")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--lambda_value', type=float, default=0.1,
                        help="balancing parameter, default: 0.1")
    parser.add_argument('--numTarID', type=float, default=600,
                        help="kmeans cluster numbers")
    parser.add_argument('--no-rerank', action='store_true', help="train without rerank")
    parser.add_argument('--gpu-devices', default='0,1', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--rho', type=float, default=1.6e-3,
                        help="rho percentage, default: 1.6e-3")
    parser.add_argument('--min_samples', type=int, default=4,
                        help="min_samples, default: 4")

    main()
