#!/usr/bin/env python
# coding: utf-8

import os
import time
# import math
import torch
# import pickle
import argparse
import random
import copy
import json
import numpy as np
import os.path as osp
import scipy.sparse as sp
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm

from layers import *
from models import *
from preprocessing import *
from torch_geometric.data import Data

from convert_datasets_to_pygDataset import dataset_Hypergraph
import sys
sys.path.append('../../')
from mla_utils import *
from modelzoo import SimpleHGNN
import pandas as pd
from scipy.stats import friedmanchisquare
from torch.utils.data import DataLoader, TensorDataset


def meta_laplacian_FGSM(H, X, y, data, HG, surrogate_class, target_model, train_mask, val_mask, test_mask, logits_orig, budget=20, epsilon=0.05, T=20, eta_H=1e-2, eta_X=1e-2, alpha=4.0, reinit_if_stuck=True):
    """
      Meta Laplacian Attack adapted to poisoning setting (training-time). 
    - The attacker perturbs H and X before training.
    - A new model is trained from scratch at every iteration to simulte a bilevel optimization. 
    Params:
        H,X,y: Original incidence matrix, features and labels
        surrogate_class: Constructor of the surrogate model (e.g. SimpleHGNN)
    """
    verbose = False
    eta_H = args.ptb_rate 
    eta_X = args.epsilon
    device = X.device
    # idx_unlabeled = val_mask | test_mask
    H = H.clone().detach()
    X = X.clone().detach()
    H.requires_grad = False
    X.requires_grad = False
    n, m = H.shape
    # Z = model(H, X).detach()
    L_orig = lap(H)
    dv_orig = H @ torch.ones((H.shape[1],), device=device)
    loss_meta_trajectory = []
    acc_drop_trajectory = []
    lap_shift_trajectory = []
    lap_dist_trajectory = []
    cls_loss_trajectory = []
    deg_penalty_trajectory = []
    feature_shift_trajectory = []
    surrogate_test_trajectory = []
    target_test_trajectory = []
    # Surrogate model training
    if surrogate_class is None:
        surrogate_model = SimpleHGNN(X.shape[1], hidden_dim = args.MLP_hidden, out_dim = args.num_classes, device = X.device).to(device)
        optimizer = torch.optim.Adam(surrogate_model.parameters(),lr=args.lr)
    else:
        # surrogate_model = surrogate_class(X.shape[1],)
        raise Exception("Other surrogates Not implemented")
    criterion = nn.CrossEntropyLoss()
    best_val_accuracy = -float('inf')
    best_model_state = None
    surrogate_epochs = args.num_epochs_sur
    for epoch in range(surrogate_epochs):
        surrogate_model.train()
        optimizer.zero_grad()
        logits = surrogate_model(X, H)
        # print(logits[train_mask].shape,y[train_mask].shape,logits.shape,y.shape)
        loss = criterion(logits[train_mask],y[train_mask])
        loss.backward(retain_graph=True)
        optimizer.step()
        surrogate_model.eval()
        # Save the surrogate model (which has the best validation accuracy) for robust training
        with torch.no_grad():   
            # val_loss = criterion(logits[val_mask], y[val_mask])
            val_accuracy = accuracy(logits[val_mask],y[val_mask])
        if val_accuracy.item() > best_val_accuracy:
            best_val_accuracy = val_accuracy.item()
            # print('Best val accuracy: ',best_val_accuracy)
            best_model_state = surrogate_model.state_dict()

        if epoch%20 == 0 and verbose:
            print('Epoch: ',epoch)
            print("Surr Loss : ",loss.item())
    surrogate_model.load_state_dict(best_model_state) # Take the best model
    with torch.no_grad():
        surrogate_model.eval()
        Z_orig = surrogate_model(X, H) # Trained surrogate model
    lap_dist = torch.tensor(0.0).to(device)
    degree_penalty = torch.tensor(0.0).to(device)
    loss_cls = torch.tensor(0.0).to(device)
    for t in tqdm(range(T)):
        runtime_start = time.time()
        delta_H = (torch.randn_like(H)).requires_grad_()
        delta_X = (torch.randn_like(X)).requires_grad_()
        H_pert = torch.clamp(H + delta_H, 0, 1)
        # H_pert = (H_pert>0.5).float()
        X_pert = X + delta_X
        L_pert = lap(H_pert)
        Z_pert = surrogate_model(X_pert, H_pert)
        delta_L = (L_pert@Z_pert - L_orig @ Z_orig)
        # lap_dist += torch.mean((delta_L**2))/T
        lap_dist += torch.norm(delta_L, p=2).mean()/T

        dv_temp = H_pert @ torch.ones((H.shape[1],), device=device)
        degree_violation = (dv_temp - dv_orig)
        degree_penalty += (torch.sum(degree_violation ** 2) / n)/T
        # degree_penalty = torch.abs(degree_violation).mean()
        logits_adv = Z_pert
        loss_cls += (F.cross_entropy(logits_adv, y))/T
        time1 = time.time() - runtime_start

        # with torch.no_grad():
        #     target_model.eval()
        #     surrogate_model.eval()
        #     surrogate_test_accuracy = accuracy(surrogate_model(X_pert, H_pert)[test_mask], y[test_mask]) 
        #     target_model_test_accuracy = accuracy(target_model(X_pert, H_pert)[test_mask], y[test_mask])
        #     surrogate_test_trajectory.append(surrogate_test_accuracy.item())
        #     target_test_trajectory.append(target_model_test_accuracy.item())
        # if t == T-1:
        #     os.makedirs(os.path.join(root,str(args.seed)), exist_ok=True)
        #     prefix = os.path.join(root,str(args.seed), 'SimpleHGNN_'+args.dataset+'_'+args.model+'_'+str(args.ptb_rate))
        #     torch.save(best_model_state, prefix+'_weights.pth')
        runtime_start2 = time.time()

        deg_penalty_val = degree_penalty.item()
        cls_loss_val = loss_cls.item()
        lap_dist_val = lap_dist.item() if isinstance(lap_dist, torch.Tensor) else lap_dist
        loss_meta = lap_dist - degree_penalty + alpha * loss_cls

        grads = torch.autograd.grad(loss_meta,[delta_H,delta_X])

        lap_dist_trajectory.append(lap_dist_val)
        loss_meta_trajectory.append(loss_meta.item())
        # _, _, acc_drop = classification_drop(args,target_model, H, HG, X, H_pert, X_pert, y)
        with torch.no_grad():
            target_model.eval()
            # _, _, acc_drop = classification_drop(args,target_model, H, HG, X, H_pert, X_pert, y)
            # test_flag = False
            # if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
            #     data_input = [data.clone(), test_flag]
            # else:
            data_input = data.clone().to(device)
            # data_input.x = X_pert
            data_input.x = X_pert
            # print("H_pert.shape:", H_pert.shape)
            # print("data.num_hyperedges:", data_input.num_hyperedges)
            edge_index = incidence_to_edge_index2(H_pert)
            # data_input.edge_index = H_pert
            # edge_index = generate_G_from_H(data)
            # print("edge_index:", edge_index)
            # print("edge_index.shape:", edge_index.shape)
            data_input.edge_index = edge_index
            data_input.n_x = X_pert.shape[0]

            
            data_input = ExtractV2E(data_input)
            # print('after extract v2e:',data_input.edge_index.shape)
            data_input = Add_Self_Loops(data_input)
            # print('after add self loops:',data_input.edge_index.shape)
            # data_input.num_hyperedges = data_input.edge_index[0].max()-data_input.n_x+1
            data_input.edge_index[1] -= data_input.edge_index[1].min()
            data_input.edge_index = data_input.edge_index.to(device)
            data_input.x = data_input.x.to(device)
            if args.method in ['AllSetTransformer', 'AllDeepSets']:
                data_input = norm_contruction(data_input, option=args.normtype)
            # print("max edge_index[1]:", data_input.edge_index[1].max().item())
            # data_input = ExtractV2E(data_input)
            test_flag = True
            if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
                data_input = [data_input, test_flag]
            # print('data_input',data_input)
            # print('data: ',data)
            # assert data_input.edge_index.shape[0] == 2
            # assert data_input.edge_index[0].max() < data_input.x.shape[0], "Invalid node index"
            # assert data_input.edge_index[1].max() < data.num_hyperedges, "Invalid hyperedge index"
            logits_adv = target_model(data_input)
        acc_orig = (logits_orig.argmax(dim=1)[test_mask] == y[test_mask]).float().mean().item()
        acc_adv = (logits_adv.argmax(dim=1)[test_mask] == y[test_mask]).float().mean().item()
        acc_drop = (acc_orig - acc_adv)/acc_orig
        acc_drop_trajectory.append(acc_drop)
        cls_loss_trajectory.append(cls_loss_val)
        deg_penalty_trajectory.append(deg_penalty_val)
        lap_diff = laplacian_diff(H, torch.clamp(H + delta_H, 0, 1))
        feature_shift = torch.norm(delta_X, p=2).item()
        lap_shift_trajectory.append(lap_diff)
        feature_shift_trajectory.append(feature_shift)

        # Proceed with original gradient ascent
        delta_H = eta_H * grads[0].sign()
        delta_X = eta_X * grads[1].sign()
        flat = delta_H.abs().flatten()
        topk = torch.topk(flat, k=min(delta_H.numel(), budget)).indices
        delta_H_new = torch.zeros_like(delta_H)
        delta_H_new.view(-1)[topk] = delta_H.view(-1)[topk]
        # In the following code segment we do not update bad nodes (nodes whose deg <= 0 ) or bad edges (whose card <= 1)
        H_temp = torch.clamp(H + delta_H_new, 0, 1)
        row_degrees = H_temp.sum(dim=1)
        col_degrees = H_temp.sum(dim=0)
        bad_nodes = (row_degrees < 1).nonzero(as_tuple=True)[0]
        if args.dataset == '20newsW100':
            bad_edges = (col_degrees < 1).nonzero(as_tuple=True)[0]
        else:
            bad_edges = (col_degrees < 2).nonzero(as_tuple=True)[0]
        # print('|bad nodes| = ',len(bad_nodes))
        # print('|bad edges| = ',len(bad_edges))
        # print('deg: ',row_degrees.mean(), row_degrees.min(),row_degrees.max())
        # print('dim: ',col_degrees.mean(), col_degrees.min(),col_degrees.max())
        delta_H_new[bad_nodes, :] = 0
        delta_H_new[:, bad_edges] = 0
        delta_H.copy_(delta_H_new)

        delta_X = delta_X.clamp(-epsilon, epsilon)
        time2 = time.time() - runtime_start2
    # results = [(t, loss_meta, acc_drop, lap_shift, deg_penalty, cls_loss, lap_dist, feature_shift)]
    # print(lap_shift_trajectory)
    results = [loss_meta_trajectory, acc_drop_trajectory, lap_shift_trajectory, lap_dist_trajectory, cls_loss_trajectory, \
               deg_penalty_trajectory,feature_shift_trajectory,surrogate_test_trajectory, target_test_trajectory]
    # mask = filter_potential_singletons(torch.clamp(H + delta_H, 0, 1))
    return torch.clamp(H + delta_H, 0, 1), X + delta_X, results, time1+time2, best_model_state

# def meta_laplacian_pois_attack(root, H, X, y, data, HG, surrogate_class, target_model, train_mask, val_mask, test_mask, logits_orig, budget=20, epsilon=0.05, T=20, eta_H=1e-2, eta_X=1e-2, alpha=4.0, reinit_if_stuck=True):
#     """
#       Meta Laplacian Attack adapted to poisoning setting (training-time). 
#     - The attacker perturbs H and X before training.
#     - A new model is trained from scratch at every iteration to simulte a bilevel optimization. 
#     Params:
#         H,X,y: Original incidence matrix, features and labels
#         surrogate_class: Constructor of the surrogate model (e.g. SimpleHGNN)
#     """
#     verbose = False
#     device = X.device
#     # idx_unlabeled = val_mask | test_mask
#     H = H.clone().detach()
#     X = X.clone().detach()
#     H.requires_grad = False
#     X.requires_grad = False
#     n, m = H.shape
#     # Z = model(H, X).detach()
#     delta_H = (1e-3 * torch.randn_like(H)).requires_grad_()
#     delta_X = (1e-3 * torch.randn_like(X)).requires_grad_()
#     L_orig = lap(H)
#     dv_orig = H @ torch.ones((H.shape[1],), device=device)
#     loss_meta_trajectory = []
#     acc_drop_trajectory = []
#     lap_shift_trajectory = []
#     lap_dist_trajectory = []
#     cls_loss_trajectory = []
#     deg_penalty_trajectory = []
#     feature_shift_trajectory = []
#     surrogate_test_trajectory = []
#     target_test_trajectory = []
#     for t in tqdm(range(T)):
#         runtime_start = time.time()
#         if surrogate_class is None:
#             surrogate_model = SimpleHGNN(X.shape[1], hidden_dim = args.MLP_hidden, out_dim = args.num_classes, device = X.device).to(device)
#             optimizer = torch.optim.Adam(surrogate_model.parameters(),lr=args.lr)
#         else:
#             # surrogate_model = surrogate_class(X.shape[1],)
#             raise Exception("Other surrogates Not implemented")
#         H_pert = torch.clamp(H + delta_H, 0, 1)
#         X_pert = X + delta_X
#         L_pert = lap(H_pert)
#         # for epoch in tqdm(range(args.num_epochs),desc = 'Training surrogate: iter = '+str(t)):
#         criterion = nn.CrossEntropyLoss()
#         best_val_accuracy = -float('inf')
#         best_model_state = None
#         if args.dataset == '20newsW100':
#             surrogate_epochs = args.num_epochs_sur
#         else:
#             if t == T-1:
#                 surrogate_epochs = args.epochs 
#             else:
#                 surrogate_epochs = args.num_epochs_sur
#         data.x = X_pert
#         data.edge_index = incidence_to_edge_index(H_pert)
#         for epoch in range(surrogate_epochs):
#             surrogate_model.train()
#             optimizer.zero_grad()
#             logits = surrogate_model(X_pert, H_pert)
#             # print(logits[train_mask].shape,y[train_mask].shape,logits.shape,y.shape)
#             loss = criterion(logits[train_mask],y[train_mask])
#             loss.backward(retain_graph=True)
#             optimizer.step()
#             if t == T-1:
#                 surrogate_model.eval()
#                 # Save the surrogate model (which has the best validation accuracy) for robust training
#                 with torch.no_grad():   
#                     # val_loss = criterion(logits[val_mask], y[val_mask])
#                     val_accuracy = accuracy(logits[val_mask],y[val_mask])
#                 if val_accuracy.item() > best_val_accuracy:
#                     best_val_accuracy = val_accuracy.item()
#                     # print('Best val accuracy: ',best_val_accuracy)
#                     best_model_state = surrogate_model.state_dict()

#             if epoch%20 == 0 and verbose:
#                 print('Epoch: ',epoch)
#                 with torch.no_grad():
#                     target_model.eval()
#                     # --------
#                     data_input = data.clone().to(device)
#                     data_input.x = X_pert
#                     edge_index = incidence_to_edge_index2(H_pert)
#                     data_input.edge_index = edge_index
#                     data_input.n_x = X_pert.shape[0]
#                     data_input = ExtractV2E(data_input)
#                     data_input = Add_Self_Loops(data_input)
#                     # _, _, acc_drop = classification_drop(args,target_model, H, HG, X, H_pert, X_pert, y)
#                     data_input.edge_index[1] -= data_input.edge_index[1].min()
#                     data_input.edge_index = data_input.edge_index.to(device)
#                     if args.method in ['AllSetTransformer', 'AllDeepSets']:
#                         data_input = norm_contruction(data_input, option=args.normtype)

#                     test_flag = True
#                     if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
#                         data_input = [data_input, test_flag]
#                     # ------
#                     logits_adv = target_model(data_input)
#                 acc_orig = (logits_orig.argmax(dim=1)[test_mask] == y[test_mask]).float().mean().item()
#                 acc_adv = (logits_adv.argmax(dim=1)[test_mask] == y[test_mask]).float().mean().item()
#                 acc_drop = (acc_orig - acc_adv)/acc_orig
#                 _, _, acc_drop_sur = classification_drop(args,surrogate_model, H, None, X, H_pert, X_pert, y)
#                 print("Surr Loss : ",loss.item()," Accuracy drop (surrogate): ", acc_drop_sur*100,'%', " Accuracy drop (target): ", acc_drop*100,'%')
#         time1 = time.time() - runtime_start
#         if t == T-1:
#             surrogate_model.load_state_dict(best_model_state) # Take the best model
#         with torch.no_grad():
#             target_model.eval()
#             surrogate_model.eval()
#             surrogate_test_accuracy = accuracy(surrogate_model(X_pert,H_pert)[test_mask], y[test_mask]) 
#             # ---
#             data_input = data.clone().to(device)
#             data_input.x = X_pert
#             edge_index = incidence_to_edge_index2(H_pert)
#             data_input.edge_index = edge_index
#             data_input.n_x = X_pert.shape[0]
#             data_input = ExtractV2E(data_input)
#             data_input = Add_Self_Loops(data_input)
#             # _, _, acc_drop = classification_drop(args,target_model, H, HG, X, H_pert, X_pert, y)
#             data_input.edge_index[1] -= data_input.edge_index[1].min()
#             data_input.edge_index = data_input.edge_index.to(device)
#             if args.method in ['AllSetTransformer', 'AllDeepSets']:
#                 data_input = norm_contruction(data_input, option=args.normtype)

#             test_flag = True
#             if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
#                 data_input = [data_input, test_flag]
#             target_Z = target_model(data_input)
#             # ---
#             target_model_test_accuracy = accuracy(target_Z[test_mask], y[test_mask])
#             surrogate_test_trajectory.append(surrogate_test_accuracy.item())
#             target_test_trajectory.append(target_model_test_accuracy.item())
#         if t == T-1:
#             os.makedirs(os.path.join(root,str(args.seed)), exist_ok=True)
#             prefix = os.path.join(root,str(args.seed), 'SimpleHGNN_'+args.dataset+'_'+args.model+'_'+str(args.ptb_rate))
#             torch.save(best_model_state, prefix+'_weights.pth')
#         runtime_start2 = time.time()
#         Z = surrogate_model(X_pert, H_pert) # Trained surrogate model
#         delta_L = (L_pert - L_orig) @ Z
#         # loss_meta = (delta_L**2).sum()
#         H_temp = torch.clamp(H + delta_H, 0, 1)
#         dv_temp = H_temp @ torch.ones((H.shape[1],), device=device)
#         degree_violation = (dv_temp - dv_orig)
#         degree_penalty = torch.sum(degree_violation ** 2) / n
#         # degree_penalty = torch.abs(degree_violation).mean()
#         deg_penalty_val = degree_penalty.item()
#         # loss_meta += degree_penalty

#         # logits_adv = target_model(X_pert,H_pert)
#         logits_adv = Z
#         loss_cls = F.cross_entropy(logits_adv, y)
#         # loss_cls = F.cross_entropy(logits_adv, model(H, X).argmax(dim=1))
#         # lap_dist = (delta_L**2).sum()
#         lap_dist = torch.norm(delta_L, p=2).mean()
#         # print(delta_L.shape)
#         # lap_dist = torch.mean(delta_L**2)
#         cls_loss_val = loss_cls.item()
#         lap_dist_val = lap_dist.item() if isinstance(lap_dist, torch.Tensor) else lap_dist
#         loss_meta = lap_dist + degree_penalty + alpha * loss_cls

#         grads = torch.autograd.grad(loss_meta,[delta_H,delta_X])

#         lap_dist_trajectory.append(lap_dist_val)
#         loss_meta_trajectory.append(loss_meta.item())
#         # _, _, acc_drop = classification_drop(args,target_model, H, HG, X, H_pert, X_pert, y)
#         with torch.no_grad():
#             target_model.eval()
#             # _, _, acc_drop = classification_drop(args,target_model, H, HG, X, H_pert, X_pert, y)
#             # -------
#             data_input = data.clone().to(device)
#             data_input.x = X_pert
#             edge_index = incidence_to_edge_index2(H_pert)
#             data_input.edge_index = edge_index
#             data_input.n_x = X_pert.shape[0]
#             data_input = ExtractV2E(data_input)
#             data_input = Add_Self_Loops(data_input)
#             # _, _, acc_drop = classification_drop(args,target_model, H, HG, X, H_pert, X_pert, y)
#             data_input.edge_index[1] -= data_input.edge_index[1].min()
#             data_input.edge_index = data_input.edge_index.to(device)
#             if args.method in ['AllSetTransformer', 'AllDeepSets']:
#                 data_input = norm_contruction(data_input, option=args.normtype)
#             test_flag = True
#             if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
#                 data_input = [data_input, test_flag]
#             # ------
#             logits_adv = target_model(data_input)
#         acc_orig = (logits_orig.argmax(dim=1)[test_mask] == y[test_mask]).float().mean().item()
#         acc_adv = (logits_adv.argmax(dim=1)[test_mask] == y[test_mask]).float().mean().item()
#         acc_drop = (acc_orig - acc_adv)/acc_orig
#         acc_drop_trajectory.append(acc_drop)
#         cls_loss_trajectory.append(cls_loss_val)
#         deg_penalty_trajectory.append(deg_penalty_val)
#         lap_diff = laplacian_diff(H, torch.clamp(H + delta_H, 0, 1))
#         feature_shift = torch.norm(delta_X, p=2).item()
#         lap_shift_trajectory.append(lap_diff)
#         feature_shift_trajectory.append(feature_shift)
#         with torch.no_grad():
#             # Proceed with original gradient ascent
#             delta_H += eta_H * grads[0].sign()
#             delta_X += eta_X * grads[1].sign()
#             flat = delta_H.abs().flatten()
#             topk = torch.topk(flat, k=min(delta_H.numel(), budget)).indices
#             delta_H_new = torch.zeros_like(delta_H)
#             delta_H_new.view(-1)[topk] = delta_H.view(-1)[topk]
#             # In the following code segment we do not update bad nodes (nodes whose deg <= 0 ) or bad edges (whose card <= 1)
#             H_temp = torch.clamp(H + delta_H_new, 0, 1)
#             row_degrees = H_temp.sum(dim=1)
#             col_degrees = H_temp.sum(dim=0)
#             bad_nodes = (row_degrees < 1).nonzero(as_tuple=True)[0]
#             if args.dataset == '20newsW100':
#                 bad_edges = (col_degrees < 1).nonzero(as_tuple=True)[0]
#             else:
#                 bad_edges = (col_degrees < 2).nonzero(as_tuple=True)[0]
#             # print('|bad nodes| = ',len(bad_nodes))
#             # print('|bad edges| = ',len(bad_edges))
#             delta_H_new[bad_nodes, :] = 0
#             delta_H_new[:, bad_edges] = 0
#             delta_H.copy_(delta_H_new)

#         delta_X = delta_X.clamp(-epsilon, epsilon)
#         time2 = time.time() - runtime_start2
#     # results = [(t, loss_meta, acc_drop, lap_shift, deg_penalty, cls_loss, lap_dist, feature_shift)]
#     results = [loss_meta_trajectory, acc_drop_trajectory, lap_shift_trajectory, lap_dist_trajectory, cls_loss_trajectory, \
#                deg_penalty_trajectory,feature_shift_trajectory,surrogate_test_trajectory, target_test_trajectory]
#     # mask = filter_potential_singletons(torch.clamp(H + delta_H, 0, 1))
#     return torch.clamp(H + delta_H, 0, 1), X + delta_X, results, time1+time2, best_model_state

# def create_local_H(batch_data):
#     """
#     Create a local incidence matrix `local_H` for the current mini-batch.
#     Args:
#         batch_data: The current batch data (containing `edge_index`).

#     Returns:
#         local_H: The incidence matrix for the current batch.
#     """
#     print('create local H: ',batch_data.edge_index)
#     print('batch_data.node_idx_orig: ',batch_data.node_idx_orig)
#     device = batch_data.x.device  # Get the device from the batch data
#     # Initialize the local incidence matrix with zeros
#     local_H = torch.zeros((batch_data.x.shape[0], batch_data.num_hyperedges), device=device)

#     # Extract node and hyperedge indices from the batch's edge_index
#     src, dst = batch_data.edge_index

#     # Determine the node-to-hyperedge connections (similar to the global H construction)
#     node_mask = src < batch_data.x.shape[0]  # Only consider node-to-hyperedge edges
#     node_indices = src[node_mask]
#     # hyperedge_indices = (dst[node_mask] - batch_data.n_x).clip(0, batch_data.num_hyperedges - 1)  # Shift back for hyperedges
#     hyperedge_indices = (dst[node_mask] - batch_data.x.shape[0])
#     print('node_indices: ',node_indices)
#     print('hyperedge_indices:',hyperedge_indices)
#     assert hyperedge_indices.max() < batch_data.num_hyperedges, "Hyperedge index out of bounds"
#     # Set the corresponding entries in the local incidence matrix to 1
#     local_H[node_indices, hyperedge_indices] = 1.0

#     return local_H

def flip_sparse_incidence(H: torch.Tensor, flip_idx: torch.Tensor) -> torch.Tensor:
    """
    H: sparse COO [n, m], values assumed 1s, coalesced
    flip_idx: LongTensor [k, 2] with (row, col) pairs to toggle
    returns: sparse COO H_adv with those entries flipped
    """
    assert H.is_sparse, "H must be a sparse COO tensor"
    H = H.coalesce()
    device = H.device
    n, m = H.shape

    if flip_idx.numel() == 0:
        return H

    flip_idx = flip_idx.to(device=device, dtype=torch.long)

    # Linearize indices for fast membership tests
    H_idx = H.indices()                      # [2, nnz]
    H_lin = H_idx[0] * m + H_idx[1]         # [nnz]
    flip_lin = flip_idx[:, 0] * m + flip_idx[:, 1]  # [k]

    # torch.isin is available in newer torch; if not, see fallback below.
    in_H = torch.isin(flip_lin, H_lin)      # [k] True if currently 1 -> should remove

    # 1) Remove: keep entries not in remove set
    remove_lin = flip_lin[in_H]
    if remove_lin.numel() > 0:
        keep_mask = ~torch.isin(H_lin, remove_lin)
        kept_idx = H_idx[:, keep_mask]
    else:
        kept_idx = H_idx

    # 2) Add: entries not in H
    add_idx = flip_idx[~in_H]               # [k_add, 2]
    if add_idx.numel() > 0:
        new_idx = torch.cat([kept_idx, add_idx.t()], dim=1)
    else:
        new_idx = kept_idx

    # Rebuild sparse tensor (all ones)
    values = torch.ones(new_idx.size(1), device=device, dtype=H.dtype)
    H_adv = torch.sparse_coo_tensor(new_idx, values, size=(n, m), device=device).coalesce()
    return H_adv

def meta_laplacian_PGD(
    args, data, split_idx,
    train_mask,
    budget=20, epsilon=0.05, T=20,
    eta_H=1e-2, eta_X=None
):
    """
    PGD-style MeLA attack (no meta-learning unrolling).

    Stronger than FGSM by multi-step ascent on a fixed meta-loss.
    """
    batch_size=args.batch_size
    alpha, beta, gamma = args.alpha, args.beta, args.gamma
    device = data.x.device
    data = data.to(device)
    
    if eta_X is None:
        eta_X = epsilon / T

    # Construct full incidence matrix (sparse)
    full_num_nodes = data.x.size(0)
    full_num_hyperedges = data.num_hyperedges
    full_indices = data.edge_index
    full_values = torch.ones(full_indices.size(1), device=device)
    H = torch.sparse_coo_tensor(
        full_indices, full_values, size=(full_num_nodes, full_num_hyperedges), device=device
    ).coalesce()
    global_delta_H = torch.zeros(H.shape).to(device)
    global_delta_X = torch.zeros_like(data.x).to(device)

    data_loader = DataLoader(TensorDataset(data.node_idx, data.x, data.y), batch_size=batch_size, shuffle=True)
    # -------------------------------------------------
    # 1) Train surrogate ONCE (clean)
    # --------------------------------------------------
    surrogate = SimpleHGNN(
        data.x.shape[1],
        hidden_dim=args.MLP_hidden,
        out_dim=args.num_classes,
        device=device
    ).to(device)

    opt = torch.optim.Adam(surrogate.parameters(), lr=args.lr)
    for batch in data_loader:
        batch_data, local_H = create_minibatch_for_hcha(data, batch, split_idx,construct_H=True)  # Create mini-batch for this batch
        batch_data = batch_data.to(device)
        for _ in range(args.num_epochs_sur):
            opt.zero_grad()
            loss = F.cross_entropy(surrogate(batch_data.x, local_H)[batch_data.train_mask], batch_data.y[batch_data.train_mask])
            loss.backward()
            opt.step()

    surrogate.eval()
    # with torch.no_grad():
    #     Z_orig = surrogate(X, H)

    # L_orig = lap(H)
    # dv_orig = H @ torch.ones((m,), device=device)

    # # --------------------------------------------------
    # # 2) Initialize perturbations
    # # --------------------------------------------------
    # delta_H = torch.zeros_like(H, requires_grad=True)
    # delta_X = torch.zeros_like(X, requires_grad=True)

    # --------------------------------------------------
    # 3) PGD loop
    # --------------------------------------------------
    # loss_metas = []
    # lap_dists = []
    # degree_penalties = []
    # loss_cls_list = []

    for t in tqdm(range(T)):
        for batch in data_loader:
            batch_data, local_H = create_minibatch_for_hcha(data, batch, split_idx,construct_H=True)  # Create mini-batch for this batch
            batch_data = batch_data.to(device)  # Move to the appropriate device

            # Construct the local incidence matrix for this batch
            # local_H = create_local_H(batch_data)
            L_orig = lap(local_H)
            dv_orig = local_H @ torch.ones((local_H.shape[1],), device=device)
            local_X = batch_data.x
            if t == 0:
                delta_X = torch.zeros_like(local_X, requires_grad=True)
                delta_H = torch.zeros_like(local_H, requires_grad=True)
            else:
                delta_X = global_delta_X.index_select(0, batch_data.node_idx_orig)
                delta_H = global_delta_H[batch_data.node_idx_orig,:][:,batch_data.involved_hyperedges-batch_data.involved_hyperedges.min()]

            H_pert = torch.clamp(local_H + delta_H, 0, 1)
            X_pert = local_X + delta_X
        
            with torch.no_grad():
                Z_orig = surrogate(local_X, local_H)
            L_pert = lap(H_pert)
            Z = surrogate(X_pert, H_pert)

            # Meta-loss
            delta_L = L_pert @ Z - L_orig @ Z_orig
            lap_dist = (
                torch.norm(delta_L, p=2)
                if args.loss == "L2"
                else (delta_L ** 2).mean()
            )

            dv_temp = H_pert @ torch.ones((local_H.shape[1],), device=device)
            deg_penalty = torch.mean((dv_temp - dv_orig) ** 2)
            cls_loss = F.cross_entropy(Z[batch_data.train_mask], batch_data.y[batch_data.train_mask])

            loss_meta = alpha * lap_dist - beta * deg_penalty + gamma * cls_loss
            # loss_metas.append(loss_meta.item())
            # lap_dists.append(lap_dist.item())
            # degree_penalties.append(deg_penalty.item())
            # loss_cls_list.append(cls_loss.item())
            gH, gX = torch.autograd.grad(loss_meta, [delta_H, delta_X])

            with torch.no_grad():
                # ---- H: gradient-based top-k PGD
                flip_gain = (1 - 2 * local_H) * gH
                score = flip_gain.clone()
                score[score <= 0] = -float("inf")

                flat = score.flatten()
                topk = torch.topk(flat, k=min(budget//(4*len(data_loader)), flat.numel())).indices
                # topk = torch.topk(flat, k=min(budget*(local_H.shape[0]*local_H.shape[1])//(H.shape[0]*H.shape[1]), flat.numel())).indices

                M = torch.zeros_like(flat)
                M[topk] = 1.0
                M = M.view_as(local_H)

                delta_H.copy_(eta_H * gH.sign())
                delta_H.mul_(M)

                # ---- X: standard PGD
                delta_X.add_(eta_X * gX.sign())
                delta_X.clamp_(-epsilon, epsilon)
            global_delta_X.index_copy_(0, batch_data.node_idx_orig, delta_X)
            row_indices = batch_data.node_idx_orig.flatten()
            col_indices = batch_data.involved_hyperedges-batch_data.involved_hyperedges.min()
            R, C = len(row_indices), len(col_indices)
            rows = row_indices.unsqueeze(1).expand(-1, C)      # shape [R, C]
            cols = col_indices.unsqueeze(0).expand(R, -1)      # shape [R, C]
            global_delta_H[rows,cols] = delta_H 

 
    # ------------------------------------------------- -
    # 4) Discretize
    # --------------------------------------------------
    with torch.no_grad():
        flip_idx = (global_delta_H != 0).nonzero(as_tuple=False)  # [k, 2]
        H_adv = flip_sparse_incidence(H, flip_idx)
        X_adv = data.x + global_delta_X #.clamp(-epsilon, epsilon)
    results = {}
    with torch.no_grad():
        # report effective budget usage
        num_changed, edges_changed, nodes_changed = sparse_change_stats(H, H_adv)
        print('#changed:', num_changed, 'budget:', budget)
        print('edges_changed:', edges_changed)
        print('nodes_changed:', nodes_changed)
        results['h_l0'] = int(num_changed)
        results['edges_changed'] = int(edges_changed)
        results['nodes_changed'] = int(nodes_changed)
        results['x_inf'] = torch.norm((data.x - X_adv), p=float('inf')).item()

    # results = {
    #     "num_changed_H": int(num_changed)    
    # }
    # print("loss trajectory: ", loss_metas)
    # print('degree penalty trajectory: ', degree_penalties)
    # print('laplacian distance trajectory: ', lap_dists)
    # print('classification loss trajectory: ', loss_cls_list)
    return H_adv, X_adv,results 

# def meta_laplacian_PGD(
#     args, data, split_idx, train_mask,
#     budget=20, epsilon=0.05, T=20,
#     eta_H=1e-2, eta_X=None,
#     cand_per_batch_cap=2048,
# ):
#     """
#     MeLA-PGD (minibatch-safe, sparse-global-H).

#     Key properties:
#       - H is maintained as sparse COO globally (H_cur).
#       - For each minibatch, local_H is dense (from create_minibatch_for_hcha(..., construct_H=True)).
#       - Feature perturbation is global dense delta_X (node-feature sized), updated by overwrite.
#       - Structure update uses global top-k flips per outer PGD step (budget).
#       - No singleton / small-hyperedge constraints (removed entirely).
#     """
#     device = data.x.device
#     data = data.to(device)

#     batch_size = args.batch_size
#     alpha, beta, gamma = args.alpha, args.beta, args.gamma

#     if eta_X is None:
#         eta_X = epsilon / max(1, T)

#     # -----------------------------
#     # Build initial sparse incidence H from data.edge_index
#     # Assumes data.edge_index entries are (node_id, hyperedge_id) in [0, num_hyperedges-1]
#     # -----------------------------
#     full_num_nodes = int(data.x.size(0))
#     full_num_hyperedges = int(data.num_hyperedges)

#     full_indices = data.edge_index.to(device)
#     full_values = torch.ones(full_indices.size(1), device=device, dtype=torch.float32)

#     H_init = torch.sparse_coo_tensor(
#         full_indices, full_values,
#         size=(full_num_nodes, full_num_hyperedges),
#         device=device
#     ).coalesce()

#     H_cur = H_init
#     global_delta_X = torch.zeros_like(data.x, device=device)

#     # -----------------------------
#     # Data loader
#     # -----------------------------
#     if not hasattr(data, "node_idx"):
#         data.node_idx = torch.arange(full_num_nodes, device=device)

#     data_loader = DataLoader(
#         TensorDataset(data.node_idx, data.x, data.y),
#         batch_size=batch_size,
#         shuffle=True
#     )

#     # -----------------------------
#     # Train surrogate ONCE on clean minibatches
#     # -----------------------------
#     surrogate = SimpleHGNN(
#         data.x.shape[1],
#         hidden_dim=args.MLP_hidden,
#         out_dim=args.num_classes,
#         device=device
#     ).to(device)

#     opt = torch.optim.Adam(surrogate.parameters(), lr=args.lr)
#     surrogate.train()
#     for batch in data_loader:
#         batch_data, local_H = create_minibatch_for_hcha(
#             data, batch, split_idx, construct_H=True
#         )
#         batch_data = batch_data.to(device)
#         local_H = local_H.to(device)
#         for _ in range(args.num_epochs_sur):
#             opt.zero_grad()
#             logits = surrogate(batch_data.x, local_H)
#             loss = F.cross_entropy(
#                 logits[batch_data.train_mask],
#                 batch_data.y[batch_data.train_mask]
#             )
#             loss.backward()
#             opt.step()
#     surrogate.eval()

#     # -----------------------------
#     # PGD outer loop
#     # -----------------------------
#     for t in tqdm(range(T), desc="MeLA-PGD (simplified)"):
#         # make minibatch builder see current structure
#         with torch.no_grad():
#             data.edge_index = H_cur.coalesce().indices()

#         cand_rows, cand_cols, cand_scores = [], [], []

#         for batch in data_loader:
#             batch_data, local_H = create_minibatch_for_hcha(
#                 data, batch, split_idx, construct_H=True
#             )
#             batch_data = batch_data.to(device)
#             local_H = local_H.to(device)

#             node_ids = batch_data.node_idx_orig.to(device)          # [B] global node ids
#             he_ids = batch_data.involved_hyperedges.to(device)      # [E_local] global hyperedge ids

#             # Leaf feature perturbation for this batch
#             delta_X_local = (
#                 global_delta_X.index_select(0, node_ids)
#                 .detach().clone().requires_grad_(True)
#             )

#             # Leaf relaxed H variable for this batch
#             H_relax = local_H.detach().clone().requires_grad_(True)

#             X_pert = batch_data.x + delta_X_local
#             H_pert = torch.clamp(H_relax, 0.0, 1.0)

#             with torch.no_grad():
#                 Z_orig = surrogate(batch_data.x, local_H)
#                 L_orig = lap(local_H)
#                 dv_orig = local_H @ torch.ones((local_H.shape[1],), device=device)

#             L_pert = lap(H_pert)
#             Z = surrogate(X_pert, H_pert)

#             delta_L = L_pert @ Z - L_orig @ Z_orig
#             lap_dist = (
#                 torch.norm(delta_L, p=2)
#                 if args.loss == "L2"
#                 else (delta_L ** 2).mean()
#             )

#             dv_temp = H_pert @ torch.ones((local_H.shape[1],), device=device)
#             deg_penalty = torch.mean((dv_temp - dv_orig) ** 2)

#             cls_loss = F.cross_entropy(
#                 Z[batch_data.train_mask],
#                 batch_data.y[batch_data.train_mask]
#             )

#             loss_meta = alpha * lap_dist - beta * deg_penalty + gamma * cls_loss

#             gH, gX = torch.autograd.grad(loss_meta, [H_relax, delta_X_local])

#             # ---- Feature PGD update: overwrite global entries (no accumulation)
#             with torch.no_grad():
#                 delta_X_upd = delta_X_local + eta_X * gX.sign()
#                 delta_X_upd.clamp_(-epsilon, epsilon)
#                 global_delta_X[node_ids] = delta_X_upd.detach()

#             # ---- Structure candidate collection (no singleton mask)
#             local_H_bin = (local_H > 0.5).float()
#             flip_gain = (1.0 - 2.0 * local_H_bin) * gH
#             score = flip_gain.detach()
#             score[score <= 0] = -float("inf")  # only beneficial flips

#             flat = score.flatten()
#             if torch.isfinite(flat).any():
#                 k_local = min(int(cand_per_batch_cap), flat.numel())
#                 top_idx = torch.topk(flat, k=k_local).indices

#                 valid = torch.isfinite(flat[top_idx])
#                 if valid.any():
#                     top_idx = top_idx[valid]

#                     mloc = score.shape[1]
#                     u_loc = top_idx // mloc
#                     e_loc = top_idx % mloc

#                     u_glob = node_ids[u_loc]
#                     e_glob = he_ids[e_loc]
#                     sc = flat[top_idx]

#                     cand_rows.append(u_glob)
#                     cand_cols.append(e_glob)
#                     cand_scores.append(sc)

#         # ---- GLOBAL top-k over all collected candidates
#         with torch.no_grad():
#             if len(cand_scores) == 0:
#                 continue

#             rows = torch.cat(cand_rows, dim=0)
#             cols = torch.cat(cand_cols, dim=0)
#             scores = torch.cat(cand_scores, dim=0)

#             # Deduplicate by (row,col) and keep max score per pair (version-safe)
#             m = H_cur.shape[1]
#             lin = rows * m + cols

#             order = torch.argsort(lin)
#             lin_s = lin[order]
#             rows_s = rows[order]
#             cols_s = cols[order]
#             scores_s = scores[order]

#             uniq_lin, counts = torch.unique_consecutive(lin_s, return_counts=True)
#             starts = torch.cat([torch.tensor([0], device=device), counts.cumsum(0)[:-1]])
#             ends = counts.cumsum(0)

#             max_rows, max_cols, max_scores = [], [], []
#             for a, b in zip(starts.tolist(), ends.tolist()):
#                 seg = scores_s[a:b]
#                 j = int(torch.argmax(seg).item())
#                 max_rows.append(rows_s[a + j])
#                 max_cols.append(cols_s[a + j])
#                 max_scores.append(seg[j])

#             max_rows = torch.stack(max_rows)
#             max_cols = torch.stack(max_cols)
#             max_scores = torch.stack(max_scores)

#             k = min(int(budget), int(max_scores.numel()))
#             if k <= 0:
#                 continue

#             top = torch.topk(max_scores, k=k).indices
#             flip_idx = torch.stack([max_rows[top], max_cols[top]], dim=1)  # [k,2]

#             H_cur = flip_sparse_incidence(H_cur, flip_idx)

#     # -----------------------------
#     # Output
#     # -----------------------------
#     with torch.no_grad():
#         H_adv = H_cur.coalesce()
#         X_adv = data.x + global_delta_X

#         results = {}
#         num_changed, edges_changed, nodes_changed = sparse_change_stats(H_init, H_adv)
#         results["h_l0"] = int(num_changed)
#         results["edges_changed"] = int(edges_changed)
#         results["nodes_changed"] = int(nodes_changed)

#     return H_adv, X_adv, results

def meta_laplacian_pois_attack(root, data, surrogate_class, target_model, split_idx, logits_orig, budget=20, epsilon=0.05, T=20, eta_H=1e-2, eta_X=1e-2, alpha=4.0):
    """
    Meta Laplacian Attack adapted to poisoning setting (training-time) with mini-batch support.
    Params:
        H, X, y: Original incidence matrix, features, and labels
        surrogate_class: Constructor of the surrogate model (e.g., SimpleHGNN)
        target_model: The model being attacked
        train_mask, val_mask, test_mask: Dataset splits
        batch_size: Size of mini-batch
        budget: Sparsity budget for perturbations
        epsilon: Maximum perturbation for features
        T: Number of iterations for the attack
        eta_H, eta_X: Learning rates for perturbations
        alpha: Weight for classification loss
    """
    batch_size=args.batch_size
    # batch_size = 16
    device = data.x.device
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    # Create DataLoader for batching
    data = data.to(device)
    data_loader = DataLoader(TensorDataset(data.node_idx, data.x, data.y), batch_size=batch_size, shuffle=True)
    # verbose = True 
    # Track metrics for each iteration
    loss_meta_trajectory = []
    acc_drop_trajectory = []
    lap_shift_trajectory = []
    lap_dist_trajectory = []
    cls_loss_trajectory = []
    deg_penalty_trajectory = []
    feature_shift_trajectory = []
    surrogate_test_trajectory = []
    target_test_trajectory = []
    if surrogate_class is None:
        surrogate_model = SimpleHGNN(data.x.shape[1], hidden_dim = args.MLP_hidden, out_dim = args.num_classes, device = device).to(device)
        optimizer = torch.optim.Adam(surrogate_model.parameters(),lr=args.lr)
    else:
        # surrogate_model = surrogate_class(X.shape[1],)
        raise Exception("Other surrogates Not implemented")
    # Construct full incidence matrix
    # full_num_nodes = data.x.size(0)
    # full_num_hyperedges = data.num_hyperedges
    # full_incidence_matrix = torch.zeros(full_num_nodes, full_num_hyperedges, dtype=torch.float, device=device)
    # for i in range(data.edge_index.size(1)):
    #     v, e = data.edge_index[0, i], data.edge_index[1, i]
    #     full_incidence_matrix[v, e] = 1.0

    # Construct full incidence matrix (sparse)
    full_num_nodes = data.x.size(0)
    full_num_hyperedges = data.num_hyperedges
    full_indices = data.edge_index
    full_values = torch.ones(full_indices.size(1), device=device)
    H = torch.sparse_coo_tensor(
        full_indices, full_values, size=(full_num_nodes, full_num_hyperedges), device=device
    ).coalesce()
    # print('H.shape: ',H.shape)
    # global_delta_H = (1e-3 * torch.randn((data.n_x, data.num_hyperedges), device=device))
    # global_delta_H = (1e-3 * torch.randn(H.shape)).to(device)
    # global_delta_X = (1e-3 * torch.randn_like(data.x)).to(device)
    global_delta_H = torch.zeros(H.shape).to(device)
    global_delta_X = torch.zeros_like(data.x).to(device)
    for t in tqdm(range(T)):
        # global_delta_X[:] = 0
        # global_delta_H[:] = 0
        runtime_start = time.time()
        criterion = nn.CrossEntropyLoss()
        # Loop over mini-batches
        for batch in data_loader:
            batch_data, local_H = create_minibatch_for_hcha(data, batch, split_idx,construct_H=True)  # Create mini-batch for this batch
            batch_data = batch_data.to(device)  # Move to the appropriate device
            
            # Construct the local incidence matrix for this batch
            # local_H = create_local_H(batch_data)
            L_orig = lap(local_H)
            dv_orig = local_H @ torch.ones((local_H.shape[1],), device=device)
            local_X = batch_data.x
            # delta_H = global_delta_H.index_select(0, batch_data.node_idx_orig)
            # delta_H = global_delta_H[batch_data.node_idx_orig,:][:,batch_data.batch_edge_index[1]-batch_data.batch_edge_index[1].min()]

            if t == 0:
                delta_X = (1e-3 * torch.randn_like(local_X)).requires_grad_()
                delta_H = (1e-3 * torch.randn_like(local_H)).requires_grad_()
            else:
                delta_X = global_delta_X.index_select(0, batch_data.node_idx_orig)
                delta_H = global_delta_H[batch_data.node_idx_orig,:][:,batch_data.involved_hyperedges-batch_data.involved_hyperedges.min()]

                # delta_H = global_delta_H[batch_data.node_idx_orig,:][:,batch_data.batch_edge_index[1]-batch_data.batch_edge_index[1].min()]

            # delta_X = global_delta_X.index_select(0, batch_data.node_idx_orig)
            # print('deltaH.shape: ',delta_H.shape)
            # print('delta_H requires grad? => ',delta_H.requires_grad)
            # print('delta_X requires grad? =>', delta_X.requires_grad)
            # # Perturbation for each batch
            # print('local_H shape: ',local_H.shape, ' delta_H shape: ',delta_H.shape)
            H_pert = torch.clamp(local_H + delta_H, 0, 1)
            X_pert = local_X + delta_X
            # Train the surrogate model on the perturbed data
            surrogate_model.train()
            surrogate_epochs = args.num_epochs_sur
            best_val_accuracy = -float('inf')
            patience_counter = 0
            # data.x = X_pert
            # data.edge_index = incidence_to_edge_index(H_pert)
            # H_pert = H_pert.cpu()
            # X_pert = X_pert.cpu()
            for epoch in range(surrogate_epochs):
                optimizer.zero_grad()
                logits = surrogate_model(X_pert, H_pert)
                # print(logits[train_mask].shape,y[train_mask].shape,logits.shape,y.shape)
                loss = criterion(logits[batch_data.train_mask],batch_data.y[batch_data.train_mask])
                loss.backward(retain_graph=True)
                optimizer.step()
                 # Optionally track validation accuracy of the surrogate model
            
                surrogate_model.eval()
                with torch.no_grad():
                    val_accuracy = accuracy(logits[batch_data.val_mask], batch_data.y[batch_data.val_mask]).item()
                    test_accuracy = accuracy(logits[batch_data.test_mask], batch_data.y[batch_data.test_mask]).item()
                # if epoch % 2 == 0 and verbose:
                #     print(f'Epoch {epoch}/{surrogate_epochs}, Surrogate Model train_loss: {loss.item()}, Validation Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

                # Save the surrogate model (which has the best validation accuracy) for robust training
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                    best_model_state = surrogate_model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        # print(f"Early stopping at epoch {epoch} with best validation accuracy: {best_val_accuracy:.4f}")
                        break
            surrogate_model.load_state_dict(best_model_state) # Take the best model

            # # Evaluate the model's performance
            # surrogate_model.eval()
            # with torch.no_grad():
            #     val_accuracy = accuracy(logits[batch_data.val_mask], batch_data.y[batch_data.val_mask])
            #     acc_drop = (logits_orig.argmax(dim=1)[test_mask] == y[test_mask]).float().mean().item()
            Z = surrogate_model(X_pert, H_pert) # Trained surrogate model
            # Compute Laplacian and degree penalties
            L_pert = lap(H_pert)
            delta_L = (L_pert - L_orig) @ Z
            lap_dist = torch.norm(delta_L, p=2).mean()

            H_temp = torch.clamp(H_pert, 0, 1)
            dv_temp = H_temp @ torch.ones((local_H.shape[1],), device=device)
            degree_violation = (dv_temp - dv_orig)
            degree_penalty = torch.sum(degree_violation ** 2) / n
            # deg_penalty_val = degree_penalty.item()

            # Compute classification loss
            loss_cls = F.cross_entropy(Z, batch_data.y)
            
            # Compute total Meta-Loss
            loss_meta = lap_dist - degree_penalty + alpha * loss_cls


            # Update the perturbations (delta_H and delta_X)
            grads = torch.autograd.grad(loss_meta, [delta_H, delta_X])
            with torch.no_grad():
                delta_H += eta_H * grads[0].sign()
                delta_X += eta_X * grads[1].sign()
                flat = delta_H.abs().flatten()
                topk = torch.topk(flat, k=min(delta_H.numel(), max(1,budget//batch_size))).indices
                delta_H_new = torch.zeros_like(delta_H)
                delta_H_new.view(-1)[topk] = delta_H.view(-1)[topk]

                # In the following code segment we do not update bad nodes (nodes whose deg <= 0 ) or bad edges (whose card <= 1)
                H_temp = torch.clamp(local_H + delta_H_new, 0, 1)
                row_degrees = H_temp.sum(dim=1)
                col_degrees = H_temp.sum(dim=0)
                bad_nodes = (row_degrees < 1).nonzero(as_tuple=True)[0]
                bad_edges = (col_degrees < 2).nonzero(as_tuple=True)[0]
                delta_H_new[bad_nodes, :] = 0
                delta_H_new[:, bad_edges] = 0
                delta_H.copy_(delta_H_new)
            delta_X = delta_X.clamp(-epsilon, epsilon)
            # Track loss and accuracy drop over iterations
            lap_dist_trajectory.append(lap_dist.item())
            loss_meta_trajectory.append(loss_meta.item())
            # print(batch_data.node_idx_orig)
            global_delta_X.index_add_(0, batch_data.node_idx_orig, delta_X)
            # print(global_delta_H.shape, delta_H.shape, batch_data.node_idx_orig.shape)
            # global_delta_H.index_add_(0, batch_data.node_idx_orig, delta_H)
            row_indices = batch_data.node_idx_orig.flatten()
            col_indices = batch_data.involved_hyperedges-batch_data.involved_hyperedges.min()
            R, C = len(row_indices), len(col_indices)
            rows = row_indices.unsqueeze(1).expand(-1, C)      # shape [R, C]
            cols = col_indices.unsqueeze(0).expand(R, -1)      # shape [R, C]
            # print(rows.shape, cols.shape,R,C)
            # print(rows, cols)
            global_delta_H[rows,cols] += delta_H 
            # print('addition done')
        # Store final model state
        # if t == T-1:
        #     surrogate_model.eval()
        #     surrogate_test_accuracy = accuracy(surrogate_model(X_pert, H_pert)[test_mask], y[test_mask])

        #     # Collect final results after T iterations
        #     surrogate_test_trajectory.append(surrogate_test_accuracy.item())
        #     target_model.eval()
        #     target_test_accuracy = accuracy(target_model(X_pert, H_pert)[test_mask], y[test_mask])
        #     target_test_trajectory.append(target_test_accuracy.item())

        # # Save model if best state
        # if t == T-1:
        #     os.makedirs(os.path.join(root, str(args.seed)), exist_ok=True)
        #     prefix = os.path.join(root, str(args.seed), 'SimpleHGNN_' + args.dataset + '_' + args.model + '_' + str(args.ptb_rate))
        #     torch.save(surrogate_model.state_dict(), prefix + '_weights.pth')

        # Track metrics for logging
        time1 = time.time() - runtime_start
        # results = [loss_meta_trajectory, acc_drop_trajectory, lap_shift_trajectory, lap_dist_trajectory, cls_loss_trajectory, deg_penalty_trajectory, feature_shift_trajectory, surrogate_test_trajectory, target_test_trajectory]
    H_pert = H+global_delta_H.to_sparse()
    X_pert = (data.x + global_delta_X).clamp(-epsilon,epsilon)
    return H_pert, X_pert, surrogate_model.state_dict()

def get_attack(target_model,H,X,y,data,HG,train_mask,val_mask,test_mask,perturbations):
    if args.attack == 'gradargmax':
        raise ValueError("gradargmax not implemented in this version")
        if args.method != 'simplehgnn':
            target_model = SimpleHGNN(X.shape[1],hidden_dim = args.MLP_hidden, out_dim = args.num_classes,device = X.device)
        # print('surrogate : ',target_model)
        # attack_model = GradArgmax(model=target_model.to(device), nnodes=X.shape[0], nnedges = H.shape[1], \
        #                         attack_structure=True, device=device)
        time1 = time.time()
        attack_model.attack(X, H.clone(), y, n_perturbations=perturbations )
        exec_time = time.time() - time1
        H_adv = attack_model.modified_H
        row_degrees = H_adv.sum(dim=1)
        col_degrees = H_adv.sum(dim=0)
        bad_nodes = (row_degrees < 1).nonzero(as_tuple=True)[0]
        bad_edges = (col_degrees < 2).nonzero(as_tuple=True)[0]
        H_adv[bad_nodes, :] = H[bad_nodes,:]
        H_adv[:, bad_edges] = H[:,bad_edges]
        X_adv = X.clone()
    elif args.attack == 'Rand-flip':
        time1 = time.time()
        H_adv = H.clone()
        n, m = H.shape
        total_elements = n * m

        # Flatten index space and choose delta random indices to modify
        indices = torch.randperm(total_elements, device=H.device)[:perturbations]

        # Convert flat indices to 2D indices (rows and columns)
        rows = indices // m
        cols = indices % m

        # Generate random signs (-1 or +1) for each of the delta indices
        signs = torch.randint(0, 2, (perturbations,), device=H.device) * 2 - 1  # {-1, +1}

        # Directly apply the perturbations to the selected indices
        H_adv[rows, cols] += signs
        X_adv = X.clone()
        row_degrees = H_adv.sum(dim=1)
        col_degrees = H_adv.sum(dim=0)
        bad_nodes = (row_degrees < 1).nonzero(as_tuple=True)[0]
        bad_edges = (col_degrees < 2).nonzero(as_tuple=True)[0]
        H_adv[bad_nodes, :] = H[bad_nodes,:]
        H_adv[:, bad_edges] = H[:,bad_edges]
        exec_time = time.time() - time1
    elif args.attack == 'Rand-feat':
        time1 = time.time()
        sign = torch.randint(0, 2, X.shape, dtype=torch.float32, device=X.device) * 2 - 1  # ∈ {-1, +1}
        # Apply perturbation
        perturbation = args.epsilon * sign
        X_adv = X.clone() + perturbation
        exec_time = time.time() - time1
        H_adv = H.clone()

    else:
        raise Exception("Attack not implemented")
    
    return H_adv, X_adv, exec_time


def parse_method(args, data, data_p):
    #     Currently we don't set hyperparameters w.r.t. different dataset
    if args.method == 'AllSetTransformer':
        if args.LearnMask:
            model = SetGNN(args, data.norm)
        else:
            model = SetGNN(args)
    
    elif args.method == 'AllDeepSets':
        args.PMA = False
        # args.aggregate = 'add'
        args.__setattr__('aggregate','add')
        if args.LearnMask:
            model = SetGNN(args,data.norm)
        else:
            model = SetGNN(args)

#     elif args.method == 'SetGPRGNN':
#         model = SetGPRGNN(args)

    elif args.method == 'CEGCN':
        model = CEGCN(in_dim=args.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.num_classes,
                      num_layers=args.All_num_layers,
                      dropout=args.dropout,
                      Normalization=args.normalization)

    elif args.method == 'CEGAT':
        model = CEGAT(in_dim=args.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.num_classes,
                      num_layers=args.All_num_layers,
                      heads=args.heads,
                      output_heads=args.output_heads,
                      dropout=args.dropout,
                      Normalization=args.normalization)

    elif args.method == 'HyperGCN':
        #         ipdb.set_trace()
        
        He_dict = get_HyperGCN_He_dict(data.cpu())
        He_dict_p = get_HyperGCN_He_dict(data_p.cpu())
        model = HyperGCN(V=data.x.shape[0],
                         E=He_dict,
                         E_p=He_dict_p,
                         X=data.x,
                         num_features=args.num_features,
                         num_layers=args.All_num_layers,
                         num_classses=args.num_classes,
                         args=args
                         )

    elif args.method == 'HGNN':
        # model = HGNN(in_ch=args.num_features,
        #              n_class=args.num_classes,
        #              n_hid=args.MLP_hidden,
        #              dropout=args.dropout)
        model = HCHA(args)

    elif args.method == 'HNHN':
        model = HNHN(args)

    elif args.method == 'HCHA':
        model = HCHA(args)

    elif args.method == 'MLP':
        model = MLP_model(args)
    elif args.method == 'UniGCNII':
            if args.cuda in [0,1,2,3,4,5,6,7]:
                device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device('cpu')
            (row, col), value = torch_sparse.from_scipy(data.edge_index)
            V, E = row, col
            V, E = V.to(device), E.to(device)
            
            (row_p, col_p), value_p = torch_sparse.from_scipy(data_p.edge_index)
            V_p, E_p = row_p, col_p
            V_p, E_p = V_p.to(device), E_p.to(device)
            model = UniGCNII(args, nfeat=args.num_features, nhid=args.MLP_hidden, nclass=args.num_classes, nlayer=args.All_num_layers, nhead=args.heads,
                             V=V, E=E, V_p=V_p, E_p=E_p)
    #     Below we can add different model, such as HyperGCN and so on
    return model


class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            return best_result[:, 1], best_result[:, 3]

    def plot_result(self, run=None):
        plt.style.use('seaborn')
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            x = torch.arange(result.shape[0])
            plt.figure()
            print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])
        else:
            result = 100 * torch.tensor(self.results[0])
            x = torch.arange(result.shape[0])
            plt.figure()
#             print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])


@torch.no_grad()
def evaluate(args, model, data_inp, eval_func, result=None):
    # test_flag = True
    # if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
    #     data_input = [data, test_flag]
    # else:
    #     data_input = data
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(data_inp)
        out = F.log_softmax(out, dim=1)
    # print(len(train_mask),' ',sum(train_mask))
    train_acc = eval_func(
        out[data_inp.train_mask],data_inp.y[data_inp.train_mask])
    valid_acc = eval_func(
        out[data_inp.val_mask],data_inp.y[data_inp.val_mask])
    test_acc = eval_func(
        out[data_inp.test_mask],data_inp.y[data_inp.test_mask])

#     Also keep track of losses
    train_loss = F.nll_loss(
        out[data_inp.train_mask], data_inp.y[data_inp.train_mask])
    valid_loss = F.nll_loss(
        out[data_inp.val_mask], data_inp.y[data_inp.val_mask])
    test_loss = F.nll_loss(
        out[data_inp.test_mask], data_inp.y[data_inp.test_mask])
    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

#     ipdb.set_trace()
#     for i in range(y_true.shape[1]):
    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def perturb_hyperedges(data, prop, perturb_type='delete'):
    data_p = copy.deepcopy(data)
    edge_index = data_p.edge_index
    num_node = data.x.shape[0]
    e_idxs = edge_index[1,:] - num_node
    num_edge = (edge_index[1,:].max()) - (edge_index[1,:].min())
    if((perturb_type == 'delete') or (perturb_type == 'replace')):
        p_num = num_edge * prop
        p_num = int(p_num)
        if p_num == 0:
            return data
        chosen_edges = torch.as_tensor(np.random.permutation(int(num_edge.numpy()))).to(edge_index.device)
        chosen_edges = chosen_edges[:p_num]
        if(perturb_type == 'delete'):
            data_p.edge_index = delete_edges(edge_index, chosen_edges, e_idxs)
        else: # replace = add + delete
            data_p.edge_index = replace_edges(edge_index, chosen_edges, e_idxs, num_node)
    elif(perturb_type == 'add'):
        # p_num = num_edge * prop / (1 - prop)
        p_num = num_edge * prop
        p_num = int(p_num)
        if p_num == 0:
            return data
        data_p.edge_index = add_edges(edge_index, p_num, num_node)
    return data_p

def delete_edges(edge_index, chosen_edges, e_idxs):
    for i in range(chosen_edges.shape[0]):
        chosen_edge = chosen_edges[i]
        edge_index = edge_index[:, (e_idxs != chosen_edge)]
        e_idxs = e_idxs[(e_idxs != chosen_edge)]
    return edge_index

def replace_edges(edge_index, chosen_edges, e_idxs, num_node):
    edge_index = delete_edges(edge_index, chosen_edges, e_idxs)
    edge_index = add_edges_r(edge_index, chosen_edges, num_node)
    return edge_index

def add_edges_r(edge_index, chosen_edges, num_node):
    edge_idxs = [edge_index]
    for i in range(chosen_edges.shape[0]):
        new_edge = torch.as_tensor(np.random.choice(int(num_node), 16, replace=False)).to(edge_index.device)
        for j in range(new_edge.shape[0]):
            edge_idx_i = torch.zeros([2,1]).to(edge_index.device)
            edge_idx_i[0,0] = new_edge[j]
            edge_idx_i[1,0] = chosen_edges[i] + num_node
            edge_idxs.append(edge_idx_i)
    edge_idxs = torch.cat(edge_idxs, dim=1)
    return torch.tensor(edge_idxs, dtype=torch.int64)
    
def add_edges(edge_index, p_num, num_node):
    start_e_idx = edge_index[1,:].max() + 1
    edge_idxs = [edge_index]
    for i in range(p_num):
        new_edge = torch.as_tensor(np.random.choice(int(num_node.cpu().numpy()), 5, replace=False)).to(edge_index.device)
        for j in range(new_edge.shape[0]):
            edge_idx_i = torch.zeros([2,1]).to(edge_index.device)
            edge_idx_i[0,0] = new_edge[j]
            edge_idx_i[1,0] = start_e_idx
            edge_idxs.append(edge_idx_i)
        start_e_idx = start_e_idx + 1
    edge_idxs = torch.cat(edge_idxs, dim=1)
    return torch.tensor(edge_idxs, dtype=torch.int64)

def unignn_ini_ve(data, device):
    data = ConstructH(data)
    data.edge_index = sp.csr_matrix(data.edge_index)
    # Compute degV and degE
    (row, col), value = torch_sparse.from_scipy(data.edge_index)
    V, E = row, col
    return V, E

def unignn_get_deg(V, E):
    degV = torch.from_numpy(data.edge_index.sum(1)).view(-1, 1).float().to(device)
    from torch_scatter import scatter
    degE = scatter(degV[V], E, dim=0, reduce='mean')
    degE = degE.pow(-0.5)
    degV = degV.pow(-0.5)
    degV[torch.isinf(degV)] = 1
    return degV, degE

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True 
    torch.cuda.current_device()
    torch.cuda._initialized = True

# --- Main part of the training ---
# # Part 0: Parse arguments


"""

"""
def create_minibatch(data, split_idx, batch_size,device):
    # Get a random subset of nodes and corresponding edges
    # print('create minibatch: ', data)
    # print(data.edge_index[0][:20])
    # print(data.edge_index[1][:20])
    # print(data.edge_index[0][-10:])
    # print(data.edge_index[1][-10:])
    # print(data.edge_index[0].max(),data.edge_index[1].min())
    num_nodes = data.n_x
    num_hyperedges = data.num_hyperedges
    num_nodes = data.x.shape[0]
    rand_nodes = torch.randperm(num_nodes)[:batch_size].to(device)  # Randomly sample nodes for the batch
    edge_mask = torch.isin(data.edge_index[0], rand_nodes) #| torch.isin(data.edge_index[1], rand_nodes)
    edge_index_batch = data.edge_index[:, edge_mask]

    # rand_edge_idx = torch.randperm(data.edge_index.shape[1])[:batch_size]
    # # rand_edges = data.edge_index[1][rand_edge_idx]  # Randomly sample edges for the batch
    # # edge_mask = torch.isin(data.edge_index[1], rand_edges) #| torch.isin(data.edge_index[1], rand_nodes)
    # edge_index_batch = data.edge_index[:, rand_edge_idx]
    # rand_nodes = edge_index_batch[0].unique()
    
    data_batch = data.clone()
    data_batch.edge_index = edge_index_batch
    # print(edge_index_batch.shape)
    # print('edge_index_batch: ',edge_index_batch[1].min(),edge_index_batch[1].max(),\
    #       edge_index_batch[0].min(),edge_index_batch[0].max())
    print(edge_index_batch[0].unique().shape[0],edge_index_batch[1].unique().shape[0])

    data_batch.x = data.x[rand_nodes]
    data_batch.y = data.y[rand_nodes]

    data_batch.n_x = batch_size
    
    data_batch.num_hyperedges = edge_index_batch[1].unique().shape[0]
    data_batch.edge_index[1] -= data_batch.edge_index[1].min()
    # print('rand_nodes: ',rand_nodes)
    rand_nodes = rand_nodes.cpu()
    # print(split_idx['train'])
    # train_idx_batch = torch.tensor([i for i,_ in split_idx['train'] if i in rand_nodes])
    # print('train_idx_batch:',train_idx_batch)
    # test_idx_batch = torch.tensor([i for i in split_idx['test'] if i in rand_nodes])
    # val_idx_batch = torch.tensor([i for i in split_idx['valid'] if i in rand_nodes])
    train_idx_batch = split_idx['train'][rand_nodes]
    val_idx_batch = split_idx['valid'][rand_nodes]
    test_idx_batch = split_idx['test'][rand_nodes]
    data_batch.train_mask = train_idx_batch.to(device)
    data_batch.test_mask = test_idx_batch.to(device)
    data_batch.val_mask = val_idx_batch.to(device)
    return data_batch

def create_minibatch_for_hcha(data, batch_data, split_idx, add_self_loops=True, construct_H = False, verbose = False):
    """
    Constructs a mini-batch compatible with HCHA/HypergraphConv.
    - Maintains bipartite incidence format.
    - Optionally adds self-loop hyperedges for each node.
    
    Returns:
        data_batch: Mini-batch PyG Data object.
        train_idx_batch: Local indices of train nodes in the batch.
    """
    device = data.x.device
    # num_nodes = data.x.shape[0]
    edge_index = data.edge_index  
    batch_nodes = batch_data[0].to(device) 
    batch_size = len(batch_nodes)
    # print(batch_nodes)
    # Sample nodes
    # batch_nodes = torch.randperm(num_nodes)[:batch_size].to(device)
    node_mask = torch.zeros(data.n_x, dtype=torch.long, device=device)
    node_mask[batch_nodes] = 1

    # # Find hyperedges connected to sampled nodes
    src_nodes, edge_ids = edge_index[0], edge_index[1]
    mask = node_mask[src_nodes]
    batch_edge_index = edge_index[:,mask.bool()]
    # if construct_H:
    #     print('batch_edgeindex shape: ',batch_edge_index.shape)
    #     print(mask.shape,mask.sum())
    involved_hyperedges = batch_edge_index[1].unique()
    # print('involved min,max: ',involved_hyperedges.min(),involved_hyperedges.max())
    he_id_old2new = {int(e.item()): i for i, e in enumerate(involved_hyperedges)}
    node_id_old2new = {int(n.item()): i for i, n in enumerate(batch_nodes)}

    # Reindex
    reindexed = []
    for i in range(batch_edge_index.shape[1]):
        v, e = int(batch_edge_index[0, i]), int(batch_edge_index[1, i])
        if v in node_id_old2new and e in he_id_old2new:
            reindexed.append([
                node_id_old2new[v],
                he_id_old2new[e]
            ])
    # if len(reindexed) == 0:
    #     print('reindexed is empty')
    #     print('len(involved hyperedges): ',len(involved_hyperedges))
    #     print('batch_nodes: ',batch_nodes)
    #     print(batch_edge_index[0,:10])
    #     print(batch_edge_index[1,:10])
    #     print(batch_edge_index.shape[1])
    #     v, e = int(batch_edge_index[0, 0]), int(batch_edge_index[1, 0])
    #     print('v, e: ',v,e)
    reindexed = torch.tensor(reindexed, dtype=torch.long, device=device).T  # shape [2, num_edges]
    # print('reindex.shape: ',reindexed.shape)
    # if verbose:
    #     print(reindexed)
    if construct_H:
        # Construct incidence matrix for the mini-batch
        incidence_matrix = torch.zeros(batch_size, len(involved_hyperedges), dtype=torch.float, device=device)
        for i in range(reindexed.shape[0]):
            node_idx = reindexed[0, i]
            he_idx = reindexed[1, i]
            incidence_matrix[node_idx, he_idx] = 1.0
        # print('agree: ',batch_edge_index[1].shape, incidence_matrix.shape)
        # if batch_edge_index[1].shape[0] != incidence_matrix.shape[1]:
        #     print(batch_edge_index[1])
        #     print(incidence_matrix.sum())
            # print(reindexed)
    # Add self-loop hyperedges (optional)
    if add_self_loops:
        start_id = len(involved_hyperedges)
        self_loops = torch.stack([
            torch.arange(batch_size, device=device),
            torch.arange(start_id, start_id + batch_size, device=device)
        ])
        reindexed = torch.cat([reindexed, self_loops], dim=1)

    # # Slice node features and labels
    # x_batch = data.x[batch_nodes]
    # y_batch = data.y[batch_nodes]

    # Local training indices
    # train_nodes_set = set(split_idx['train'].cpu().numpy())
    # train_idx_batch = [i for i, n in enumerate(batch_nodes.cpu().numpy()) if n in train_nodes_set]
    # train_idx_batch = torch.tensor(train_idx_batch, dtype=torch.long, device=device)
    
    rand_nodes = batch_nodes.cpu()
    train_idx_batch = split_idx['train'][rand_nodes]
    val_idx_batch = split_idx['valid'][rand_nodes]
    test_idx_batch = split_idx['test'][rand_nodes]

    # Construct mini-batch PyG Data object
    data_batch = Data(
        node_idx_orig = batch_data[0],
        x=batch_data[1],
        y=batch_data[2],
        edge_index=reindexed,
        involved_hyperedges = involved_hyperedges
        # batch_edge_index = batch_edge_index
    )
    data_batch.train_mask = train_idx_batch.to(device)
    data_batch.test_mask = test_idx_batch.to(device)
    data_batch.val_mask = val_idx_batch.to(device)
    if args.method in ['AllSetTransformer', 'AllDeepSets']:
        data_batch = norm_contruction(data_batch, option=args.normtype)
    if construct_H:
        return data_batch, incidence_matrix 
    else:
        return data_batch

# def inference_on_test_set(model, data, split_idx, device):
#     """
#     Perform inference on the test set using a mini-batch approach.

#     Args:
#         model: The trained model.
#         data: The full dataset (including features and edge indices).
#         batch_size: The batch size for mini-batches.
#         split_idx: Dictionary containing the train, validation, and test splits.
#         device: The device on which the data is loaded (e.g., 'cuda' or 'cpu').

#     Returns:
#         test_accuracy: Accuracy on the test set.
#     """
#     model.eval()  # Set model to evaluation mode
#     # test_mask = data.test_mask  # Test indices (for masking)
    
#     data_loader = DataLoader(TensorDataset(data.node_idx, data.x, data.y), batch_size=args.batch_size, shuffle=False)

#     all_preds = []
#     all_labels = []
    
#     # Inference loop over mini-batches
#     with torch.no_grad():  # No need to compute gradients during inference
#         for batch in data_loader:
#             # Create a mini-batch for the test set
#             batch_data = create_minibatch_for_hcha(data, batch, split_idx)
#             test_mask = batch_data.test_mask  # Test indices (for masking)
#             # Apply the test mask to select only test nodes
#             # batch_data.x = batch_data.x[test_mask]  # Only test set nodes
#             batch_data.y = batch_data.y[test_mask]  # Only test set labels
#             # batch_data.edge_index = batch_data.edge_index[:, test_mask]  # Only edges connected to test nodes

#             # Forward pass
#             out = model(batch_data)
#             preds = out.argmax(dim=1)  # Get the predicted class labels
            
#             # Append predictions and true labels
#             all_preds.append(preds[test_mask])
#             all_labels.append(batch_data.y)

#     # Concatenate all predictions and true labels
#     all_preds = torch.cat(all_preds, dim=0)
#     all_labels = torch.cat(all_labels, dim=0)

#     # Calculate accuracy
#     correct = all_preds.eq(all_labels).sum().item()
#     accuracy = correct / len(all_labels) * 100

#     return accuracy, all_preds
def sparse_change_stats(H: torch.Tensor, H_adv: torch.Tensor):
    """
    Assumes binary incidence stored as sparse COO with values 1.
    Returns: (num_changed_entries, num_changed_edges, num_changed_nodes)
    """
    assert H.is_sparse and H_adv.is_sparse
    H = H.coalesce()
    H_adv = H_adv.coalesce()
    device = H.device
    assert H_adv.device == device
    n, m = H.shape

    # Linearize indices to compare supports
    idx1 = H.indices()        # [2, nnz1]
    idx2 = H_adv.indices()    # [2, nnz2]
    lin1 = idx1[0] * m + idx1[1]
    lin2 = idx2[0] * m + idx2[1]

    # Symmetric difference of supports
    in2 = torch.isin(lin1, lin2)
    in1 = torch.isin(lin2, lin1)

    removed = idx1[:, ~in2]   # entries that were 1 and became 0
    added   = idx2[:, ~in1]   # entries that were 0 and became 1

    diff_idx = torch.cat([removed, added], dim=1)  # [2, K]
    num_changed = diff_idx.size(1)

    if num_changed == 0:
        return 0, 0, 0

    nodes_changed = torch.unique(diff_idx[0]).numel()
    edges_changed = torch.unique(diff_idx[1]).numel()
    return num_changed, edges_changed, nodes_changed

def sparse_H_to_edge_index(H: torch.Tensor) -> torch.Tensor:
    """
    Converts a sparse incidence matrix H (n x m) to a PyTorch Geometric edge_index
    in bipartite format: edge_index = [V -> E; E -> V], both directions.
    
    This version handles sparse matrices efficiently.

    Args:
        H (torch.Tensor): Sparse incidence matrix of shape (n, m), where H[i, j] = 1
                          if node i is in hyperedge j.

    Returns:
        torch.Tensor: edge_index of shape (2, 2 * num_nonzeros)
                      in bipartite form where hyperedges are shifted by n.
    """
    # Ensure H is sparse
    if not H.is_sparse:
        H = H.to_sparse()

    # Get all non-zero indices (i, j) where H[i, j] = 1
    node_idx, edge_idx = H._indices()

    # Get the number of nodes and edges
    n = H.size(0)  # number of nodes
    m = H.size(1)  # number of hyperedges

    # Shift hyperedge indices by n to treat them as distinct nodes
    edge_nodes = edge_idx + n

    # Create bidirectional edges between nodes and hyperedges
    row1 = torch.cat([node_idx, edge_nodes])  # source nodes
    row2 = torch.cat([edge_nodes, node_idx])  # target nodes

    # Stack the rows to form the edge_index (2, 2*num_nonzeros)
    edge_index = torch.stack([row1, row2], dim=0)
    
    return edge_index

def inference_on_test_set(model, data, split_idx):
    """
    Perform inference on the test set using a mini-batch approach.

    Args:
        model: The trained model.
        data: The full dataset (including features and edge indices).
        batch_size: The batch size for mini-batches.
        split_idx: Dictionary containing the train, validation, and test splits.
        device: The device on which the data is loaded (e.g., 'cuda' or 'cpu').

    Returns:
        test_accuracy: Accuracy on the test set.
    """
    model.eval()  # Set model to evaluation mode
    # test_mask = data.test_mask  # Test indices (for masking)
    
    data_loader = DataLoader(TensorDataset(data.node_idx, data.x, data.y), batch_size=args.batch_size, shuffle=False)

    all_preds_test = []
    all_preds_train = []
    all_preds_val = []
    all_labels = []
    all_labels_train = []
    all_labels_val = []
    all_labels_test = []
    # print('inference')
    # Inference loop over mini-batches
    with torch.no_grad():  # No need to compute gradients during inference
        for batch in data_loader:
            # Create a mini-batch for the test set
            batch_data = create_minibatch_for_hcha(data, batch, split_idx)
            test_mask = batch_data.test_mask  # Test indices (for masking)
            # Apply the test mask to select only test nodes
            # batch_data.x = batch_data.x[test_mask]  # Only test set nodes
            # batch_data.y = batch_data.y[test_mask]  # Only test set labels
            # batch_data.edge_index = batch_data.edge_index[:, test_mask]  # Only edges connected to test nodes

            # Forward pass
            out = model(batch_data)
            preds = out.argmax(dim=1)  # Get the predicted class labels
            
            # Append predictions and true labels
            all_preds_train.append(preds[batch_data.train_mask])
            all_preds_val.append(preds[batch_data.val_mask])
            all_preds_test.append(preds[batch_data.test_mask])
            all_labels.append(batch_data.y)
            all_labels_train.append(batch_data.train_mask)
            all_labels_val.append(batch_data.val_mask)
            all_labels_test.append(batch_data.test_mask)

    # Concatenate all predictions and true labels
    all_preds_train = torch.cat(all_preds_train, dim=0)
    all_preds_val = torch.cat(all_preds_val, dim=0)
    all_preds_test = torch.cat(all_preds_test, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    # print(all_labels.shape,torch.cat(all_labels_train,dim=0).shape)

    all_labels_train = all_labels[torch.cat(all_labels_train,dim=0)]
    all_labels_val = all_labels[torch.cat(all_labels_val,dim=0)]
    all_labels_test = all_labels[torch.cat(all_labels_test,dim=0)]

    # Calculate accuracy
    # correct = all_preds_test.eq(all_labels).sum().item()
    # train_accuracy = correct / len(all_labels) * 100
    test_accuracy = all_preds_test.eq(all_labels_test).sum().item() / len(all_labels_test) * 100
    train_accuracy = all_preds_train.eq(all_labels_train).sum().item() / len(all_labels_train) * 100
    val_accuracy = all_preds_val.eq(all_labels_val).sum().item() / len(all_labels_val) * 100
    return torch.cat([all_preds_train, all_preds_val,all_preds_test]), train_accuracy, val_accuracy, test_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--dname', default='walmart-trips-100')
    # method in ['SetGNN','CEGCN','CEGAT','HyperGCN','HGNN','HCHA']
    parser.add_argument('--method', default='AllSetTransformer')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--seed', default=1, type=int)
    # Number of runs for each split (test fix, only shuffle train/val)
    parser.add_argument('--runs', default=20, type=int)
    parser.add_argument('--cuda', default=0, choices=[-1,0,1,2,3,4,5,6,7], type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--wd', default=0.0, type=float)
    # How many layers of full NLConvs
    parser.add_argument('--All_num_layers', default=2, type=int)
    parser.add_argument('--MLP_num_layers', default=2,
                        type=int)  # How many layers of encoder
    parser.add_argument('--MLP_hidden', default=64,
                        type=int)  # Encoder hidden units
    parser.add_argument('--Classifier_num_layers', default=2,
                        type=int)  # How many layers of decoder
    parser.add_argument('--Classifier_hidden', default=64,
                        type=int)  # Decoder hidden units
    parser.add_argument('--display_step', type=int, default=-1)
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    # ['all_one','deg_half_sym']
    parser.add_argument('--normtype', default='all_one')
    parser.add_argument('--add_self_loop', action='store_false')
    # NormLayer for MLP. ['bn','ln','None']
    parser.add_argument('--normalization', default='ln')
    parser.add_argument('--deepset_input_norm', default = True)
    parser.add_argument('--GPR', action='store_false')  # skip all but last dec
    # skip all but last dec
    parser.add_argument('--LearnMask', action='store_false')
    parser.add_argument('--num_features', default=0, type=int)  # Placeholder
    parser.add_argument('--num_classes', default=0, type=int)  # Placeholder
    # Choose std for synthetic feature noise
    parser.add_argument('--feature_noise', default='1', type=str)
    parser.add_argument('--perturb_type', default='delete', type=str)
    parser.add_argument('--perturb_prop', default=0.0, type=float)
    # whether the he contain self node or not
    parser.add_argument('--exclude_self', action='store_true')
    parser.add_argument('--PMA', action='store_true')
    #     Args for HyperGCN
    parser.add_argument('--HyperGCN_mediators', action='store_true')
    parser.add_argument('--HyperGCN_fast', action='store_true')
    #     Args for Attentions: GAT and SetGNN
    parser.add_argument('--heads', default=1, type=int)  # Placeholder
    parser.add_argument('--output_heads', default=1, type=int)  # Placeholder
    #     Args for HNHN
    parser.add_argument('--HNHN_alpha', default=-1.5, type=float)
    parser.add_argument('--HNHN_beta', default=-0.5, type=float)
    parser.add_argument('--HNHN_nonlinear_inbetween', default=True, type=bool)
    #     Args for HCHA
    parser.add_argument('--HCHA_symdegnorm', action='store_true')
    #     Args for UniGNN
    parser.add_argument('--UniGNN_use-norm', action="store_true", help='use norm in the final layer')
    parser.add_argument('--UniGNN_degV', default = 0)
    parser.add_argument('--UniGNN_degE', default = 0)
    ## Attack args
    parser.add_argument('--attack', type=str, default='mla', \
                    choices=['mla','Rand-flip', 'Rand-feat','gradargmax','mla_fgsm'], help='model variant')
    parser.add_argument('--epsilon', type=float, default=0.05, help='Node Feature perturbation bound')
    parser.add_argument('--ptb_rate', type=float, default=0.2,  help='pertubation rate')
    parser.add_argument('--patience', type=int, default=20,
                    help='Patience for training with early stopping.')
    parser.add_argument('--T', type=int, default=80, help='Number of iterations for the attack.')
    parser.add_argument('--mla_alpha', type=float, default=4.0, help='weight for classification loss')
    parser.add_argument('--eta_H', type=float, default=1e-2, help='Learning rate for H perturbation')
    parser.add_argument('--eta_X', type=float, default=1e-2, help='Learning rate for X perturbation')
    parser.add_argument('--num_epochs_sur', type=int, default=80, help='#epochs for the surrogate training.')
    parser.add_argument('--batch_size', type=int, default=512, help='Size of Minibatch.')
    parser.add_argument('--beta', type=float, default= 1.0, help='weight for degree penalty loss component')
    parser.add_argument('--gamma', type=float, default=4.0, help='weight for classification loss component')
    parser.add_argument('--alpha', type=float, default=4, help='weight for laplacian Loss component')
    parser.add_argument('--loss', type=str, default='L2', help='Loss to measure laplacian distance.', choices=['MSE','L2'])

    # parser.add_argument('--delxdelh',default='Both',choices=['Both','delx','delh'])
    parser.set_defaults(PMA=True)  # True: Use PMA. False: Use Deepsets.
    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(exclude_self=False)
    parser.set_defaults(GPR=False)
    parser.set_defaults(LearnMask=False)
    parser.set_defaults(HyperGCN_mediators=True)
    parser.set_defaults(HyperGCN_fast=True)
    parser.set_defaults(HCHA_symdegnorm=False)
    
    #     Use the line below for .py file
    args = parser.parse_args()
    #     Use the line below for notebook
    # args = parser.parse_args([])
    # args, _ = parser.parse_known_args()
    
    
    # # Part 1: Load data
    root='./base_newsplit'
    # root='./'+args.attack+'_hypergraphMLP_final2'
    os.makedirs(root, exist_ok=True)
    save = True
    # AllSetTransformer co-citeseer mla_fgsm 1
    ### Load and preprocess data ###
    existing_dataset = ['20newsW100', 'ModelNet40', 'zoo',
                        'NTU2012', 'Mushroom',
                        'coauthor_cora', 'coauthor_dblp',
                        'yelp', 'amazon-reviews', 'walmart-trips', 'house-committees',
                        'walmart-trips-100', 'house-committees-100',
                        'cora', 'citeseer', 'pubmed']
    
    synthetic_list = ['amazon-reviews', 'walmart-trips', 'house-committees', 'walmart-trips-100', 'house-committees-100']
    
    if args.dname in existing_dataset:
        dname = args.dname
        f_noise = args.feature_noise
        if (f_noise is not None) and dname in synthetic_list:
            p2raw = '../data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname, 
                    feature_noise=f_noise,
                    p2raw = p2raw)
        else:
            if dname in ['cora', 'citeseer','pubmed']:
                p2raw = '../data/AllSet_all_raw_data/cocitation/'
            elif dname in ['coauthor_cora', 'coauthor_dblp']:
                p2raw = '../data/AllSet_all_raw_data/coauthorship/'
            elif dname in ['yelp']:
                p2raw = '../data/AllSet_all_raw_data/yelp/'
            else:
                p2raw = '../data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname,root = '../data/pyg_data/hypergraph_dataset_updated/',
                                         p2raw = p2raw)
        data = dataset.data
        print('dataset.data: ',data)
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        if args.dname in ['yelp', 'walmart-trips', 'house-committees', 'walmart-trips-100', 'house-committees-100']:
            #         Shift the y label to start with 0
            args.num_classes = len(data.y.unique())
            data.y = data.y - data.y.min()
            print('dataset.data: ',data)
        if not hasattr(data, 'n_x'):
            data.n_x = torch.tensor([data.x.shape[0]])
        if not hasattr(data, 'num_hyperedges'):
            # note that we assume the he_id is consecutive.
            data.num_hyperedges = torch.tensor(
                [data.edge_index[0].max()-data.n_x[0]+1])
    
    # ipdb.set_trace()
    #     Preprocessing
    # if args.method in ['SetGNN', 'SetGPRGNN', 'SetGNN-DeepSet']:
    setup_seed(args.seed)
    batch_size = args.batch_size
    logger = Logger(args.runs, args)

    


    if args.method in ['AllSetTransformer', 'AllDeepSets']:
        data = ExtractV2E(data)
    
    elif args.method in ['HCHA', 'HGNN']:
        # print('if: ',data)
        # print(data.edge_index[1].min(),data.edge_index[1].max())
        # print(data.edge_index[0].min(),data.edge_index[0].max())
        data = ExtractV2E(data)

    edge_index = data.edge_index
    # print(edge_index[0].min(),edge_index[0].max(),edge_index[1].min(),edge_index[1].max())
    # print('unique nodes: ',len(edge_index[0].unique()))
    # print('unique he: ',len(edge_index[1].unique()))

    if args.dname == 'cora':
        dataset = 'co-cora'
    elif args.dname == 'citeseer':
        dataset = 'co-citeseer'
    elif args.dname == 'coauthor_cora':
        dataset = 'coauth_cora'
    elif args.dname == '20newsW100':
        dataset = "news20"
    elif args.dname == 'house-committees-100':
        dataset = "house"
    else:
        dataset = args.dname
        # raise ValueError('dataset not supported')
    
    args.__setattr__('dataset',dataset)
    if dataset not in ['cora','citeseer','coauthor_cora']:
        split_idx = rand_train_test_idx(data.y)
        # print('here')
        train_mask, val_mask, test_mask = split_idx['train'], split_idx['valid'], split_idx['test']
    else:
        _, _, _, train_mask, val_mask, test_mask = get_dataset(args, device='cpu')
    print('% Train: ',sum(train_mask)*100/len(train_mask))

        
    # split_idx_lst = kfold_train_test_idx(data.y, args.runs)
    # # Part 2: Load model
    
    # model = parse_method(args, data, data_p)
    if args.method == 'AllSetTransformer':
        if args.LearnMask:
            model = SetGNN(args, norm_size=batch_size)
        else:
            model = SetGNN(args)
    elif args.method == 'HGNN':
        model = HCHA(args)
    # put things to device
    if args.cuda in [0,1,2,3,4,5,6,7]:
        device = torch.device('cuda:'+str(args.cuda)
                            if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
        
    model, data = model.to(device), data.to(device)
    # if args.method == 'UniGCNII':
    #     args.UniGNN_degV = args.UniGNN_degV.to(device)
    #     args.UniGNN_degE = args.UniGNN_degE.to(device)
    #     args.UniGNN_degV_p = args.UniGNN_degV_p.to(device)
    #     args.UniGNN_degE_p = args.UniGNN_degE_p.to(device)
    
    num_params = count_parameters(model)
    # # Part 3: Main. Training + Evaluation
    split_idx = {'train': train_mask, 'valid': val_mask, 'test': test_mask}
    train_idx = split_idx['train'].to(device)
    
    ### Training loop ###
    model.reset_parameters()
    criterion = nn.NLLLoss()
    eval_func = accuracy
    if args.method == 'UniGCNII':
        optimizer = torch.optim.Adam([
            dict(params=model.reg_params, weight_decay=0.01),
            dict(params=model.non_reg_params, weight_decay=5e-4)
        ], lr=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val_loss = float('inf')
    patience = args.patience 
    patience_counter = 0
    best_model_state = None
    # Z_orig = None 
    data.node_idx = torch.arange(data.x.shape[0],device = device)
    data_loader = DataLoader(TensorDataset(data.node_idx, data.x, data.y), batch_size=batch_size, shuffle=True)
    avg_val_loss = 0
    total_val_accuracy = 0
    total_train_accuracy = 0
    total_test_accuracy = 0
    for epoch in tqdm(range(args.epochs),desc='Epochs:'):
        total_loss = 0
        all_preds = []
        all_labels = []
        outs = []
        outs_val = []
        all_labels_val = []
        #  Training part
        model.train()
        # print('epoch: ',epoch)
        loader_cnt = 0
        for batch in data_loader:
            loader_cnt += 1
            # print('loader_cnt: ',loader_cnt)
            # print('batch: ',batch[0].shape,batch[1].shape)
            batch_data = create_minibatch_for_hcha(data, batch, split_idx)
            # print('% train in this minibatch: ', sum(batch_data.train_mask).item()*100/len(batch_data.train_mask))
            # print(batch_data)
            # print(batch_data.edge_index)
            optimizer.zero_grad()
            # test_flag = False
            # if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
            #     data_input = [batch_data, test_flag]
            # else:
            #     data_input = batch_data
            out = model(batch_data)
            out = F.log_softmax(out, dim=1)
            preds = out.argmax(dim=1)
            loss = criterion(out[batch_data.train_mask], batch_data.y[batch_data.train_mask])
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.item()
            # result = evaluate(args, model, batch_data, eval_func)
            # train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out = result
            # avg_val_loss += valid_loss.item()
            # total_train_accuracy += train_acc
            # total_val_accuracy += valid_acc
            # total_test_accuracy += test_acc
            # Append predictions and true labels
            # print('preds shapes: ',preds.shape, batch_data.y.shape)
            outs.append(out)
            all_preds.append(preds)
            all_labels.append(batch_data.y)
            # print(outs.shape,' ',batch_data.val_mask.shape)
            outs_val.append(out[batch_data.val_mask])
            all_labels_val.append(batch_data.y[batch_data.val_mask])
        model.eval()
        with torch.no_grad():
            Z_orig = torch.cat(outs,dim=0)
        #     # Concatenate all predictions and true labels
        #     all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            outs = torch.cat(outs,dim=0)
            outs_val = torch.cat(outs_val,dim=0)
            all_labels_val = torch.cat(all_labels_val,dim=0)
        #     print('lens: ',all_preds.shape,all_labels.shape)
        #     # Calculate accuracy
        #     total_correct = all_preds.eq(all_labels).double()
        #     total_train_accuracy = total_correct[split_idx['train']].sum().item() / len(all_labels) * 100
        #     total_val_accuracy = total_correct[split_idx['valid']].sum().item() / len(all_labels) * 100  
        #     total_test_accuracy = total_correct[split_idx['test']].sum().item()/ len(all_labels) * 100
            avg_val_loss =  criterion(outs_val, all_labels_val).item()
        #     avg_val_loss = avg_val_loss / len(data_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print('best val loss: ',best_val_loss)
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            # print('patience counter: ',patience_counter)
            if patience_counter > patience:
                print(f'Early stopping at epoch {epoch}.')
                break
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        avg_loss = total_loss / len(data_loader)
        print(f'train loss: {avg_loss:.2f}')
        if epoch % 5 == 0:
            _,avg_train_accuracy, avg_val_accuracy, avg_test_accuracy = inference_on_test_set(model, data, split_idx)
            # avg_test_accuracy = total_test_accuracy 
            # avg_val_accuracy = total_val_accuracy
            # avg_train_accuracy = total_train_accuracy 
            print(
                # f'epoch: {epoch}, '
                f'train loss: {avg_loss:.2f}, '
                f'val loss: {avg_val_loss:.2f}, '
                f'accuracy (train): {avg_train_accuracy:.2f}, '
                f'accuracy (val): {avg_val_accuracy:.2f}, '
                f'accuracy (test): {avg_test_accuracy:.2f}')

        # best_val, best_test = logger.print_statistics()
    _,acc_orig_train,acc_orig_val,acc_orig_test = inference_on_test_set(model, data, split_idx=split_idx)
    # print(len(preds))
    # print(preds)
    print(f"Train Accuracy: {acc_orig_train:.3f}%")
    print(f"Val Accuracy: {acc_orig_val:.3f}%")
    print(f"Test Accuracy: {acc_orig_test:.3f}%")

    # res_root1 = 'log/{}'.format(args.method)#method
    # if not osp.isdir(res_root1):
    #     os.makedirs(res_root1)
    # res_root = '{}/{}'.format(res_root1, args.perturb_type)
    # if not osp.isdir(res_root):
    #     os.makedirs(res_root)

    # filename = f'{res_root}/{args.dname}_noise_{args.feature_noise}.csv'
    # print(f"Saving results to {filename}")
    # with open(filename, 'a+') as write_obj:
    #     cur_line = f'{args.perturb_prop}\n'
    #     cur_line += f'{best_test.mean():.3f}\n'
    #     cur_line += f'{best_test.std():.3f}\n'
    #     write_obj.write(cur_line)

    # all_args_file = f'{res_root}/all_args_{args.dname}_noise_{args.feature_noise}.csv'
    # with open(all_args_file, 'a+') as f:
    #     f.write(str(args))
    #     f.write('\n')

    # print('All done! Exit python code')
    # quit()

    # H = torch.Tensor(ConstructH(data).edge_index).to(device)
    # H = torch.zeros((data.n_x, data.num_hyperedges))
    # src, dst = data.edge_index
    # # # Find node-to-hyperedge edges (i.e., edges where node index < num_nodes and hyperedge index >= num_nodes)
    # node_mask = src < data.n_x
    # node_indices = src[node_mask]
    # # ExtractV2E(data)
    # hyperedge_indices = (data.edge_index[1] - data.n_x + 1).clip(0,data.num_hyperedges-1)  # shift back
    # H[node_indices, hyperedge_indices] = 1.0
    # H = H.to(device)
    
    # row_degrees = H.sum(dim=1)
    # col_degrees = H.sum(dim=0)
    # # print('original dataset (degrees): ',row_degrees.mean().item(),row_degrees.std().item(),row_degrees.min().item(),row_degrees.max().item())
    # # print('original dataset (dim): ',col_degrees.mean().item(),col_degrees.std().item(),col_degrees.min().item(),col_degrees.max().item())

    # row_sums = H.sum(dim=1)
    # has_zero_row = (row_sums == 0).any()
    # print('H degree 0: ',has_zero_row)
    # print('H empty edge: ', (H.sum(dim=0) == 0).any())

    # # print('H: ',H.shape)
    # X = data.x.to(device)
    n = data.n_x 
    e = data.edge_index.shape[1]
    if type(e) == np.int32:
        e = int(e)
    if type(n) == np.int32:
        n = int(n)
    # y = data.y.to(device)

    budget = int(args.ptb_rate * e)
    args.__setattr__('model', args.method)
    print("============ ",args.model, args.dataset,args.attack,str(args.seed),"==================")
    if args.attack == 'mla':
        start_tm = time.time()
        # H_adv, X_adv, robust_model_states = meta_laplacian_pois_attack(root, data, None, model, \
        #                 split_idx, Z_orig, budget=budget, epsilon=args.epsilon, T=args.T, \
        #                 eta_H=args.eta_H, eta_X=args.eta_X, alpha=args.mla_alpha, \
        # )
        H_adv, X_adv, results = meta_laplacian_PGD(args, data, split_idx, train_mask, \
                                    budget = budget, epsilon=args.epsilon, T=args.T, \
                                    eta_H=args.eta_H, eta_X=None)
        exec_time = time.time() - start_tm 
        # save_npz(root, args.seed, results)
        # H_adv = H_adv.detach()
        # X_adv = X_adv.detach()
        # X_adv.requires_grad = False

    # elif args.attack == 'mla_fgsm':
    #     H_adv, X_adv, results, exec_time, robust_model_states = meta_laplacian_FGSM(H, X, y, data, None, None, model, \
    #                     train_mask, val_mask, test_mask, Z_orig, budget=perturbations, epsilon=args.epsilon, T=args.T, \
    #                     eta_H=args.eta_H, eta_X=args.eta_X, alpha=args.mla_alpha, \
    #                     reinit_if_stuck=True)

    # else:
    #     H_adv, X_adv, exec_time = get_attack(model, H, X, y,data, None, train_mask,val_mask,test_mask,perturbations = perturbations)
    
    # root='./'+args.method+'_Melad'
    # os.system('mkdir -p '+root)
    # np.savez(os.path.join(root, args.model+"_"+args.attack+"_"+args.dataset+"_"+str(args.seed)+ '_H_adv.npz'), H_adv.clone().cpu().numpy())
    # np.savez(os.path.join(root, args.model+"_"+args.attack+"_"+args.dataset+"_"+str(args.seed)+ '_X_adv.npz'), X_adv.clone().cpu().numpy())
    # torch.save(data,os.path.join(root,args.model+"_"+args.attack+"_"+args.dataset+"_"+str(args.seed)+ "_data.pth"))

    # print('H_adv:', H_adv)
    # if save and args.attack == 'mla':
    #     plot_results(args,results,root)
    # H_adv_HG = Hypergraph(n, incidence_matrix_to_edge_list(H_adv),device=device)
    data.x = X_adv
    data.edge_index = sparse_H_to_edge_index(H_adv)
    # evasion_dict = evasion_setting_evaluate_hyperMLP(args, H, X, y, data, Z_orig, H_adv, X_adv, None, model, \
    #                          model,train_mask,val_mask,test_mask)
    # --------------- COnvert H_adv to pytorch geometric data format ---------------
    data_clone = data.clone().to(device)
    data_clone.x = X_adv
    # print("H_pert.shape:", H_pert.shape)
    # print("data.num_hyperedges:", data_input.num_hyperedges)
    # row_sums = H_adv.sum(dim=1)
    # has_zero_row = (row_sums == 0).any()
    # print('H_adv degree 0: ',has_zero_row)
    # print('H_adv empty edge: ', (H_adv.sum(dim=0) == 0).any())
    edge_index = sparse_H_to_edge_index(H_adv)
    # data_input.edge_index = H_pert
    # edge_index = generate_G_from_H(data)
    # print("edge_index:", edge_index)
    # print("edge_index.shape:", edge_index.shape)
    data_clone.edge_index = edge_index
    data_clone.n_x = X_adv.shape[0]
    
    data_clone = ExtractV2E(data_clone)
    # print('after extract v2e:',data_input.edge_index.shape)
    # data_clone = Add_Self_Loops(data_clone)
    # print('after add self loops:',data_input.edge_index.shape)
    # data_input.num_hyperedges = data_input.edge_index[0].max()-data_input.n_x+1
    # data_clone.edge_index[1] -= data_clone.edge_index[1].min()
    # data_clone.edge_index = data_clone.edge_index.to(device)
    # data_clone.x = data_clone.x.to(device)
    data_clone = data_clone.to(device)
    # if args.method in ['AllSetTransformer', 'AllDeepSets']:
    #     data_clone = norm_contruction(data_clone, option=args.normtype)

    # test_flag = True
    # if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
    #     data_input = [data_clone, test_flag]
    # else:
    #     data_input = data_clone
    # print('data_input',data_input)
    # print('data: ',data)
    # assert data_input.edge_index.shape[0] == 2
    # assert data_input.edge_index[0].max() < data_input.x.shape[0], "Invalid node index"
    # assert data_input.edge_index[1].max() < data.num_hyperedges, "Invalid hyperedge index"
    # ------------------------ Evasion setting ------------------------
    # Z_adv = model(data_input)
    Z_adv,train_acc,val_acc,test_acc = inference_on_test_set(model, data_clone, split_idx=split_idx)
    print(f"Train Accuracy: {train_acc:.3f}%")
    print(f"Val Accuracy: {val_acc:.3f}%")
    print(f"Test Accuracy: {test_acc:.3f}%")
    evasion_dict = {'exec_time': exec_time,
        'clean_train': acc_orig_train, 'clean_test': acc_orig_test, 'clean_val': acc_orig_val,
        'adv_train': train_acc, 'adv_test': test_acc, 'adv_val': val_acc
        }
    evasion_dict.update(vars(args))
    evasion_dict.update(results)
    evasion_dict['num_edges'] = e
    evasion_dict['num_vertices'] = n
    evasion_dict['num_to_perturb'] = budget
    os.system('mkdir -p large')
    save_to_csv(evasion_dict,filename=os.path.join('large/','evasion_results4.csv'))
    # import sys 
    # sys.exit(1)
    # evasion_dict = compute_statistics(H,H_adv,Z_orig,Z_adv,X,X_adv,train_mask,val_mask,test_mask,y)
    # evasion_dict['exec_time'] = exec_time
    # if type(e) == np.int32:
    #     e = int(e)
    # if type(n) == np.int32:
    #     n = int(n)
    # evasion_dict['num_edges'] = e
    # evasion_dict['num_vertices'] = n
    # evasion_dict['num_edges_perturbed'] = budget
    # # print(evasion_dict)
    # print(json.dumps(evasion_dict,indent = 4))
    # # # print('H_adv - H:',torch.sum((H_adv-H).abs()))
    # # ----------------------- Poisoning setting -----------------------
    # print('---------------- After attack ----------------')
    # logger = Logger(args.runs, args)
    # criterion = nn.NLLLoss()
    # eval_func = eval_acc
    # if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
    #     model.reset_parameters()
    # else:
    #     model = parse_method(args, data_clone, data_p).to(device)
    # model.train()
    # for run in range(args.runs):
    #     start_time = time.time()
    #     # split_idx = split_idx_lst[run]
    #     split_idx = {'train': train_mask, 'valid': val_mask, 'test': test_mask}
    #     train_idx = split_idx['train'].to(device)
    #     setup_seed(args.seed)
    #     model.reset_parameters()
    #     if args.method == 'UniGCNII':
    #         optimizer = torch.optim.Adam([
    #             dict(params=model.reg_params, weight_decay=0.01),
    #             dict(params=model.non_reg_params, weight_decay=5e-4)
    #         ], lr=0.01)
    #     else:
    #         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # #     This is for HNHN only
    # #     if args.method == 'HNHN':
    # #         scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100, gamma=0.51)
    #     best_val_loss = float('inf')
    #     patience = args.patience 
    #     patience_counter = 0
    #     best_model_state = None
    #     Z_adv = None 
    #     for epoch in tqdm(range(args.epochs)):
    #         #         Training part
    #         model.train()
    #         optimizer.zero_grad()
    #         test_flag = False
    #         if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
    #             data_input_adv = [data_clone.to(device), test_flag]
    #         else:
    #             data_input_adv = data_clone
                
    #         out = model(data_input_adv)
    #         out = F.log_softmax(out, dim=1)
    #         loss = criterion(out[train_idx], data_clone.y[train_idx])
    #         loss.backward(retain_graph=True)
    #         optimizer.step()
    # #         if args.method == 'HNHN':
    # #             scheduler.step()
    # #         Evaluation part
    #         result = evaluate(args, model, data_clone, split_idx, eval_func)
    #         train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out = result
    #         logger.add_result(run, result[:3])
    #         if valid_loss.item() < best_val_loss:
    #             best_val_loss = valid_loss.item()
    #             best_model_state = model.state_dict()
    #             patience_counter = 0
    #             Z_adv = out.detach()
    #         else:
    #             if epoch == 0:
    #                 Z_adv = out.detach()
    #             patience_counter += 1
    #             if patience_counter > patience:
    #                 print(f'Early stopping at epoch {epoch}.')
    #                 break
    #         if epoch % 1 == 0 and args.display_step >= 0:
    #             print(f'Epoch: {epoch:02d}, '
    #                   f'Train Loss: {loss:.4f}, '
    #                   f'Valid Loss: {result[4]:.4f}, '
    #                   f'Test  Loss: {result[5]:.4f}, '
    #                   f'Train Acc: {100 * result[0]:.2f}%, '
    #                   f'Valid Acc: {100 * result[1]:.2f}%, '
    #                   f'Test  Acc: {100 * result[2]:.2f}%')
    #     if best_model_state is not None:
    #         model.load_state_dict(best_model_state)

    # # logger.print_statistics(run)
    # ### Save results ###
    # # best_val, best_test = logger.print_statistics()
    # results = compute_statistics(H, H_adv, Z_orig, Z_adv, X, X_adv, train_mask, val_mask, test_mask, y)
    # results['exec_time'] = exec_time
    # results['num_edges'] = e
    # results['num_vertices'] = n
    # results['num_edges_perturbed'] = perturbations
    # degree_adv = H_adv.sum(dim=1)
    # # np.savez(os.path.join(root, args.model+"_"+args.attack+"_"+args.dataset+"_"+str(args.seed)+ '_deg_H_adv.npz'), degree_adv.clone().cpu().numpy())

    # print('================ Poisoning setting =================')
    # verbose = False
    # if verbose:
    #     print("Laplacian Frobenius norm change:", results['laplacian_norm'])
    #     print("Embedding shift (ΔZ Fro norm):", results['embedding_shift'])
    #     print("Structural L0 perturbation:", results['h_l0'])
    #     print("Feature L-infinity perturbation:", results["x_linf"])
    #     print("Total shift in degree distribution (Linf):", results["deg_shift_linf"])
    #     print("Total shift in degree distribution (L1):", results["deg_shift_l1"])
    #     print("Total shift in degree distribution (L2):",results["deg_shift_l2"])
    #     print("Total shift in edge-cardinality distribution (Linf):", results["edge_card_shift_linf"])
    #     print("Total shift in edge-cardinality distribution (L1):", results["edge_card_shift_l1"])
    #     print("Total shift in edge-cardinality distribution (L2):", results["edge_card_shift_l2"])
    #     print("Semantic change in features (1 - avg. cosine):", results['semantic_change'])
    #     print("Embedding sensitivity vs node degree (Pearson r):", results["degree_sensitivity"])
    #     print("Classification accuracy before attack: %.3f %.3f %.3f" %(results['clean_train'],results['clean_val'],results['clean_test']))
    #     print("Classification accuracy after attack: %.3f %.3f %.3f" %(results['adv_train'],results['adv_val'],results['adv_test']))
    #     print("Accuracy drop due to attack: %.2f%%" %results['acc_drop%'])
    #     print('Actual |H_adv - H|_0:', (H_adv - H).abs().sum().item(),' ptb: ',perturbations)
    # print(json.dumps(results,indent=4))
    # if save:
    #     results.update(vars(args))
    #     evasion_dict.update(vars(args))
        
    #     save_to_csv(evasion_dict,filename=os.path.join(root,'evasion_results.csv'))
    #     save_to_csv(results,filename=os.path.join(root,'pois_results.csv'))
    