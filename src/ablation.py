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
from mla_utils import *
from modelzoo import SimpleHGNN,GradArgmax
import pandas as pd

def topk_budget_flip(H, delta_H, budget):
    H_adv = H.clone()
    scores = delta_H.abs().flatten()
    k = min(budget, scores.numel())
    idx = torch.topk(scores, k=k, largest=True).indices
    flat = H_adv.flatten()
    dflat = delta_H.flatten()
    # Flip toward the sign of delta_H
    add_mask = dflat[idx] > 0   # set to 1
    rem_mask = ~add_mask        # set to 0
    flat[idx[add_mask]] = 1
    flat[idx[rem_mask]] = 0
    return H_adv.view_as(H)

# MeLA-FGSM
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

# MeLA-D
def meta_laplacian_pois_attack(args, root, H, X, y, data, HG, surrogate_class, target_model, train_mask, val_mask, test_mask, logits_orig, budget=20, epsilon=0.05, T=20, eta_H=1e-2, eta_X=1e-2, alpha=4.0, reinit_if_stuck=True):
    """
      Meta Laplacian Attack adapted to poisoning setting (training-time). 
    - The attacker perturbs H and X before training.
    - A new model is trained from scratch at every iteration to simulte a bilevel optimization. 
    Params:
        H,X,y: Original incidence matrix, features and labels
        surrogate_class: Constructor of the surrogate model (e.g. SimpleHGNN)
    """
    verbose = False
    device = X.device
    # idx_unlabeled = val_mask | test_mask
    H = H.clone().detach()
    X = X.clone().detach()
    H.requires_grad = False
    X.requires_grad = False
    n, m = H.shape
    # Z = model(H, X).detach()
    delta_H = (1e-3 * torch.randn_like(H)).requires_grad_()
    delta_X = (1e-3 * torch.randn_like(X)).requires_grad_()
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
    for t in tqdm(range(T)):
        runtime_start = time.time()
        if surrogate_class is None:
            surrogate_model = SimpleHGNN(X.shape[1], hidden_dim = args.MLP_hidden, out_dim = args.num_classes, device = X.device).to(device)
            optimizer = torch.optim.Adam(surrogate_model.parameters(),lr=args.lr)
        else:
            # surrogate_model = surrogate_class(X.shape[1],)
            raise Exception("Other surrogates Not implemented")
        H_pert = torch.clamp(H + delta_H, 0, 1)
        X_pert = X + delta_X
        L_pert = lap(H_pert)
        # for epoch in tqdm(range(args.num_epochs),desc = 'Training surrogate: iter = '+str(t)):
        criterion = nn.CrossEntropyLoss()
        best_val_accuracy = -float('inf')
        best_model_state = None
        if args.dataset == '20newsW100':
            surrogate_epochs = args.num_epochs_sur
        else:
            if t == T-1:
                surrogate_epochs = args.epochs 
            else:
                surrogate_epochs = args.num_epochs_sur
        data.x = X_pert
        data.edge_index = incidence_to_edge_index(topk_budget_flip(H,delta_H,budget))
        surrogate_model.train()
        for epoch in range(surrogate_epochs):
            optimizer.zero_grad()
            logits = surrogate_model(X_pert, H_pert)
            # print(logits[train_mask].shape,y[train_mask].shape,logits.shape,y.shape)
            loss = criterion(logits[train_mask],y[train_mask])
            loss.backward(retain_graph=True)
            optimizer.step()
            if t == T-1:
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
                with torch.no_grad():
                    target_model.eval()
                    # --------
                    data_input = data.clone().to(device)
                    data_input.x = X_pert
                    edge_index = incidence_to_edge_index2(H_pert)
                    data_input.edge_index = edge_index
                    data_input.n_x = X_pert.shape[0]
                    data_input = ExtractV2E(data_input)
                    data_input = Add_Self_Loops(data_input)
                    # _, _, acc_drop = classification_drop(args,target_model, H, HG, X, H_pert, X_pert, y)
                    data_input.edge_index[1] -= data_input.edge_index[1].min()
                    data_input.edge_index = data_input.edge_index.to(device)
                    if args.method in ['AllSetTransformer', 'AllDeepSets']:
                        data_input = norm_contruction(data_input, option=args.normtype)

                    test_flag = True
                    if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
                        data_input = [data_input, test_flag]
                    # ------
                    logits_adv = target_model(data_input)
                acc_orig = (logits_orig.argmax(dim=1)[test_mask] == y[test_mask]).float().mean().item()
                acc_adv = (logits_adv.argmax(dim=1)[test_mask] == y[test_mask]).float().mean().item()
                acc_drop = (acc_orig - acc_adv)/acc_orig
                _, _, acc_drop_sur = classification_drop(args,surrogate_model, H, None, X, H_pert, X_pert, y)
                print("Surr Loss : ",loss.item()," Accuracy drop (surrogate): ", acc_drop_sur*100,'%', " Accuracy drop (target): ", acc_drop*100,'%')
        time1 = time.time() - runtime_start
        if t == T-1:
            surrogate_model.load_state_dict(best_model_state) # Take the best model
        with torch.no_grad():
            target_model.eval()
            surrogate_model.eval()
            surrogate_test_accuracy = accuracy(surrogate_model(X_pert,H_pert)[test_mask], y[test_mask]) 
            # ---
            data_input = data.clone().to(device)
            data_input.x = X_pert
            edge_index = incidence_to_edge_index2(H_pert)
            data_input.edge_index = edge_index
            data_input.n_x = X_pert.shape[0]
            data_input = ExtractV2E(data_input)
            data_input = Add_Self_Loops(data_input)
            # _, _, acc_drop = classification_drop(args,target_model, H, HG, X, H_pert, X_pert, y)
            data_input.edge_index[1] -= data_input.edge_index[1].min()
            data_input.edge_index = data_input.edge_index.to(device)
            if args.method in ['AllSetTransformer', 'AllDeepSets']:
                data_input = norm_contruction(data_input, option=args.normtype)

            test_flag = True
            if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
                data_input = [data_input, test_flag]
            target_Z = target_model(data_input)
            # ---
            target_model_test_accuracy = accuracy(target_Z[test_mask], y[test_mask])
            surrogate_test_trajectory.append(surrogate_test_accuracy.item())
            target_test_trajectory.append(target_model_test_accuracy.item())
        if t == T-1:
            os.makedirs(os.path.join(root,str(args.seed)), exist_ok=True)
            prefix = os.path.join(root,str(args.seed), 'SimpleHGNN_'+args.dataset+'_'+args.model+'_'+str(args.ptb_rate))
            torch.save(best_model_state, prefix+'_weights.pth')
        runtime_start2 = time.time()
        Z = surrogate_model(X_pert, H_pert) # Trained surrogate model
        delta_L = (L_pert - L_orig) @ Z
        # loss_meta = (delta_L**2).sum()
        H_temp = torch.clamp(H + delta_H, 0, 1)
        dv_temp = H_temp @ torch.ones((H.shape[1],), device=device)
        degree_violation = (dv_temp - dv_orig)
        degree_penalty = torch.sum(degree_violation ** 2) / n
        # degree_penalty = torch.abs(degree_violation).mean()
        deg_penalty_val = degree_penalty.item()
        # loss_meta += degree_penalty

        # logits_adv = target_model(X_pert,H_pert)
        logits_adv = Z
        loss_cls = F.cross_entropy(logits_adv, y)
        # loss_cls = F.cross_entropy(logits_adv, model(H, X).argmax(dim=1))
        # lap_dist = (delta_L**2).sum()
        lap_dist = torch.norm(delta_L, p=2).mean()
        # print(delta_L.shape)
        # lap_dist = torch.mean(delta_L**2)
        cls_loss_val = loss_cls.item()
        lap_dist_val = lap_dist.item() if isinstance(lap_dist, torch.Tensor) else lap_dist
        loss_meta = args.alpha * lap_dist - args.beta*degree_penalty + args.gamma * loss_cls

        grads = torch.autograd.grad(loss_meta,[delta_H,delta_X])

        lap_dist_trajectory.append(lap_dist_val)
        loss_meta_trajectory.append(loss_meta.item())
        # _, _, acc_drop = classification_drop(args,target_model, H, HG, X, H_pert, X_pert, y)
        with torch.no_grad():
            target_model.eval()
            # _, _, acc_drop = classification_drop(args,target_model, H, HG, X, H_pert, X_pert, y)
            # -------
            data_input = data.clone().to(device)
            data_input.x = X_pert
            edge_index = incidence_to_edge_index2(H_pert)
            data_input.edge_index = edge_index
            data_input.n_x = X_pert.shape[0]
            data_input = ExtractV2E(data_input)
            data_input = Add_Self_Loops(data_input)
            # _, _, acc_drop = classification_drop(args,target_model, H, HG, X, H_pert, X_pert, y)
            data_input.edge_index[1] -= data_input.edge_index[1].min()
            data_input.edge_index = data_input.edge_index.to(device)
            if args.method in ['AllSetTransformer', 'AllDeepSets']:
                data_input = norm_contruction(data_input, option=args.normtype)
            test_flag = True
            if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
                data_input = [data_input, test_flag]
            # ------
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
        with torch.no_grad():
            # Proceed with original gradient ascent
            delta_H += eta_H * grads[0].sign()
            delta_X += eta_X * grads[1].sign()
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
            delta_H_new[bad_nodes, :] = 0
            delta_H_new[:, bad_edges] = 0
            delta_H.copy_(delta_H_new)

        delta_X = delta_X.clamp(-epsilon, epsilon)
        time2 = time.time() - runtime_start2
    # results = [(t, loss_meta, acc_drop, lap_shift, deg_penalty, cls_loss, lap_dist, feature_shift)]
    results = [loss_meta_trajectory, acc_drop_trajectory, lap_shift_trajectory, lap_dist_trajectory, cls_loss_trajectory, \
               deg_penalty_trajectory,feature_shift_trajectory,surrogate_test_trajectory, target_test_trajectory]
    # mask = filter_potential_singletons(torch.clamp(H + delta_H, 0, 1))
    H_adv = topk_budget_flip(H, delta_H, budget)
    return H_adv, X + delta_X, results, time1+time2, best_model_state

def get_attack(target_model,H,X,y,data,HG,train_mask,val_mask,test_mask,perturbations):
    if args.attack == 'gradargmax':
        if args.method != 'simplehgnn':
            target_model = SimpleHGNN(X.shape[1],hidden_dim = args.MLP_hidden, out_dim = args.num_classes,device = X.device)
        # print('surrogate : ',target_model)
        attack_model = GradArgmax(model=target_model.to(device), nnodes=X.shape[0], nnedges = H.shape[1], \
                                attack_structure=True, device=device)
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
def evaluate(args, model, data, split_idx, eval_func, result=None):
    test_flag = True
    if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
        data_input = [data, test_flag]
    else:
        data_input = data
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(data_input)
        out = F.log_softmax(out, dim=1)
    train_acc = eval_func(
        data.y[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        data.y[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        data.y[split_idx['test']], out[split_idx['test']])

#     Also keep track of losses
    train_loss = F.nll_loss(
        out[split_idx['train']], data.y[split_idx['train']])
    valid_loss = F.nll_loss(
        out[split_idx['valid']], data.y[split_idx['valid']])
    test_loss = F.nll_loss(
        out[split_idx['test']], data.y[split_idx['test']])
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--dname', default='walmart-trips-100')
    # method in ['SetGNN','CEGCN','CEGAT','HyperGCN','HGNN','HCHA']
    parser.add_argument('--method', default='AllSetTransformer',type=str)
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
                    choices=['mla','Rand-flip', 'Rand-feat','gradargmax','mla_fgsm'], help='attack variant')
    parser.add_argument('--epsilon', type=float, default=0.05, help='Node Feature perturbation bound')
    parser.add_argument('--ptb_rate', type=float, default=0.2,  help='pertubation rate')
    parser.add_argument('--patience', type=int, default=150,
                    help='Patience for training with early stopping.')
    parser.add_argument('--T', type=int, default=50, help='Number of iterations for the attack.')
    parser.add_argument('--eta_H', type=float, default=1e-2, help='Learning rate for H perturbation')
    parser.add_argument('--eta_X', type=float, default=1e-2, help='Learning rate for X perturbation')
    parser.add_argument('--num_epochs_sur', type=int, default=50, help='#epochs for the surrogate training.')
    parser.add_argument('--beta', type=float, default= 1.0, help='weight for degree penalty loss component')
    parser.add_argument('--gamma', type=float, default=2.0, help='weight for classification loss component')
    parser.add_argument('--alpha', type=float, default=1.0, help='weight for laplacian Loss component')
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
    root='./ablation' # Stores the results and various statistics
    # root='./'+args.attack+'_hypergraphMLP_final2'
    os.makedirs(root, exist_ok=True)
    save = True
    if args.cuda in [0,1,2,3,4,5,6,7]:
        device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
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
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        if args.dname in ['yelp', 'walmart-trips', 'house-committees', 'walmart-trips-100', 'house-committees-100']:
            #         Shift the y label to start with 0
            args.num_classes = len(data.y.unique())
            data.y = data.y - data.y.min()
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
    if args.method in ['AllSetTransformer', 'AllDeepSets']:
        data = ExtractV2E(data)
        H = torch.Tensor(ConstructH(data)).to(device)
        data_p = perturb_hyperedges(data, args.perturb_prop, args.perturb_type)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
            data_p = Add_Self_Loops(data_p)
        if args.exclude_self:
            data = expand_edge_index(data)
            data_p = expand_edge_index(data_p)
    
        #     Compute deg normalization: option in ['all_one','deg_half_sym'] (use args.normtype)
        # data.norm = torch.ones_like(data.edge_index[0])
        data = norm_contruction(data, option=args.normtype)
        data_p = norm_contruction(data_p, option=args.normtype)
        
    elif args.method in ['CEGCN', 'CEGAT']:
        data = ExtractV2E(data)
        data_p = perturb_hyperedges(data, args.perturb_prop, args.perturb_type)
        data = ConstructV2V(data)
        data = norm_contruction(data, TYPE='V2V')
        data_p = ConstructV2V(data_p)
        data_p = norm_contruction(data_p, TYPE='V2V')
    
    elif args.method in ['HyperGCN']:
        data = ExtractV2E(data)
        H = torch.Tensor(ConstructH(data)).to(device)
        data_p = perturb_hyperedges(data, args.perturb_prop, args.perturb_type)
    #     ipdb.set_trace()
    #   Feature normalization, default option in HyperGCN
        # X = data.x
        # X = sp.csr_matrix(utils.normalise(np.array(X)), dtype=np.float32)
        # X = torch.FloatTensor(np.array(X.todense()))
        # data.x = X
    
    # elif args.method in ['HGNN']:
    #     data = ExtractV2E(data)
    #     if args.add_self_loop:
    #         data = Add_Self_Loops(data)
    #     data = ConstructH(data)
    #     data = generate_G_from_H(data)
    
    elif args.method in ['HNHN']:
        data = ExtractV2E(data)
        data_p = perturb_hyperedges(data, args.perturb_prop, args.perturb_type)
        if args.add_self_loop:
            data_p = Add_Self_Loops(data_p)
            data = Add_Self_Loops(data)
            
        H = ConstructH_HNHN(data)
        data = generate_norm_HNHN(H, data, args)
        data.edge_index[1] -= data.edge_index[1].min()
        
        H_p = ConstructH_HNHN(data_p)
        data_p = generate_norm_HNHN(H_p, data_p, args)
        data_p.edge_index[1] -= data_p.edge_index[1].min()
    
    elif args.method in ['HCHA', 'HGNN']:
        # print('if: ',data)
        # print(data.edge_index[1].min(),data.edge_index[1].max())
        # print(data.edge_index[0].min(),data.edge_index[0].max())
        data = ExtractV2E(data)
        H = torch.Tensor(ConstructH(data)).to(device)
        # print('H.shape: ',H.shape)
        # print(H)
        # print('after extract: ',data)
        data_p = perturb_hyperedges(data, args.perturb_prop, args.perturb_type)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
            # print('after add self loop: ',data)
            data_p = Add_Self_Loops(data_p)
    #    Make the first he_id to be 0
        data_p.edge_index[1] -= data_p.edge_index[1].min()
        # print('data.edge_index[1].min(): ',data.edge_index[1].min())
        data.edge_index[1] -= data.edge_index[1].min()
        # print('after min: ',data)
    elif args.method in ['UniGCNII']:
        data = ExtractV2E(data)
        data_p = perturb_hyperedges(data, args.perturb_prop, args.perturb_type)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
            data_p = Add_Self_Loops(data_p)
        V, E = unignn_ini_ve(data, args)
        V, E = V.to(device), E.to(device)
        
        V_p, E_p = unignn_ini_ve(data_p, args)
        V_p, E_p = V_p.to(device), E_p.to(device)

        args.UniGNN_degV, args.UniGNN_degE = unignn_get_deg(V, E)
        args.UniGNN_degV_p, args.UniGNN_degE_p = unignn_get_deg(V_p, E_p)
    
        V, E = V.cpu(), E.cpu()
        del V
        del E
        V_p, E_p = V_p.cpu(), E_p.cpu()
        del V_p
        del E_p
    
    #     Get splits
    # split_idx_lst = []
    # for run in range(args.runs):
    #     split_idx = rand_train_test_idx(
    #         data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
    #     split_idx_lst.append(split_idx)
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
    split_idx = rand_train_test_idx(data.y,train_prop=args.train_prop, valid_prop=args.valid_prop)
    train_mask, val_mask, test_mask = split_idx['train'], split_idx['valid'], split_idx['test']
    print('% Train: ',sum(train_mask)*100/len(train_mask))

    
    # split_idx_lst = kfold_train_test_idx(data.y, args.runs)
    # # Part 2: Load model
    
    model = parse_method(args, data, data_p)
    # put things to device
    
    model, data, data_p = model.to(device), data.to(device), data_p.to(device)
    if args.method == 'UniGCNII':
        args.UniGNN_degV = args.UniGNN_degV.to(device)
        args.UniGNN_degE = args.UniGNN_degE.to(device)
        args.UniGNN_degV_p = args.UniGNN_degV_p.to(device)
        args.UniGNN_degE_p = args.UniGNN_degE_p.to(device)
    
    num_params = count_parameters(model)
    # # Part 3: Main. Training + Evaluation
    
    
    logger = Logger(args.runs, args)
    
    criterion = nn.NLLLoss()
    eval_func = eval_acc
    
    model.train()
    # print('MODEL:', model)
    
    ### Training loop ###
    runtime_list = []
    for run in range(args.runs):
        start_time = time.time()
        # split_idx = split_idx_lst[run]
        split_idx = {'train': train_mask, 'valid': val_mask, 'test': test_mask}
        train_idx = split_idx['train'].to(device)
        setup_seed(args.seed)
        model.reset_parameters()
        if args.method == 'UniGCNII':
            optimizer = torch.optim.Adam([
                dict(params=model.reg_params, weight_decay=0.01),
                dict(params=model.non_reg_params, weight_decay=5e-4)
            ], lr=0.01)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    #     This is for HNHN only
    #     if args.method == 'HNHN':
    #         scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100, gamma=0.51)
        best_val = float('-inf')
        best_val_loss = float('inf')
        patience = args.patience 
        patience_counter = 0
        best_model_state = None
        Z_orig = None 
        for epoch in tqdm(range(args.epochs)):
            #         Training part
            model.train()
            optimizer.zero_grad()
            test_flag = False
            if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
                data_input = [data, test_flag]
            else:
                data_input = data
            # print(epoch,' ',data_input)
            out = model(data_input)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out[train_idx], data.y[train_idx])
            loss.backward(retain_graph=True)
            optimizer.step()
    #         if args.method == 'HNHN':
    #             scheduler.step()
    #         Evaluation part
            result = evaluate(args, model, data, split_idx, eval_func)
            train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out = result
            logger.add_result(run, result[:3])
            if valid_loss.item() < best_val_loss:
                best_val_loss = valid_loss.item()
                best_model_state = model.state_dict()
                patience_counter = 0
                Z_orig = out.detach()
            else:
                patience_counter += 1
                if patience_counter > patience:
                    print(f'Early stopping at epoch {epoch}.')
                    break
            if epoch % args.display_step == 0 and args.display_step > 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Train Loss: {loss:.4f}, '
                      f'Valid Loss: {result[4]:.4f}, '
                      f'Test  Loss: {result[5]:.4f}, '
                      f'Train Acc: {100 * result[0]:.2f}%, '
                      f'Valid Acc: {100 * result[1]:.2f}%, '
                      f'Test  Acc: {100 * result[2]:.2f}%')
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        end_time = time.time()
        runtime_list.append(end_time - start_time)
    
        # logger.print_statistics(run)
    
    ### Save results ###
    avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)


    X = data.x.to(device)
    n = data.n_x 
    e = data.edge_index.shape[1]
    y = data.y.to(device)

    perturbations = int(args.ptb_rate * e)
    args.__setattr__('model', args.method)
    print("============ ",args.model, args.dataset,args.attack,str(args.seed),"==================")
    if args.attack == 'mla':
        H_adv, X_adv, results, exec_time, robust_model_states = meta_laplacian_pois_attack(args,root, H, X, y, data, None, None, model, \
                        train_mask, val_mask, test_mask, Z_orig, budget=perturbations, epsilon=args.epsilon, T=args.T, \
                        eta_H=args.eta_H, eta_X=args.eta_X, alpha=args.alpha, \
                        reinit_if_stuck=True)
        save_npz(root, args.seed, results)
        H_adv = H_adv.detach()
        X_adv = X_adv.detach()
        X_adv.requires_grad = False

    elif args.attack == 'mla_fgsm':
        H_adv, X_adv, results, exec_time, robust_model_states = meta_laplacian_FGSM(H, X, y, data, None, None, model, \
                        train_mask, val_mask, test_mask, Z_orig, budget=perturbations, epsilon=args.epsilon, T=args.T, \
                        eta_H=args.eta_H, eta_X=args.eta_X, alpha=args.alpha, \
                        reinit_if_stuck=True)

    else:
        H_adv, X_adv, exec_time = get_attack(model, H, X, y,data, None, train_mask,val_mask,test_mask,perturbations = perturbations)
    
    # print('H_adv:', H_adv)
    if save and args.attack == 'mla':
        plot_results(args,results,root)
    # H_adv_HG = Hypergraph(n, incidence_matrix_to_edge_list(H_adv),device=device)
    # data.x = X_adv
    # data.edge_index = incidence_to_edge_index(H_adv)
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
    # data_clone.edge_index = incidence_to_edge_index2(H_adv)
    # print('H_adv:', H_adv)
    data_clone.edge_index = incidence_to_edge_index2(H_adv)

    # data_input.edge_index = H_pert
    # edge_index = generate_G_from_H(data)
    # print("edge_index:", edge_index)
    # print("edge_index.shape:", edge_index.shape)
    # data_clone.edge_index = edge_index
    data_clone.n_x = X_adv.shape[0]
    
    data_clone = ExtractV2E(data_clone)
    # print('after extract v2e:',data_input.edge_index.shape)
    data_clone = Add_Self_Loops(data_clone)
    # print('after add self loops:',data_input.edge_index.shape)
    # data_input.num_hyperedges = data_input.edge_index[0].max()-data_input.n_x+1
    data_clone.edge_index[1] -= data_clone.edge_index[1].min()
    data_clone.edge_index = data_clone.edge_index.to(device)
    data_clone.x = data_clone.x.to(device)
    # print(data_clone.edge_index[0].max(),data_clone.edge_index[0].min(),data_clone.edge_index[1].max(),data_clone.edge_index[1].min())
    if args.method in ['AllSetTransformer', 'AllDeepSets']:
        data_clone = norm_contruction(data_clone, option=args.normtype)

    test_flag = True
    if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
        data_input = [data_clone, test_flag]
    else:
        data_input = data_clone

    # ----------------------- Poisoning setting -----------------------
    print('---------------- After attack ----------------')
    logger = Logger(args.runs, args)
    criterion = nn.NLLLoss()
    eval_func = eval_acc
    if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
        model.reset_parameters()
    else:
        model = parse_method(args, data_clone, data_p).to(device)
    model.train()
    for run in range(args.runs):
        start_time = time.time()
        # split_idx = split_idx_lst[run]
        split_idx = {'train': train_mask, 'valid': val_mask, 'test': test_mask}
        train_idx = split_idx['train'].to(device)
        setup_seed(args.seed)
        model.reset_parameters()
        if args.method == 'UniGCNII':
            optimizer = torch.optim.Adam([
                dict(params=model.reg_params, weight_decay=0.01),
                dict(params=model.non_reg_params, weight_decay=5e-4)
            ], lr=0.01)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    #     This is for HNHN only
    #     if args.method == 'HNHN':
    #         scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100, gamma=0.51)
        best_val = float('-inf')
        best_val_loss = float('inf')
        patience = args.patience 
        patience_counter = 0
        best_model_state = None
        Z_adv = None 
        for epoch in tqdm(range(args.epochs)):
            #         Training part
            model.train()
            optimizer.zero_grad()
            test_flag = False
            if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
                data_input_adv = [data_clone.to(device), test_flag]
            else:
                data_input_adv = data_clone
                
            out = model(data_input_adv)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out[train_idx], data_clone.y[train_idx])
            loss.backward(retain_graph=True)
            optimizer.step()
    #         if args.method == 'HNHN':
    #             scheduler.step()
    #         Evaluation part
            result = evaluate(args, model, data_clone, split_idx, eval_func)
            train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out = result
            logger.add_result(run, result[:3])
            if valid_loss.item() < best_val_loss:
                best_val_loss = valid_loss.item()
                best_model_state = model.state_dict()
                patience_counter = 0
                Z_adv = out.detach()
            else:
                if epoch == 0:
                    Z_adv = out.detach()
                patience_counter += 1
                if patience_counter > patience:
                    print(f'Early stopping at epoch {epoch}.')
                    break
            if epoch % 1 == 0 and args.display_step >= 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Train Loss: {loss:.4f}, '
                      f'Valid Loss: {result[4]:.4f}, '
                      f'Test  Loss: {result[5]:.4f}, '
                      f'Train Acc: {100 * result[0]:.2f}%, '
                      f'Valid Acc: {100 * result[1]:.2f}%, '
                      f'Test  Acc: {100 * result[2]:.2f}%')
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

    # logger.print_statistics(run)
    ### Save results ###
    best_val, best_test = logger.print_statistics()
    results = compute_statistics(H, H_adv, Z_orig, Z_adv, X, X_adv, train_mask, val_mask, test_mask, y)
    results['exec_time'] = exec_time
    results['num_edges'] = e
    results['num_vertices'] = n
    results['num_to_perturb'] = perturbations
    results['num_edges_perturbed'] = (H_adv - H).abs().sum().item()
    degree_adv = H_adv.sum(dim=1)
    # np.savez(os.path.join(root, args.model+"_"+args.attack+"_"+args.dataset+"_"+str(args.seed)+ '_deg_H_adv.npz'), degree_adv.clone().cpu().numpy())

    print('================ Poisoning setting =================')
    verbose = False
    if verbose:
        print("Laplacian Frobenius norm change:", results['laplacian_norm'])
        print("Embedding shift (ΔZ Fro norm):", results['embedding_shift'])
        print("Structural L0 perturbation:", results['h_l0'])
        print("Feature L-infinity perturbation:", results["x_linf"])
        print("Total shift in degree distribution (Linf):", results["deg_shift_linf"])
        print("Total shift in degree distribution (L1):", results["deg_shift_l1"])
        print("Total shift in degree distribution (L2):",results["deg_shift_l2"])
        print("Total shift in edge-cardinality distribution (Linf):", results["edge_card_shift_linf"])
        print("Total shift in edge-cardinality distribution (L1):", results["edge_card_shift_l1"])
        print("Total shift in edge-cardinality distribution (L2):", results["edge_card_shift_l2"])
        print("Semantic change in features (1 - avg. cosine):", results['semantic_change'])
        print("Embedding sensitivity vs node degree (Pearson r):", results["degree_sensitivity"])
        print("Classification accuracy before attack: %.3f %.3f %.3f" %(results['clean_train'],results['clean_val'],results['clean_test']))
        print("Classification accuracy after attack: %.3f %.3f %.3f" %(results['adv_train'],results['adv_val'],results['adv_test']))
        print("Accuracy drop due to attack: %.2f%%" %results['acc_drop%'])
        print('Actual |H_adv - H|_0:', (H_adv - H).abs().sum().item(),' ptb: ',perturbations)
    print(json.dumps(results,indent=4))
    if save:
        results.update(vars(args))
        # evasion_dict.update(vars(args))
        
        # save_to_csv(evasion_dict,filename=os.path.join(root,'evasion_results.csv'))
        save_to_csv(results,filename=os.path.join(root,'pois_results.csv'))
    