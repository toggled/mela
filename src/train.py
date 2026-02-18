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
from torch.nn.utils.stateless import functional_call

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

# def meta_laplacian_FGSM(H, X, y, data, HG, surrogate_class, target_model, train_mask, val_mask, test_mask, logits_orig, budget=20, epsilon=0.05, T=20, eta_H=1e-2, eta_X=1e-2, alpha=1.0, beta=1.0, gamma=2.0, reinit_if_stuck=True):
#     """
#       Meta Laplacian Attack adapted to poisoning setting (training-time). 
#     - The attacker perturbs H and X before training.
#     - A new model is trained from scratch at every iteration to simulte a bilevel optimization. 
#     Params:
#         H,X,y: Original incidence matrix, features and labels
#         surrogate_class: Constructor of the surrogate model (e.g. SimpleHGNN)
#     """
#     verbose = False
#     eta_H = args.ptb_rate 
#     eta_X = args.epsilon
#     device = X.device
#     # idx_unlabeled = val_mask | test_mask
#     H = H.clone().detach()
#     X = X.clone().detach()
#     H.requires_grad = False
#     X.requires_grad = False
#     n, m = H.shape
#     # Z = model(H, X).detach()
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
#     # Surrogate model training
#     if surrogate_class is None:
#         surrogate_model = SimpleHGNN(X.shape[1], hidden_dim = args.MLP_hidden, out_dim = args.num_classes, device = X.device).to(device)
#         optimizer = torch.optim.Adam(surrogate_model.parameters(),lr=args.lr)
#     else:
#         # surrogate_model = surrogate_class(X.shape[1],)
#         raise Exception("Other surrogates Not implemented")
#     criterion = nn.CrossEntropyLoss()
#     best_val_accuracy = -float('inf')
#     best_model_state = None
#     surrogate_epochs = args.num_epochs_sur
#     for epoch in range(surrogate_epochs):
#         surrogate_model.train()
#         optimizer.zero_grad()
#         logits = surrogate_model(X, H)
#         # print(logits[train_mask].shape,y[train_mask].shape,logits.shape,y.shape)
#         loss = criterion(logits[train_mask],y[train_mask])
#         loss.backward(retain_graph=True)
#         optimizer.step()
#         surrogate_model.eval()
#         # Save the surrogate model (which has the best validation accuracy) for robust training
#         with torch.no_grad():   
#             # val_loss = criterion(logits[val_mask], y[val_mask])
#             val_accuracy = accuracy(logits[val_mask],y[val_mask])
#         if val_accuracy.item() > best_val_accuracy:
#             best_val_accuracy = val_accuracy.item()
#             # print('Best val accuracy: ',best_val_accuracy)
#             best_model_state = surrogate_model.state_dict()

#         if epoch%20 == 0 and verbose:
#             print('Epoch: ',epoch)
#             print("Surr Loss : ",loss.item())
#     surrogate_model.load_state_dict(best_model_state) # Take the best model
#     with torch.no_grad():
#         surrogate_model.eval()
#         Z_orig = surrogate_model(X, H) # Trained surrogate model
#     lap_dist = torch.tensor(0.0).to(device)
#     degree_penalty = torch.tensor(0.0).to(device)
#     loss_cls = torch.tensor(0.0).to(device)
#     for t in tqdm(range(T)):
#         runtime_start = time.time()
#         delta_H = (torch.randn_like(H)).requires_grad_()
#         delta_X = (torch.randn_like(X)).requires_grad_()
#         H_pert = torch.clamp(H + delta_H, 0, 1)
#         # H_pert = (H_pert>0.5).float()
#         X_pert = X + delta_X
#         L_pert = lap(H_pert)
#         Z_pert = surrogate_model(X_pert, H_pert)
#         delta_L = (L_pert@Z_pert - L_orig @ Z_orig)
#         lap_dist += torch.mean((delta_L**2))/T
#         # lap_dist += torch.norm(delta_L, p=2).mean()/T

#         dv_temp = H_pert @ torch.ones((H.shape[1],), device=device)
#         degree_violation = (dv_temp - dv_orig)
#         degree_penalty += (torch.sum(degree_violation ** 2) / n)/T
#         # degree_penalty = torch.abs(degree_violation).mean()
#         logits_adv = Z_pert
#         loss_cls += (F.cross_entropy(logits_adv[train_mask], y[train_mask]))/T
#         time1 = time.time() - runtime_start

#         # with torch.no_grad():
#         #     target_model.eval()
#         #     surrogate_model.eval()
#         #     surrogate_test_accuracy = accuracy(surrogate_model(X_pert, H_pert)[test_mask], y[test_mask]) 
#         #     target_model_test_accuracy = accuracy(target_model(X_pert, H_pert)[test_mask], y[test_mask])
#         #     surrogate_test_trajectory.append(surrogate_test_accuracy.item())
#         #     target_test_trajectory.append(target_model_test_accuracy.item())
#         # if t == T-1:
#         #     os.makedirs(os.path.join(root,str(args.seed)), exist_ok=True)
#         #     prefix = os.path.join(root,str(args.seed), 'SimpleHGNN_'+args.dataset+'_'+args.model+'_'+str(args.ptb_rate))
#         #     torch.save(best_model_state, prefix+'_weights.pth')
#         runtime_start2 = time.time()

#         deg_penalty_val = degree_penalty.item()
#         cls_loss_val = loss_cls.item()
#         lap_dist_val = lap_dist.item() if isinstance(lap_dist, torch.Tensor) else lap_dist
#         # loss_meta = lap_dist - degree_penalty + alpha * loss_cls
#         loss_meta = args.alpha * lap_dist - args.beta*degree_penalty + args.gamma * loss_cls

#         grads = torch.autograd.grad(loss_meta,[delta_H,delta_X])

#         lap_dist_trajectory.append(lap_dist_val)
#         loss_meta_trajectory.append(loss_meta.item())
#         # _, _, acc_drop = classification_drop(args,target_model, H, HG, X, H_pert, X_pert, y)
#         with torch.no_grad():
#             target_model.eval()
#             # _, _, acc_drop = classification_drop(args,target_model, H, HG, X, H_pert, X_pert, y)
#             # test_flag = False
#             # if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
#             #     data_input = [data.clone(), test_flag]
#             # else:
#             data_input = data.clone().to(device)
#             # data_input.x = X_pert
#             data_input.x = X_pert
#             # print("H_pert.shape:", H_pert.shape)
#             # print("data.num_hyperedges:", data_input.num_hyperedges)
#             edge_index = incidence_to_edge_index2(H_pert)
#             # data_input.edge_index = H_pert
#             # edge_index = generate_G_from_H(data)
#             # print("edge_index:", edge_index)
#             # print("edge_index.shape:", edge_index.shape)
#             data_input.edge_index = edge_index
#             data_input.n_x = X_pert.shape[0]

            
#             data_input = ExtractV2E(data_input)
#             # print('after extract v2e:',data_input.edge_index.shape)
#             data_input = Add_Self_Loops(data_input)
#             # print('after add self loops:',data_input.edge_index.shape)
#             # data_input.num_hyperedges = data_input.edge_index[0].max()-data_input.n_x+1
#             data_input.edge_index[1] -= data_input.edge_index[1].min()
#             data_input.edge_index = data_input.edge_index.to(device)
#             data_input.x = data_input.x.to(device)
#             if args.method in ['AllSetTransformer', 'AllDeepSets']:
#                 data_input = norm_contruction(data_input, option=args.normtype)
#             # print("max edge_index[1]:", data_input.edge_index[1].max().item())
#             # data_input = ExtractV2E(data_input)
#             test_flag = True
#             if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
#                 data_input = [data_input, test_flag]
#             # print('data_input',data_input)
#             # print('data: ',data)
#             # assert data_input.edge_index.shape[0] == 2
#             # assert data_input.edge_index[0].max() < data_input.x.shape[0], "Invalid node index"
#             # assert data_input.edge_index[1].max() < data.num_hyperedges, "Invalid hyperedge index"
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

#         # Proceed with original gradient ascent
#         delta_H = eta_H * grads[0].sign()
#         delta_X = eta_X * grads[1].sign()
#         flat = delta_H.abs().flatten()
#         topk = torch.topk(flat, k=min(delta_H.numel(), budget)).indices
#         delta_H_new = torch.zeros_like(delta_H)
#         delta_H_new.view(-1)[topk] = delta_H.view(-1)[topk]
#         # In the following code segment we do not update bad nodes (nodes whose deg <= 0 ) or bad edges (whose card <= 1)
#         H_temp = torch.clamp(H + delta_H_new, 0, 1)
#         row_degrees = H_temp.sum(dim=1)
#         col_degrees = H_temp.sum(dim=0)
#         bad_nodes = (row_degrees < 1).nonzero(as_tuple=True)[0]
#         if args.dataset == '20newsW100':
#             bad_edges = (col_degrees < 1).nonzero(as_tuple=True)[0]
#         else:
#             bad_edges = (col_degrees < 2).nonzero(as_tuple=True)[0]
#         # print('|bad nodes| = ',len(bad_nodes))
#         # print('|bad edges| = ',len(bad_edges))
#         # print('deg: ',row_degrees.mean(), row_degrees.min(),row_degrees.max())
#         # print('dim: ',col_degrees.mean(), col_degrees.min(),col_degrees.max())
#         delta_H_new[bad_nodes, :] = 0
#         delta_H_new[:, bad_edges] = 0
#         delta_H.copy_(delta_H_new)

#         delta_X = delta_X.clamp(-epsilon, epsilon)
#         time2 = time.time() - runtime_start2
#     # results = [(t, loss_meta, acc_drop, lap_shift, deg_penalty, cls_loss, lap_dist, feature_shift)]
#     # print(lap_shift_trajectory)
#     results = [loss_meta_trajectory, acc_drop_trajectory, lap_shift_trajectory, lap_dist_trajectory, cls_loss_trajectory, \
#                deg_penalty_trajectory,feature_shift_trajectory,surrogate_test_trajectory, target_test_trajectory]
#     # mask = filter_potential_singletons(torch.clamp(H + delta_H, 0, 1))
#     H_adv = topk_budget_flip(H, delta_H, budget)
#     return torch.clamp(H + delta_H, 0, 1), X + delta_X, results, time1+time2, best_model_state

# MeLA-FGSM
def meta_laplacian_FGSM(
    args, H, X, y, data, HG, surrogate_class, target_model,
    train_mask, val_mask, test_mask, logits_orig,
    budget=20, epsilon=0.05,  T=-1,          
    eta_H=1e-2, eta_X=1e-2,           
    alpha=1.0, beta = 1.0, gamma = 4.0, 
    reinit_if_stuck=True, verbose = False):
    """
    Clean MeLA-FGSM (single-step) poisoning attack.

    Steps:
      1) Train surrogate once on (X,H), keep best by val accuracy.
      2) Build one-step meta-objective:
           L_meta = alpha * || (L(H') - L(H)) Z || - beta * DegDrift(H') + gamma * CE(Z, y_train)
         where H' = clamp(H + ΔH, 0,1), X' = X + ΔX, and Z = surrogate(X', H').
      3) One FGSM step on (ΔH, ΔX):
           ΔH <- eta_H * sign(∇ΔH L_meta), then keep only top-k entries (budget) in magnitude
           ΔX <- clamp(eta_X * sign(∇ΔX L_meta), -epsilon, epsilon)
      4) Output discrete H_adv via topk_budget_flip(H, ΔH_step, budget) and X_adv = X + ΔX_step.
    """

    device = X.device
    H = H.clone().detach()
    X = X.clone().detach()

    n, m = H.shape

    # Precompute originals
    L_orig = lap(H)
    dv_orig = H @ torch.ones((m,), device=device)

    # -------------------------
    # 1) Train surrogate once
    # -------------------------
    if surrogate_class is None:
        surrogate_model = SimpleHGNN(
            X.shape[1], hidden_dim=args.MLP_hidden, out_dim=args.num_classes, device=device
        ).to(device)
    else:
        # surrogate_model = surrogate_class(...).to(device)  # if you ever implement this
        raise Exception("Other surrogates Not implemented")
    optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val = -float('inf')
    best_state = None

    for epoch in range(args.num_epochs_sur):
        surrogate_model.train()
        optimizer.zero_grad()
        logits = surrogate_model(X, H)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward(retain_graph=True)
        optimizer.step()

        surrogate_model.eval()
        with torch.no_grad():
            logits_val = surrogate_model(X, H)
            val_acc = accuracy(logits_val[val_mask], y[val_mask]).item()
        if val_acc > best_val:
            best_val = val_acc
            best_state = surrogate_model.state_dict() # {k: v.detach().clone() for k, v in surrogate_model.state_dict().items()}
        
        if epoch%20 == 0 and verbose:
            print('Epoch: ',epoch, " Surr Loss : ",loss.item())
    if best_state is not None:
        surrogate_model.load_state_dict(best_state)

    # surrogate_model.eval()
    with torch.no_grad():
        surrogate_model.eval()
        Z_orig = surrogate_model(X, H) # Trained surrogate model
    # -------------------------
    # 2) One-step meta objective
    # -------------------------
    # Single-step variables (start from 0 for FGSM)
    runtime_start = time.time()
    delta_H = torch.zeros_like(H, device=device, requires_grad=True)
    delta_X = torch.zeros_like(X, device=device, requires_grad=True)
    # delta_H = (1e-3 * torch.randn_like(H)).requires_grad_()
    # delta_X = (1e-3 * torch.randn_like(X)).requires_grad_()

    H_pert = torch.clamp(H + delta_H, 0.0, 1.0)  # continuous relaxation
    X_pert = X + delta_X

    L_pert = lap(H_pert)
    Z_pert = surrogate_model(X_pert, H_pert)

    # Laplacian embedding drift: (L' - L) Z
    delta_L = (L_pert@Z_pert - L_orig @ Z_orig)
    if args.loss == 'L2':
        lap_dist = torch.norm(delta_L, p=2)
    else:
        lap_dist = (delta_L**2).mean()
    # Degree drift penalty
    dv_temp = H_pert @ torch.ones((m,), device=device)
    degree_penalty = torch.sum((dv_temp - dv_orig) ** 2) / n

    # Classification (poisoning): use TRAIN labels only
    loss_cls = F.cross_entropy(Z_pert[train_mask], y[train_mask])

    # Meta objective (maximize)
    loss_meta = alpha*lap_dist - beta*degree_penalty + gamma * loss_cls

    # -------------------------
    # 3) FGSM update (single step)
    # -------------------------
    grads = torch.autograd.grad(loss_meta, [delta_H, delta_X])

    # score = grads[0].abs().flatten()
    # k = min(int(budget), score.numel())
    # topk_idx = torch.topk(score, k=k, largest=True).indices

    # # H step: sign(grad) + top-k projection to enforce budget
    # delta_H_step = 1 * grads[0].sign()
    # delta_H_sparse = torch.zeros_like(delta_H_step)
    # delta_H_sparse.view(-1)[topk_idx] = delta_H_step.view(-1)[topk_idx]
    # # print('max(), min(): ',delta_H_sparse.max(), delta_H_sparse.min())
    # # delta_H_sparse has -1/0/+1; convert to mask of selected entries
    # M = (delta_H_sparse != 0).float()   # {0,1}

    # # exact flip on selected entries
    # H_adv = H + (1 - 2*H) * M
    # # (optional) if H might not be perfectly binary:
    # H_adv = (H_adv > 0.5).float()
    gH = grads[0]
    flip_gain = (1 - 2*H) * gH          # positive => flipping increases loss_meta
    eligible = (flip_gain > 0)

    score = flip_gain.clone()
    score[~eligible] = -float("inf")

    flat = score.flatten()
    k = min(budget, flat.numel())
    topk_idx = torch.topk(flat, k=k).indices

    M = torch.zeros_like(flat)
    M[topk_idx] = 1.0
    M = M.view_as(H)
    # print(H.dtype)
    H_adv = H + (1 - 2*H) * M
    # print(H_adv.dtype)
    H_adv = (H_adv > 0.5).float()
    # print(H_adv.dtype)
    # print(grads[1].max(), grads[1].min(), grads[1].mean())
    # X step: l_inf FGSM + clamp to epsilon
    delta_X_step = epsilon * grads[1].sign()
    # delta_X_step = delta_X_step.clamp(-epsilon, epsilon)

    # Discretize H with your existing helper (0/1 with budget)
    # H_adv =  torch.clamp(H + delta_H_sparse, 0, 1)

    # Feature perturbation final
    X_adv = X + delta_X_step.detach()
    time_elapsed = time.time() - runtime_start
    # -------------------------
    # (Optional) quick diagnostics (single-step)
    # -------------------------
    with torch.no_grad():
        # report effective budget usage
        num_changed = (H_adv != H).sum().item()

    results = {
        "loss_meta": float(loss_meta.detach().cpu()),
        "lap_dist": float(lap_dist.detach().cpu()),
        "degree_penalty": float(degree_penalty.detach().cpu()),
        "loss_cls_train": float(loss_cls.detach().cpu()),
        "num_changed_H": int(num_changed),
        "best_val_acc_surrogate": float(best_val),
    }
    # print(results)

    return H_adv, X_adv, results, time_elapsed, best_state

# def meta_laplacian_pois_attack(root, H, X, y, data, HG, surrogate_class, target_model, train_mask, val_mask, test_mask, logits_orig, budget=20, epsilon=0.05, T=20, eta_H=1e-2, eta_X=1e-2, alpha=1.0, beta=1.0, gamma=2.0, reinit_if_stuck=True):
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
#         # delta_L = (L_pert - L_orig) @ Z
#         delta_L = L_pert@Z - L_orig @ Z_orig
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
#         loss_cls = F.cross_entropy(logits_adv[train_mask], y[train_mask])
#         # loss_cls = F.cross_entropy(logits_adv, model(H, X).argmax(dim=1))
#         lap_dist = (delta_L**2).mean()
#         # lap_dist = torch.norm(delta_L, p=2).mean()
#         # print(delta_L.shape)
#         # lap_dist = torch.mean(delta_L**2)
#         cls_loss_val = loss_cls.item()
#         lap_dist_val = lap_dist.item() if isinstance(lap_dist, torch.Tensor) else lap_dist
#         # loss_meta = lap_dist - degree_penalty + alpha * loss_cls
#         loss_meta = args.alpha * lap_dist - args.beta*degree_penalty + args.gamma * loss_cls

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
#     H_adv = topk_budget_flip(H, delta_H, budget)
#     return H_adv, X + delta_X, results, time1+time2, best_model_state

def _clone_params(model: nn.Module):
    return {k: v.detach().clone().requires_grad_(True)
            for k, v in model.named_parameters()}

def _clone_buffers(model: nn.Module):
    # buffers don't need grad; just carry them
    return {k: b for k, b in model.named_buffers()}

def _sgd_step(params: dict, grads: list, lr: float):
    """Differentiable SGD update: p <- p - lr * g"""
    out = {}
    for (k, p), g in zip(params.items(), grads):
        out[k] = p - lr * g
    return out

def _valid_topk_mask(H, delta_H, budget, min_node_deg=1, min_edge_deg=2):
    """
    Build a hard top-k mask over entries that won't violate degree/cardinality constraints.
    We only consider flipping entries that keep:
      node degree >= min_node_deg
      edge cardinality >= min_edge_deg
    after applying the flip direction suggested by delta_H.
    """
    device = H.device
    n, m = H.shape

    # Current degrees/cardinalities
    row_deg = H.sum(dim=1)          # (n,)
    col_deg = H.sum(dim=0)          # (m,)

    # Candidate scores
    scores = delta_H.abs().flatten()

    # We'll screen candidates by feasibility:
    # if delta_H>0 -> we set H_ij to 1
    # if delta_H<0 -> we set H_ij to 0
    # Only risky operation is removing a 1 (could drop degrees below threshold).
    H_flat = H.flatten()
    d_flat = delta_H.flatten()

    # If we remove (set to 0) where H_ij==1, node deg and edge deg decrease by 1.
    # If we add (set to 1) where H_ij==0, degrees increase by 1 (always safe).
    # So invalid only when: removing causes node deg-1 < min_node_deg OR edge deg-1 < min_edge_deg
    # Compute per-entry validity
    rows = torch.arange(n, device=device).repeat_interleave(m)
    cols = torch.arange(m, device=device).repeat(n)

    removing = (d_flat < 0) & (H_flat > 0.5)
    invalid_remove = removing & ((row_deg[rows] - 1 < min_node_deg) | (col_deg[cols] - 1 < min_edge_deg))
    valid = ~invalid_remove

    # Mask scores for invalid entries
    masked_scores = scores.clone()
    masked_scores[~valid] = -1.0  # never selected

    k = min(int(budget), (masked_scores > 0).sum().item())
    idx = torch.topk(masked_scores, k=k, largest=True).indices

    mask = torch.zeros_like(d_flat)
    mask[idx] = 1.0
    return mask.view_as(delta_H)

# def _topk_mask_like(delta_H: torch.Tensor, k: int):
#     """Hard top-k mask (0/1) computed from |delta_H|; treated as piecewise-constant."""
#     flat = delta_H.abs().flatten()
#     k = min(int(k), flat.numel())
#     idx = torch.topk(flat, k=k, largest=True).indices
#     mask = torch.zeros_like(flat)
#     mask[idx] = 1.0
#     return mask.view_as(delta_H)

# def meta_laplacian_pois_attack_unrolled(
#     args,
#     H, X, y,
#     data, HG,
#     surrogate_class,         # keep for compatibility; you can ignore if only SimpleHGNN
#     target_model,            # only used for logging if you want
#     train_mask, val_mask, test_mask,
#     logits_orig=None,        # only used for logging if you want
#     budget=20,
#     epsilon=0.05,
#     T=20,                    # outer steps (perturbation steps)
#     K=20,                    # inner steps (surrogate training epochs)  <-- you requested 20
#     eta_H=1e-2, eta_X=1e-2,  # outer step sizes
#     lr_inner=1e-2,           # inner SGD lr (can set = args.lr if you want)
#     alpha=1.0, beta=1.0, gamma=1.0,
#     use_val_meta=True,       # meta-loss on val (recommended)
#     use_lap_term=True,       # include alpha * ||(L'-L)Z|| term
#     use_deg_penalty=True,    # include beta * degree drift
#     reinit_theta_each_outer=True,  # match your “retrain each outer iter” style but differentiable
# ):
#     """
#     Bilevel / unrolled version of MLA poisoning:

#     Inner (K steps): theta <- argmin CE_train(theta; H', X')
#     Outer (T steps): maximize:
#         L_meta = gamma * CE_val(theta_K; H', X') + alpha * ||(L'-L)Z|| - beta * DegDrift(H')
#     wrt (delta_H, delta_X), with budget projection on delta_H and l_inf bound on delta_X.

#     Notes:
#     - Uses continuous relaxation H_pert = clamp(H + delta_H, 0, 1) during optimization.
#     - Enforces budget by hard top-k masking on delta_H each outer step (piecewise constant mask).
#     - Avoids invalid singleton nodes/edges using your row/col-degree filters.
#     """

#     device = X.device
#     H = H.clone().detach().to(device)
#     X = X.clone().detach().to(device)
#     y = y.to(device)

#     n, m = H.shape
#     ones_m = torch.ones((m,), device=device)

#     # Precompute original Laplacian / degrees
#     L_orig = lap(H)                      # (n,n) or appropriate operator
#     dv_orig = H @ ones_m                 # (n,)

#     clean_sur = SimpleHGNN(X.shape[1], hidden_dim=args.MLP_hidden,
#                            out_dim=args.num_classes, device=device).to(device)
#     with torch.no_grad():
#         Z_orig = clean_sur(X, H)

#     # Initialize perturbations
#     delta_H = (1e-3 * torch.randn_like(H, device=device)).requires_grad_()
#     delta_X = (1e-3 * torch.randn_like(X, device=device)).requires_grad_()

#     # Logs
#     meta_vals = []
#     lap_vals = []
#     deg_vals = []
#     ce_vals = []
#     loss_meta_trajectory = []
#     acc_drop_trajectory = []
#     lap_shift_trajectory = []
#     lap_dist_trajectory = []
#     cls_loss_trajectory = []
#     deg_penalty_trajectory = []
#     feature_shift_trajectory = []
#     surrogate_test_trajectory = []
#     target_test_trajectory = []
#     start_time = time.time()
#     for t in tqdm(range(T)):
#         # ---- 1) Project delta_H to budget (hard top-k) ----
#         with torch.no_grad():
#             delta_X.clamp_(-epsilon, epsilon)

#         # ---- 2) Build continuous perturbed inputs ----
#         H_pert = torch.clamp(H + delta_H, 0.0, 1.0)
#         X_pert = X + delta_X

#         # ---- 3) Inner training: unroll K steps of surrogate minimizing train CE ----
#         # Initialize surrogate model
#         if surrogate_class is None:
#             surrogate_model = SimpleHGNN(
#                 X.shape[1],
#                 hidden_dim=args.MLP_hidden,
#                 out_dim=args.num_classes,
#                 device=device
#             ).to(device)
#         else:
#             raise RuntimeError("surrogate_class not implemented here; keep SimpleHGNN for now.")

#         # Grab initial parameters for functional training
#         theta = _clone_params(surrogate_model)
#         # buffers = _clone_buffers(surrogate_model)

#         if not reinit_theta_each_outer and t > 0:
#             # If you want to warm start, you’d carry theta across iterations.
#             # For now, default is True (reinit each outer iter), matching your style.
#             pass

#         # Unrolled SGD steps
#         for _ in range(args.num_epochs_sur):
#             # logits = functional_call(surrogate_model, (theta, buffers), (X_pert, H_pert))
#             logits = functional_call(surrogate_model, theta, (X_pert, H_pert))
#             loss_train = F.cross_entropy(logits[train_mask], y[train_mask])
#             grads = torch.autograd.grad(
#                 loss_train,
#                 list(theta.values()),
#                 create_graph=True,   # IMPORTANT: enables outer gradient through inner steps
#                 retain_graph=True
#             )
#             theta = _sgd_step(theta, grads, lr_inner)

#         # ---- 4) Outer meta objective computed with trained theta_K ----
#         # Z = functional_call(surrogate_model, (theta, buffers), (X_pert, H_pert))
#         Z = functional_call(surrogate_model, theta, (X_pert, H_pert))
#         with torch.no_grad():
#             surrogate_test_accuracy = accuracy(Z[test_mask], y[test_mask]) 
#             surrogate_test_trajectory.append(surrogate_test_accuracy.item())

#         # Meta classification loss: val (recommended) or train (if you insist)
#         if use_val_meta:
#             loss_ce_meta = F.cross_entropy(Z[val_mask], y[val_mask])
#         else:
#             loss_ce_meta = F.cross_entropy(Z[train_mask], y[train_mask])

#         # Laplacian embedding drift: || (L' - L) Z ||  (paper form)
#         if use_lap_term:
#             L_pert = lap(H_pert)
#             # delta_LZ = (L_pert - L_orig) @ Z
#             delta_LZ = ((L_pert @ Z) - (L_orig @ Z_orig))
#             lap_dist = (delta_LZ**2).mean() #torch.norm(delta_LZ, p=2).mean()
#         else:
#             lap_dist = torch.zeros((), device=device)

#         # Degree drift penalty
#         if use_deg_penalty:
#             dv_temp = H_pert @ ones_m
#             deg_penalty = torch.sum((dv_temp - dv_orig) ** 2) / n
#         else:
#             deg_penalty = torch.zeros((), device=device)

#         # We want attacker to MAXIMIZE meta objective
#         # (so we ASCEND its gradient w.r.t delta_H, delta_X)
#         loss_meta = gamma * loss_ce_meta + alpha * lap_dist - beta * deg_penalty

#         # ---- 5) Outer gradient step on perturbations (FGSM/PGD style) ----
#         grads = torch.autograd.grad(loss_meta, [delta_H, delta_X], retain_graph=False)

#         with torch.no_grad():
#             delta_H.add_(eta_H * grads[0].sign())
#             delta_X.add_(eta_X * grads[1].sign())
#             # Re-enforce l_inf bound immediately
#             delta_X.clamp_(-epsilon, epsilon)
#             # ---- NOW enforce budget ----
#             min_edge = 1 if getattr(args, "dataset", "") == "20newsW100" else 2
#             mask = _valid_topk_mask(H, delta_H, budget, min_node_deg=1, min_edge_deg=min_edge)
#             delta_H.mul_(mask)
#         # with torch.no_grad():
#         #     l0 = (H_adv != H).sum().item() if 'H_adv' in locals() else -1
#         #     cont_nz = (delta_H.abs() > 1e-12).sum().item()
#         #     print(
#         #     f"[t={t}] lap={lap_dist.item():.4e} deg={degree_penalty.item():.4e} "
#         #     f"ce={loss_cls.item():.4e} | "
#         #     f"gradH(max/mean)={grads_H.abs().max().item():.3e}/{grads_H.abs().mean().item():.3e} "
#         #     f"gradX(max/mean)={grads_X.abs().max().item():.3e}/{grads_X.abs().mean().item():.3e} | "
#         #     f"deltaH_nz={cont_nz} deltaX_linf={delta_X.abs().max().item():.3e}"
#         #     )
#         # ---- logging ----
#         meta_vals.append(float(loss_meta.detach().cpu()))
#         ce_vals.append(float(loss_ce_meta.detach().cpu()))
#         lap_vals.append(float(lap_dist.detach().cpu()))
#         deg_vals.append(float(deg_penalty.detach().cpu()))
#         lap_dist_trajectory.append(float(lap_dist.detach().cpu()))
#         loss_meta_trajectory.append(float(loss_meta.detach().cpu()))
#         cls_loss_trajectory.append(ce_vals[-1])
#         deg_penalty_trajectory.append(deg_vals[-1])
#         feature_shift = torch.norm(delta_X, p=2).item()
#         feature_shift_trajectory.append(feature_shift)
#         lap_shift_trajectory.append(laplacian_diff(H, torch.clamp(H + delta_H, 0, 1)))
#     # ---- 6) Final discrete projection for H_adv ----
#     with torch.no_grad():
#         # enforce final budget again (safety)
#         min_edge = 1 if getattr(args, "dataset", "") == "20newsW100" else 2
#         # mask = _valid_topk_mask(H, delta_H, budget, min_node_deg=1, min_edge_deg=min_edge)
#         # delta_H_final = delta_H * mask
#         delta_H_final = delta_H
#         H_adv = topk_budget_flip(H, delta_H_final, budget)
#         X_adv = X + delta_X.clamp(-epsilon, epsilon)
#     total_time = time.time() - start_time
#     # results = {
#     #     "loss_meta_traj": meta_vals,
#     #     "ce_meta_traj": ce_vals,
#     #     "lap_dist_traj": lap_vals,
#     #     "deg_penalty_traj": deg_vals,
#     #     "final_x_linf": float(delta_X.detach().abs().max().cpu()),
#     #     "final_h_nonzero": int((delta_H.detach() != 0).sum().cpu().item()),
#     # }
#     results = [loss_meta_trajectory, acc_drop_trajectory, lap_shift_trajectory, lap_dist_trajectory, cls_loss_trajectory, \
#                deg_penalty_trajectory,feature_shift_trajectory,surrogate_test_trajectory, target_test_trajectory]

#     return H_adv, X_adv, results, total_time, None

# MeLA-D
def meta_laplacian_pois_attack(args, root, H, X, y, data, HG, surrogate_class, target_model, train_mask, val_mask, test_mask, logits_orig, budget=20, epsilon=0.05, T=20, eta_H=1e-2, eta_X=1e-2, reinit_if_stuck=True):
    """
      Meta Laplacian Attack adapted to poisoning setting (training-time). 
    - The attacker perturbs H and X before training.
    - A new model is trained from scratch at every iteration to simulte a bilevel optimization. 
    Params:
        H,X,y: Original incidence matrix, features and labels
        surrogate_class: Constructor of the surrogate model (e.g. SimpleHGNN)
    """
    alpha, beta, gamma = args.alpha, args.beta, args.gamma
    verbose = False
    device = X.device
    idx_unlabeled = val_mask | test_mask
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
    time1, time2, time3 = 0, 0, 0
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
        # if args.dataset == '20newsW100':
        #     surrogate_epochs = args.num_epochs_sur
        # else:
        #     if t == T-1:
        #         surrogate_epochs = args.epochs 
        #     else:
        #         surrogate_epochs = args.num_epochs_sur
        # data.x = X_pert
        # data.edge_index = incidence_to_edge_index(H_pert)
        for epoch in range(args.num_epochs_sur):
            surrogate_model.train()
            optimizer.zero_grad()
            logits = surrogate_model(X_pert, H_pert)
            # print(logits[train_mask].shape,y[train_mask].shape,logits.shape,y.shape)
            loss = criterion(logits[train_mask],y[train_mask])
            loss.backward(retain_graph=True)
            optimizer.step()
            # if t == T-1:
            #     surrogate_model.eval()
            #     # Save the surrogate model (which has the best validation accuracy) for robust training
            #     with torch.no_grad():   
            #         # val_loss = criterion(logits[val_mask], y[val_mask])
            #         val_accuracy = accuracy(logits[val_mask],y[val_mask])
            #     if val_accuracy.item() > best_val_accuracy:
            #         best_val_accuracy = val_accuracy.item()
            #         # print('Best val accuracy: ',best_val_accuracy)
            #         best_model_state = surrogate_model.state_dict()

            # if epoch%20 == 0 and verbose:
            #     print('Epoch: ',epoch)
            #     with torch.no_grad():
            #         target_model.eval()
            #         # --------
            #         data_input = data.clone().to(device)
            #         data_input.x = X_pert
            #         edge_index = incidence_to_edge_index2(H_pert)
            #         data_input.edge_index = edge_index
            #         data_input.n_x = X_pert.shape[0]
            #         data_input = ExtractV2E(data_input)
            #         data_input = Add_Self_Loops(data_input)
            #         # _, _, acc_drop = classification_drop(args,target_model, H, HG, X, H_pert, X_pert, y)
            #         data_input.edge_index[1] -= data_input.edge_index[1].min()
            #         data_input.edge_index = data_input.edge_index.to(device)
            #         if args.method in ['AllSetTransformer', 'AllDeepSets']:
            #             data_input = norm_contruction(data_input, option=args.normtype)

            #         test_flag = True
            #         if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
            #             data_input = [data_input, test_flag]
            #         # ------
            #         logits_adv = target_model(data_input)
            #     acc_orig = (logits_orig.argmax(dim=1)[test_mask] == y[test_mask]).float().mean().item()
            #     acc_adv = (logits_adv.argmax(dim=1)[test_mask] == y[test_mask]).float().mean().item()
            #     acc_drop = (acc_orig - acc_adv)/acc_orig
            #     _, _, acc_drop_sur = classification_drop(args,surrogate_model, H, None, X, H_pert, X_pert, y)
            #     print("Surr Loss : ",loss.item()," Accuracy drop (surrogate): ", acc_drop_sur*100,'%', " Accuracy drop (target): ", acc_drop*100,'%')
        
        time1 += (time.time() - runtime_start)
        # if t == T-1:
        #     surrogate_model.load_state_dict(best_model_state) # Take the best model
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
        # if t == T-1:
        #     os.makedirs(os.path.join(root,str(args.seed)), exist_ok=True)
            # prefix = os.path.join(root,str(args.seed), 'SimpleHGNN_'+args.dataset+'_'+args.model+'_'+str(args.ptb_rate))
            # torch.save(best_model_state, prefix+'_weights.pth')
        runtime_start2 = time.time()
        Z_orig = surrogate_model(X, H) # Trained surrogate model on unperturbed X,H
        Z = surrogate_model(X_pert, H_pert) # Trained surrogate model
        # delta_L = (L_pert - L_orig) @ Z
        delta_L = (L_pert @ Z- L_orig @ Z_orig)
        # loss_meta = (delta_L**2).sum()
        # H_temp = torch.clamp(H + delta_H, 0, 1)
        dv_temp = H_pert @ torch.ones((H.shape[1],), device=device)
        degree_violation = (dv_temp - dv_orig)
        degree_penalty = torch.sum(degree_violation ** 2) / n
        # degree_penalty = torch.abs(degree_violation).mean()
        deg_penalty_val = degree_penalty.item()
        # loss_meta += degree_penalty

        # logits_adv = target_model(X_pert,H_pert)
        logits_adv = Z
        loss_cls = F.cross_entropy(logits_adv[train_mask], y[train_mask])
        # loss_cls = F.cross_entropy(logits_adv, model(H, X).argmax(dim=1))
        if args.loss == 'L2':
            lap_dist = torch.norm(delta_L, p=2)
        else:
            lap_dist = (delta_L**2).mean()
        # print(delta_L.shape)
        # lap_dist = torch.mean(delta_L**2)
        cls_loss_val = loss_cls.item()
        lap_dist_val = lap_dist.item() if isinstance(lap_dist, torch.Tensor) else lap_dist
        loss_meta = args.alpha * lap_dist - args.beta*degree_penalty + args.gamma * loss_cls

        grads = torch.autograd.grad(loss_meta,[delta_H,delta_X])
        time2 += (time.time() - runtime_start2)

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
        runtime_start3 = time.time()
        with torch.no_grad():
            # Proceed with original gradient ascent
            delta_H += eta_H * grads[0].sign()
            delta_X += eta_X * grads[1].sign()
            flat = delta_H.abs().flatten()
            topk = torch.topk(flat, k=min(delta_H.numel(), budget)).indices
            delta_H_new = torch.zeros_like(delta_H)
            delta_H_new.view(-1)[topk] = delta_H.view(-1)[topk]
            # In the following code segment we do not update bad nodes (nodes whose deg <= 0 ) or bad edges (whose card <= 1)
            # H_temp = torch.clamp(H + delta_H_new, 0, 1)
            # row_degrees = H_temp.sum(dim=1)
            # col_degrees = H_temp.sum(dim=0)
            # bad_nodes = (row_degrees < 1).nonzero(as_tuple=True)[0]
            # if args.dataset == '20newsW100':
            #     bad_edges = (col_degrees < 1).nonzero(as_tuple=True)[0]
            # else:
            #     bad_edges = (col_degrees < 2).nonzero(as_tuple=True)[0]
            # # print('|bad nodes| = ',len(bad_nodes))
            # # print('|bad edges| = ',len(bad_edges))
            # delta_H_new[bad_nodes, :] = 0
            # delta_H_new[:, bad_edges] = 0
            delta_H.copy_(delta_H_new)

        delta_X = delta_X.clamp(-epsilon, epsilon)
        time3 += (time.time() - runtime_start3)
    # results = [(t, loss_meta, acc_drop, lap_shift, deg_penalty, cls_loss, lap_dist, feature_shift)]
    results = [loss_meta_trajectory, acc_drop_trajectory, lap_shift_trajectory, lap_dist_trajectory, cls_loss_trajectory, \
               deg_penalty_trajectory,feature_shift_trajectory,surrogate_test_trajectory, target_test_trajectory]
    # results = [] 
    M = (delta_H != 0).float()
    # exact flip on selected entries
    H_adv = H + (1 - 2*H) * M
    # (optional) if H might not be perfectly binary:
    H_adv = (H_adv > 0.5).float()
    # mask = filter_potential_singletons(torch.clamp(H + delta_H, 0, 1))
    return H_adv.detach(), X + delta_X.detach(), results, time1+time2+time3, None

def meta_laplacian_PGD(
    args, H, X, y,
    train_mask,
    budget=20, epsilon=0.05, T=20,
    eta_H=1e-2, eta_X=None
):
    """
    PGD-style MeLA attack (no meta-learning unrolling).

    Stronger than FGSM by multi-step ascent on a fixed meta-loss.
    """
    alpha, beta, gamma = args.alpha, args.beta, args.gamma
    device = X.device
    if eta_X is None:
        eta_X = epsilon / T

    H = H.clone().detach()
    X = X.clone().detach()

    n, m = H.shape
    start_time = time.time()
    # --------------------------------------------------
    # 1) Train surrogate ONCE (clean)
    # --------------------------------------------------
    surrogate = SimpleHGNN(
        X.shape[1],
        hidden_dim=args.MLP_hidden,
        out_dim=args.num_classes,
        device=device
    ).to(device)

    opt = torch.optim.Adam(surrogate.parameters(), lr=args.lr)
    for _ in range(args.num_epochs_sur):
        opt.zero_grad()
        loss = F.cross_entropy(surrogate(X, H)[train_mask], y[train_mask])
        loss.backward()
        opt.step()

    surrogate.eval()
    with torch.no_grad():
        Z_orig = surrogate(X, H)

    L_orig = lap(H)
    dv_orig = H @ torch.ones((m,), device=device)

    # --------------------------------------------------
    # 2) Initialize perturbations
    # --------------------------------------------------
    delta_H = torch.zeros_like(H, requires_grad=True)
    delta_X = torch.zeros_like(X, requires_grad=True)

    # --------------------------------------------------
    # 3) PGD loop
    # --------------------------------------------------
    loss_metas = []
    lap_dists = []
    degree_penalties = []
    loss_cls_list = []

    for _ in tqdm(range(T)):

        H_pert = torch.clamp(H + delta_H, 0, 1)
        X_pert = X + delta_X

        L_pert = lap(H_pert)
        Z = surrogate(X_pert, H_pert)

        # Meta-loss
        delta_L = L_pert @ Z - L_orig @ Z_orig
        lap_dist = (
            torch.norm(delta_L, p=2)
            if args.loss == "L2"
            else (delta_L ** 2).mean()
        )

        dv_temp = H_pert @ torch.ones((m,), device=device)
        deg_penalty = torch.sum((dv_temp - dv_orig) ** 2) / n
        cls_loss = F.cross_entropy(Z[train_mask], y[train_mask])

        loss_meta = alpha * lap_dist - beta * deg_penalty + gamma * cls_loss
        loss_metas.append(loss_meta.item())
        lap_dists.append(lap_dist.item())
        degree_penalties.append(deg_penalty.item())
        loss_cls_list.append(cls_loss.item())
        gH, gX = torch.autograd.grad(loss_meta, [delta_H, delta_X])

        with torch.no_grad():
            # ---- H: gradient-based top-k PGD
            flip_gain = (1 - 2 * H) * gH
            score = flip_gain.clone()
            score[score <= 0] = -float("inf")

            flat = score.flatten()
            topk = torch.topk(flat, k=min(budget, flat.numel())).indices

            M = torch.zeros_like(flat)
            M[topk] = 1.0
            M = M.view_as(H)

            delta_H.copy_(eta_H * gH.sign())
            delta_H.mul_(M)

            # ---- X: standard PGD
            delta_X.add_(eta_X * gX.sign())
            delta_X.clamp_(-epsilon, epsilon)

    # --------------------------------------------------
    # 4) Discretize
    # --------------------------------------------------
    with torch.no_grad():
        M = (delta_H != 0).float()
        H_adv = H + (1 - 2 * H) * M
        H_adv = (H_adv > 0.5).float()
        X_adv = X + delta_X
    total_time = time.time() - start_time
    with torch.no_grad():
        # report effective budget usage
        num_changed = (H_adv != H).sum().item()

    results = {
        "num_changed_H": int(num_changed)    
    }
    # print("loss trajectory: ", loss_metas)
    # print('degree penalty trajectory: ', degree_penalties)
    # print('laplacian distance trajectory: ', lap_dists)
    # print('classification loss trajectory: ', loss_cls_list)
    return H_adv, X_adv, results, total_time

def get_attack(target_model,H,X,y,data,HG,train_mask,val_mask,test_mask,perturbations):
    if args.attack == 'gradargmax':
        if args.method != 'simplehgnn':
            target_model = SimpleHGNN(X.shape[1],hidden_dim = args.MLP_hidden, out_dim = args.num_classes,device = X.device)
        # print('surrogate : ',target_model)
        attack_model = GradArgmax(model=target_model.to(device), nnodes=X.shape[0], nnedges = H.shape[1], \
                                attack_structure=True, device=device)
        time1 = time.time()
        attack_model.attack(X, H.clone(), y, n_perturbations=perturbations, train_mask=train_mask)
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

        # # Flatten index space and choose delta random indices to modify
        # indices = torch.randperm(total_elements, device=H.device)[:perturbations]

        # # Convert flat indices to 2D indices (rows and columns)
        # rows = indices // m
        # cols = indices % m

        # Generate random signs (-1 or +1) for each of the delta indices
        # signs = torch.randint(0, 2, (perturbations,), device=H.device) * 2 - 1  # {-1, +1}

        # Directly apply the perturbations to the selected indices
        # H_adv[rows, cols] += signs
        k = min(perturbations, total_elements)  # safety

        # Sample k unique flat indices
        flat_idx = torch.randperm(k, device=H.device)[:k]

        # Convert to 2D indices
        rows = flat_idx // m
        cols = flat_idx % m

        # Flip bits: 0 -> 1, 1 -> 0
        H_adv[rows, cols] = 1 - H_adv[rows, cols]
        # H_adv[rows, cols] = 1 - H_adv[rows, cols]

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
    parser.add_argument('--method', default='AllSetTransformer',choices=['AllSetTransformer', 'HGNN'])
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
                    choices=['mla','Rand-flip', 'Rand-feat','gradargmax','mla_fgsm','mla_pgd'], help='attack variant')
    parser.add_argument('--epsilon', type=float, default=0.05, help='Node Feature perturbation bound')
    parser.add_argument('--ptb_rate', type=float, default=0.2,  help='pertubation rate')
    parser.add_argument('--patience', type=int, default=150,
                    help='Patience for training with early stopping.')
    parser.add_argument('--T', type=int, default=80, help='Number of iterations for the attack.')
    # parser.add_argument('--mla_alpha', type=float, default=4.0, help='weight for classification loss')
    parser.add_argument('--beta', type=float, default= 1.0, help='weight for degree penalty loss component')
    parser.add_argument('--gamma', type=float, default=4.0, help='weight for classification loss component')
    parser.add_argument('--alpha', type=float, default=0.1, help='weight for laplacian Loss component')
    parser.add_argument('--eta_H', type=float, default=1e-2, help='Learning rate for H perturbation')
    parser.add_argument('--eta_X', type=float, default=1e-2, help='Learning rate for X perturbation')
    parser.add_argument('--num_epochs_sur', type=int, default=80, help='#epochs for the surrogate training.')
    parser.add_argument('--loss', type=str, default='L2', help='Loss to measure laplacian distance.', choices=['MSE','L2'])

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
    root='./base_newsplit' # Stores the results and various statistics
    root2='./'+args.method+'_Melad' # Stores the perturbed H, X in various formats
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
    setup_seed(33) 
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

    # best_val, best_test = logger.print_statistics()
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
    # H = torch.zeros((data.x.shape[0], data.num_hyperedges))
    # src, dst = data.edge_index
    # # # Find node-to-hyperedge edges (i.e., edges where node index < num_nodes and hyperedge index >= num_nodes)
    # node_mask = src < data.x.shape[0]
    # node_indices = src[node_mask]
    # # ExtractV2E(data)
    # hyperedge_indices = (data.edge_index[1] - data.x.shape[0] + 1).clip(0,data.num_hyperedges-1)  # shift back
    # assert hyperedge_indices.max() < data.num_hyperedges, "Hyperedge index out of bounds"
    # H[node_indices, hyperedge_indices] = 1.0
    # H = H.to(device)
    # row_degrees = H.sum(dim=1)
    # col_degrees = H.sum(dim=0)
    # print('original dataset (degrees): ',row_degrees.mean().item(),row_degrees.std().item(),row_degrees.min().item(),row_degrees.max().item())
    # print('original dataset (dim): ',col_degrees.mean().item(),col_degrees.std().item(),col_degrees.min().item(),col_degrees.max().item())

    # row_sums = H.sum(dim=1)
    # has_zero_row = (row_sums == 0).any()
    # print('H degree 0: ',has_zero_row)
    # print('H empty edge: ', (H.sum(dim=0) == 0).any())

    # print('H: ',H.shape)
    X = data.x.to(device)
    n = data.n_x 
    # e = data.num_hyperedges
    e = data.edge_index.shape[1]
    y = data.y.to(device)

    perturbations = int(args.ptb_rate * e)
    args.__setattr__('model', args.method)
    print("============ ",args.model, args.dataset,args.attack,str(args.seed),"==================")
    if args.attack == 'mla':
        H_adv, X_adv, results, exec_time, robust_model_states = meta_laplacian_pois_attack(args,root, H, X, y, data, None, None, model, \
                        train_mask, val_mask, test_mask, Z_orig, budget=perturbations, epsilon=args.epsilon, T=args.T, \
                        eta_H=args.eta_H, eta_X=args.eta_X, \
                        reinit_if_stuck=True)
        # H_adv, X_adv, results, exec_time, _ = meta_laplacian_pois_attack_unrolled(
        #     args,
        #     H, X, y,
        #     data, None,
        #     None,         # keep for compatibility; you can ignore if only SimpleHGNN
        #     model,            # only used for logging if you want
        #     train_mask, val_mask, test_mask,
        #     logits_orig=Z_orig,        # only used for logging if you want
        #     budget=perturbations,
        #     epsilon=args.epsilon,
        #     T=args.T,                    # outer steps (perturbation steps)
        #     K=20,                    # inner steps (surrogate training epochs)  <-- you requested 20
        #     eta_H=args.eta_H, eta_X=args.eta_X,  # outer step sizes
        #     lr_inner=args.lr,           # inner SGD lr (can set = args.lr if you want)
        #     alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        #     use_val_meta=False,       # meta-loss on val (recommended)
        #     use_lap_term=True,       # include alpha * ||(L'-L)Z|| term
        #     use_deg_penalty=True,    # include beta * degree drift
        #     reinit_theta_each_outer=True  # match your “retrain each outer iter” style but differentiable
        # )
        # with torch.no_grad():
        #     diff = (H_adv != H)
        #     actual_l0 = diff.sum().item()
        #     per_node = diff.sum(dim=1)   # how many flips per vertex
        #     per_edge = diff.sum(dim=0)   # how many flips per hyperedge
        #     print(f"[FINAL] actual ||H_adv-H||_0 = {actual_l0} (budget={perturbations})")
        #     print(f"[FINAL] nodes_touched = {(per_node>0).sum().item()}, max_flips_on_one_node = {per_node.max().item()}")
        #     print(f"[FINAL] edges_touched = {(per_edge>0).sum().item()}, max_flips_on_one_edge = {per_edge.max().item()}")

        # save_npz(root, args.seed, results)
        H_adv = H_adv.detach()
        X_adv = X_adv.detach()
        X_adv.requires_grad = False

    elif args.attack == 'mla_fgsm':
        H_adv, X_adv, results, exec_time, robust_model_states = meta_laplacian_FGSM(args, H, X, y, data, None, None, model, \
                        train_mask, val_mask, test_mask, Z_orig, budget=perturbations, epsilon=args.epsilon, \
                        eta_H=args.eta_H, eta_X=args.eta_X, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
        print(json.dumps(results,indent = 4))
    
    elif args.attack == 'mla_pgd':
        H_adv, X_adv, results, exec_time = meta_laplacian_PGD(args, H, X, y, train_mask, \
                                          budget = perturbations, epsilon=args.epsilon, T=args.T, \
                                            eta_H=args.eta_H, eta_X=None)
        H_adv = H_adv.detach()
        X_adv = X_adv.detach()
        X_adv.requires_grad = False
    else:
        H_adv, X_adv, exec_time = get_attack(model, H, X, y,data, None, train_mask,val_mask,test_mask,perturbations = perturbations)
    
    os.system('mkdir -p '+root2)
    np.savez(os.path.join(root2, args.model+"_"+args.attack+"_"+args.dataset+"_"+str(args.seed)+ '_H_adv.npz'), H_adv.clone().cpu().numpy())
    np.savez(os.path.join(root2, args.model+"_"+args.attack+"_"+args.dataset+"_"+str(args.seed)+ '_X_adv.npz'), X_adv.clone().cpu().numpy())
    # print('H_adv:', H_adv)
    # if save and args.attack == 'mla':
    #     plot_results(args,results,root)
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
    save_pth = os.path.join(root2,args.model+"_"+args.attack+"_"+args.dataset+"_"+str(args.seed)+ "_data.pth")
    print('saving perturations from ',args.attack,' on ',args.dataset,' at: ',save_pth)
    # torch.save(data_clone,save_pth)
    # print(data_clone.edge_index[0].max(),data_clone.edge_index[0].min(),data_clone.edge_index[1].max(),data_clone.edge_index[1].min())
    if args.method in ['AllSetTransformer', 'AllDeepSets']:
        data_clone = norm_contruction(data_clone, option=args.normtype)

    test_flag = True
    if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
        data_input = [data_clone, test_flag]
    else:
        data_input = data_clone
        # save_pth = os.path.join(root,args.model+"_"+args.attack+"_"+args.dataset+"_"+str(args.seed)+ "_data.pth")
        # print('saving ',save_pth)
        # torch.save(data_input,save_pth)


    # print('data_input',data_input)
    # print('data: ',data)
    # assert data_input.edge_index.shape[0] == 2
    # assert data_input.edge_index[0].max() < data_input.x.shape[0], "Invalid node index"
    # assert data_input.edge_index[1].max() < data.num_hyperedges, "Invalid hyperedge index"
    # ------------------------ Evasion setting ------------------------
    Z_adv = model(data_input)
    evasion_dict = compute_statistics(H,H_adv,Z_orig,Z_adv,X,X_adv,train_mask,val_mask,test_mask,y)
    evasion_dict['exec_time'] = exec_time
    if type(e) == np.int32:
        e = int(e)
    if type(n) == np.int32:
        n = int(n)
    evasion_dict['num_edges'] = e
    evasion_dict['num_vertices'] = n
    evasion_dict['num_to_perturb'] = perturbations
    # l0_entry = (H_adv != H).sum().item()
    # evasion_dict['changed_ratio'] = l0_entry / H.numel()
    evasion_dict['edges_changed'] = ((H_adv != H).any(dim=0)).sum().item()
    evasion_dict['nodes_changed'] = ((H_adv != H).any(dim=1)).sum().item()

    # print(evasion_dict)
    print(json.dumps(evasion_dict,indent = 4))
    # # print('H_adv - H:',torch.sum((H_adv-H).abs()))
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
    # l0_entry = (H_adv != H).sum().item()
    # results['changed_ratio'] = l0_entry / H.numel()
    results['edges_changed'] = ((H_adv != H).any(dim=0)).sum().item()
    results['nodes_changed'] = ((H_adv != H).any(dim=1)).sum().item()
    degree_adv = H_adv.sum(dim=1)
    np.savez(os.path.join(root, args.model+"_"+args.attack+"_"+args.dataset+"_"+str(args.seed)+ '_deg_H_adv.npz'), degree_adv.clone().cpu().numpy())

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
    # print('l0_entry:', l0_entry)

    if save:
        results.update(vars(args))
        evasion_dict.update(vars(args))
        
        save_to_csv(evasion_dict,filename=os.path.join(root,'evasion_results_ICML2.csv'))
        save_to_csv(results,filename=os.path.join(root,'pois_results_ICML2.csv'))
    