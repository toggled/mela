#!/usr/bin/env python
# coding: utf-8

import os
import json
import time
# import math
import torch
import pickle
import argparse
import random

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

from convert_datasets_to_pygDataset import dataset_Hypergraph
import sys
# sys.path.append('../../')
from mla_utils import *
from modelzoo import SimpleHGNN,GradArgmax

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
    idx_unlabeled = val_mask | test_mask
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
        loss_meta = lap_dist + degree_penalty + alpha * loss_cls

        grads = torch.autograd.grad(loss_meta,[delta_H,delta_X])

        lap_dist_trajectory.append(lap_dist_val)
        loss_meta_trajectory.append(loss_meta.item())
        # _, _, acc_drop = classification_drop(args,target_model, H, HG, X, H_pert, X_pert, y)
        with torch.no_grad():
            target_model.eval()
            # _, _, acc_drop = classification_drop(args,target_model, H, HG, X, H_pert, X_pert, y)
            logits_adv,_ = target_model(data)
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
        bad_edges = (col_degrees < 2).nonzero(as_tuple=True)[0]
        delta_H_new[bad_nodes, :] = 0
        delta_H_new[:, bad_edges] = 0
        delta_H.copy_(delta_H_new)

        delta_X = delta_X.clamp(-epsilon, epsilon)
        time2 = time.time() - runtime_start2
    # results = [(t, loss_meta, acc_drop, lap_shift, deg_penalty, cls_loss, lap_dist, feature_shift)]
    results = [loss_meta_trajectory, acc_drop_trajectory, lap_shift_trajectory, lap_dist_trajectory, cls_loss_trajectory, \
               deg_penalty_trajectory,feature_shift_trajectory,surrogate_test_trajectory, target_test_trajectory]
    # mask = filter_potential_singletons(torch.clamp(H + delta_H, 0, 1))
    return torch.clamp(H + delta_H, 0, 1), X + delta_X, results, time1+time2, best_model_state

def meta_laplacian_pois_attack(root, H, X, y, data, HG, surrogate_class, target_model, train_mask, val_mask, test_mask, logits_orig, budget=20, epsilon=0.05, T=20, eta_H=1e-2, eta_X=1e-2, alpha=4.0, reinit_if_stuck=True):
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
        if t == T-1:
            surrogate_epochs = args.epochs 
        else:
            surrogate_epochs = args.num_epochs_sur
        data.x = X_pert
        data.edge_index = incidence_to_edge_index(H_pert)
        for epoch in range(surrogate_epochs):
            surrogate_model.train()
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
                    # _, _, acc_drop = classification_drop(args,target_model, H, HG, X, H_pert, X_pert, y)
                    logits_adv,_ = target_model(data)
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
            target_Z,_ = target_model(data)
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
        loss_meta = lap_dist + degree_penalty + alpha * loss_cls

        grads = torch.autograd.grad(loss_meta,[delta_H,delta_X])

        lap_dist_trajectory.append(lap_dist_val)
        loss_meta_trajectory.append(loss_meta.item())
        # _, _, acc_drop = classification_drop(args,target_model, H, HG, X, H_pert, X_pert, y)
        with torch.no_grad():
            target_model.eval()
            # _, _, acc_drop = classification_drop(args,target_model, H, HG, X, H_pert, X_pert, y)
            logits_adv,_ = target_model(data)
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
            bad_edges = (col_degrees < 2).nonzero(as_tuple=True)[0]
            delta_H_new[bad_nodes, :] = 0
            delta_H_new[:, bad_edges] = 0
            delta_H.copy_(delta_H_new)

        delta_X = delta_X.clamp(-epsilon, epsilon)
        time2 = time.time() - runtime_start2
    # results = [(t, loss_meta, acc_drop, lap_shift, deg_penalty, cls_loss, lap_dist, feature_shift)]
    results = [loss_meta_trajectory, acc_drop_trajectory, lap_shift_trajectory, lap_dist_trajectory, cls_loss_trajectory, \
               deg_penalty_trajectory,feature_shift_trajectory,surrogate_test_trajectory, target_test_trajectory]
    # mask = filter_potential_singletons(torch.clamp(H + delta_H, 0, 1))
    return torch.clamp(H + delta_H, 0, 1), X + delta_X, results, time1+time2, best_model_state

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

def perturb_hyperedges(data, prop):
    data_p = data
    edge_index = data_p.edge_index
    num_node = data.x.shape[0]
    e_idxs = edge_index[1,:] - num_node
    edges = (edge_index[1,:].max()) - (edge_index[1,:].min())
    p_num = ((edge_index[1,:].max()) - (edge_index[1,:].min())) * prop 
    p_num = int(p_num)
    chosen_edges = torch.as_tensor(np.random.permutation(int(edges.numpy()))).to(edge_index.device)
    chosen_edges = chosen_edges[:p_num]
    for i in range(chosen_edges.shape[0]):
        chosen_edge = chosen_edges[i]
        edge_index = edge_index[:, (e_idxs != chosen_edge)]
        e_idxs = e_idxs[(e_idxs != chosen_edge)]
    edge_idxs = [edge_index]
    for i in range(chosen_edges.shape[0]):
        new_edge = torch.as_tensor(np.random.choice(int(num_node), 5, replace=False)).to(edge_index.device)
        for j in range(new_edge.shape[0]):
            edge_idx_i = torch.zeros([2,1]).to(edge_index.device)
            edge_idx_i[0,0] = new_edge[j]
            edge_idx_i[1,0] = chosen_edges[i] + num_node
            edge_idxs.append(edge_idx_i)
    edge_idxs = torch.cat(edge_idxs, dim=1)
    data_p.edge_index = edge_idxs.long()
    return data_p


def parse_method(args):
    model = hyperMLP_model(args)
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
def evaluate(model, data, split_idx, eval_func, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out, _ = model(data)
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

def extract_node_hyperedge_indices(edge_index, num_nodes):
    """
    Extract (node, hyperedge) pairs from a symmetric bipartite edge_index.
    Assumes hyperedges have indices offset by +num_nodes.
    """
    src, dst = edge_index
    node_mask = src < num_nodes
    hyperedge_mask = dst >= num_nodes

    valid_mask = node_mask & hyperedge_mask
    n_idxs = src[valid_mask]
    e_idxs = dst[valid_mask] - num_nodes  # map hyperedge index back to [0, num_hyperedges)

    return n_idxs, e_idxs

def smooth_loss(node_feature, n_idxs, e_idxs, edge_index, epoch):
    loss = 0.0
    edges = int(edge_index[1,:].max() - edge_index[1,:].min())
    flag = False
    if(edges > 3000):
        setup_seed(epoch)
        chosen_edges = torch.as_tensor(np.random.permutation(edges)).to(node_feature.device)
        edges = 2000
        chosen_edges = chosen_edges[:edges]
        flag = True
    count = 0
    for i in range(edges):
        idx = chosen_edges[i] if flag else i
        temp_n = n_idxs[(e_idxs == idx)]
        
        # Skip if less than 2 nodes in the hyperedge
        if temp_n.numel() < 2:
            continue
        
        node_featur_i = node_feature[temp_n, :]
        z_i = torch.cdist(node_featur_i, node_featur_i, compute_mode='donot_use_mm_for_euclid_dist')
        loss_i = torch.max(z_i)
        loss += loss_i
        count += 1

    return loss / count if count > 0 else torch.tensor(0.0, device=node_feature.device)

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True 
    if torch.cuda.is_available():
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
    parser.add_argument('--method', default='AllSetTransformer')
    parser.add_argument('--epochs', default=500, type=int)
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
    parser.add_argument('--Classifier_num_layers', default=1,
                        type=int)  # How many layers of decoder
    parser.add_argument('--Classifier_hidden', default=256,
                        type=int)  # Decoder hidden units
    parser.add_argument('--display_step', type=int, default=-1)
    parser.add_argument('--alpha', type=float, default=1)
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
    parser.add_argument('--perturb_type', default='toxic', type=str)
    parser.add_argument('--perturb_prop', default=0.0, type=float)
    parser.add_argument('--feature_noise', default='1', type=str)
    parser.add_argument('--sth_type', default='max_s', type=str)
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
    parser.add_argument('--seed', type=int, default=1000, help='Random seed.')
    parser.add_argument('--attack', type=str, default='mla', \
                    choices=['mla','Rand-flip', 'Rand-feat','gradargmax','mla_fgsm'], help='model variant')
    parser.add_argument('--epsilon', type=float, default=0.05, help='Node Feature perturbation bound')
    parser.add_argument('--ptb_rate', type=float, default=0.2,  help='pertubation rate')
    parser.add_argument('--patience', type=int, default=150,
                    help='Patience for training with early stopping.')
    parser.add_argument('--T', type=int, default=80, help='Number of iterations for the attack.')
    parser.add_argument('--mla_alpha', type=float, default=4.0, help='weight for classification loss')
    parser.add_argument('--eta_H', type=float, default=1e-2, help='Learning rate for H perturbation')
    parser.add_argument('--eta_X', type=float, default=1e-2, help='Learning rate for X perturbation')
    parser.add_argument('--num_epochs_sur', type=int, default=80, help='#epochs for the surrogate training.')

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
    print(args)
    # # Part 1: Load data
    root2 = './hypergraphMLP_Melad'
    root='./hypergraphMLP'
    # root='./'+args.attack+'_hypergraphMLP_final2'
    os.makedirs(root, exist_ok=True)
    save = True
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
            dataset = dataset_Hypergraph(name=dname, root = '../data/pyg_data/hypergraph_dataset_updated/',
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
    data = ExtractV2E(data)
    setup_seed(args.seed)
    if((args.perturb_type == 'toxic') and (args.perturb_prop > 0)):
        data = perturb_hyperedges(data, args.perturb_prop)
    if args.dname == 'cora':
        dataset = 'co-cora'
    elif args.dname == 'citeseer':
        dataset = 'co-citeseer'
    elif args.dname == 'coauthor_cora':
        dataset = 'coauth_cora' 
    elif args.dname == 'coauthor_dblp':
        dataset = 'coauth_dblp'
    elif args.dname == '20newsW100':
        dataset = "news20"
    elif args.dname == 'house-committees-100':
        dataset = "house"
    else:
        dataset = args.dname
    # else:
    #     raise ValueError('dataset not supported')
    args.__setattr__('dataset',dataset)
    if dataset not in ['cora','citeseer','coauthor_cora']:
        split_idx = rand_train_test_idx(data.y)
        train_mask, val_mask, test_mask = split_idx['train'], split_idx['valid'], split_idx['test']
    else:
        _, _, _, train_mask, val_mask, test_mask = get_dataset(args, device='cpu')
    print(sum(train_mask)*100/len(train_mask))
    # # Part 2: Load model
    
    
    model = parse_method(args)
    # put things to device
    if args.cuda in [0,1,2,3,4,5,6,7]:
        device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    model, data = model.to(device), data.to(device)
    
    num_params = count_parameters(model)
    
    # # Part 3: Main. Training + Evaluation
    
    
    logger = Logger(args.runs, args)
    
    criterion = nn.NLLLoss()
    eval_func = eval_acc
    
    model.train()
    # print('MODEL:', model)
    
    ### Training loop ###
    edge_index = data.edge_index
    
    n_idxs = edge_index[0,:] - edge_index[0,:].min()
    e_idxs = edge_index[1,:] - edge_index[1,:].min()
    x = data.x
    
    train_acc_tensor = torch.zeros((args.runs, args.epochs))
    val_acc_tensor = torch.zeros((args.runs, args.epochs))
    test_acc_tensor = torch.zeros((args.runs, args.epochs))
    smooth_loss_tensor = torch.zeros((args.runs, args.epochs))
    assert args.runs == 1, "Only one run is supported for now"
    for run in range(args.runs):
        setup_seed(args.seed)
        # split_idx = split_idx_lst[run]
        split_idx = {'train': train_mask, 'valid': val_mask, 'test': test_mask}
        train_idx = split_idx['train'].to(device)
        model.reset_parameters()
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
            out, x = model(data)
            out = F.log_softmax(out, dim=1)
            loss_sth = 0
            if(args.alpha > 0):
                loss_sth = smooth_loss(x, n_idxs, e_idxs, edge_index, epoch) * args.alpha
            loss_cls = criterion(out[train_idx], data.y[train_idx])
            loss = loss_cls + loss_sth
            loss.backward(retain_graph=True)
            optimizer.step()
    #         if args.method == 'HNHN':
    #             scheduler.step()
    #         Evaluation part
            result = evaluate(model, data, split_idx, eval_func)
            logger.add_result(run, result[:3])
            train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out = result
            train_acc_tensor[run, epoch] = result[0]
            val_acc_tensor[run, epoch] = result[1]
            test_acc_tensor[run, epoch] = result[2]
            smooth_loss_tensor[run, epoch] = loss_sth
            if(args.alpha > 0):
                smooth_loss_tensor[run, epoch] = loss_sth.cpu()
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
                      f'Total Train Loss: {loss:.4f}, '
                      f'Smooth Train Loss: {loss_sth:.4f}, '
                      f'Cls Train Loss: {loss_cls:.4f}, '
                      f'Valid Loss: {result[4]:.4f}, '
                      f'Test  Loss: {result[5]:.4f}, '
                      f'Train Acc: {100 * result[0]:.2f}%, '
                      f'Valid Acc: {100 * result[1]:.2f}%, '
                      f'Test  Acc: {100 * result[2]:.2f}%')
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        # logger.print_statistics(run)
    # out, _ = model(data)
    # Z_orig = out.detach()
    # print("Z_orig shape: ",Z_orig.shape)
    ### Save results ###
    # best_val, best_test = logger.print_statistics()
    # res_root = 'results'
    # if not osp.isdir(res_root):
    #     os.makedirs(res_root)
    # res_root = '{}/layer_{}'.format(res_root, args.All_num_layers)
    # if not osp.isdir(res_root):
    #     os.makedirs(res_root)
    # res_root = '{}/{}'.format(res_root, args.method)
    # if not osp.isdir(res_root):
    #     os.makedirs(res_root)

    # filename = f'{res_root}/{args.dname}_noise_{args.feature_noise}.csv'
    # print(f"Saving results to {filename}")
    # with open(filename, 'a+') as write_obj:
    #     cur_line = f'{args.method}_{args.lr}_{args.wd}_{args.alpha}\n'
    #     cur_line += f'{args.perturb_type}_{args.perturb_prop}\n'
    #     cur_line += f',{best_val.mean():.3f} ± {best_val.std():.3f}\n'
    #     cur_line += f',{best_test.mean():.3f} ± {best_test.std():.3f}\n'
    #     cur_line += f',{num_params}\n'
    #     cur_line += f'\n'
    #     write_obj.write(cur_line)

    # all_args_file = f'{res_root}/all_args_{args.dname}_noise_{args.feature_noise}.csv'
    # with open(all_args_file, 'a+') as f:
    #     f.write(str(args))
    #     f.write('\n')
    
    # res_root_2 = './storage'
    # if not osp.isdir(res_root_2):
    #     os.makedirs(res_root_2)
    # filename = f'{res_root_2}/{args.dname}_{args.feature_noise}_noise_{args.alpha}_alpha.pickle'
    # res_data = {
    #     'train_acc_tensor': train_acc_tensor,
    #     'val_acc_tensor': val_acc_tensor,
    #     'test_acc_tensor': test_acc_tensor,
    #     'smooth_loss_tensor': smooth_loss_tensor
    # }
    # with open(filename, 'wb') as handle:
    #     pickle.dump(res_data, handle, protocol=4)    
        
    # print('All done! Exit python code')
    # quit()
    H = torch.Tensor(ConstructH(data)).to(device)
    X = data.x.to(device)
    n = data.n_x 
    e = data.num_hyperedges
    if type(e) == np.int32:
        e = int(e)
    if type(n) == np.int32:
        n = int(n)
    y = data.y.to(device)
    
    # print('train_mask:', train_mask.sum()/len(train_mask))
    perturbations = int(args.ptb_rate * e)
    args.__setattr__('model', args.method)
    if args.attack == 'mla':
        H_adv, X_adv, results, exec_time, robust_model_states = meta_laplacian_pois_attack(root, H, X, y, data, None, None, model, \
                        train_mask, val_mask, test_mask, Z_orig, budget=perturbations, epsilon=args.epsilon, T=args.T, \
                        eta_H=args.eta_H, eta_X=args.eta_X, alpha=args.mla_alpha, \
                        reinit_if_stuck=True)
        save_npz(root, args.seed, results)
        H_adv = H_adv.detach()
        X_adv = X_adv.detach()
        X_adv.requires_grad = False

    elif args.attack == 'mla_fgsm':
        H_adv, X_adv, results, exec_time, robust_model_states = meta_laplacian_FGSM(H, X, y, data, None, None, model, \
                        train_mask, val_mask, test_mask, Z_orig, budget=perturbations, epsilon=args.epsilon, T=args.T, \
                        eta_H=args.eta_H, eta_X=args.eta_X, alpha=args.mla_alpha, \
                        reinit_if_stuck=True)

    else:
        H_adv, X_adv, exec_time = get_attack(model, H, X, y,data, None, train_mask,val_mask,test_mask,perturbations = perturbations)
    data.x = X_adv
    data.edge_index = incidence_to_edge_index(H_adv)
    
    os.system('mkdir -p '+root2)
    np.savez(os.path.join(root2, args.model+"_"+args.attack+"_"+args.dataset+"_"+str(args.seed)+ '_H_adv.npz'), H_adv.clone().cpu().numpy())
    np.savez(os.path.join(root2, args.model+"_"+args.attack+"_"+args.dataset+"_"+str(args.seed)+ '_X_adv.npz'), X_adv.clone().cpu().numpy())
    torch.save(data,os.path.join(root2,args.model+"_"+args.attack+"_"+args.dataset+"_"+str(args.seed)+ "_data.pth"))
    # import sys 
    # sys.exit(1)
    # print('H_adv:', H_adv)
    if save and args.attack == 'mla':
        plot_results(args,results,root)
    # H_adv_HG = Hypergraph(n, incidence_matrix_to_edge_list(H_adv),device=device)
    evasion_dict = evasion_setting_evaluate_hyperMLP(args, H, X, y, data, Z_orig, H_adv, X_adv, None, model, \
                             model,train_mask,val_mask,test_mask)
    evasion_dict['exec_time'] = exec_time
    evasion_dict['num_edges'] = e
    evasion_dict['num_vertices'] = n
    evasion_dict['num_edges_perturbed'] = perturbations

    print(json.dumps(evasion_dict,indent = 4))
    # # print('H_adv - H:',torch.sum((H_adv-H).abs()))

    edge_index = data.edge_index
    n_idxs, e_idxs = extract_node_hyperedge_indices(edge_index, num_nodes=data.x.size(0))
    print('---------------- After attack ----------------:',args.attack,args.dataset,args.model)
    logger = Logger(args.runs, args)
    criterion = nn.NLLLoss()
    eval_func = eval_acc
    model = parse_method(args).to(device)
    data = data.to(device)
    model.train()
    train_acc_tensor = torch.zeros((args.runs, args.epochs))
    val_acc_tensor = torch.zeros((args.runs, args.epochs))
    test_acc_tensor = torch.zeros((args.runs, args.epochs))
    smooth_loss_tensor = torch.zeros((args.runs, args.epochs))
    for run in tqdm(range(args.runs)):
        setup_seed(args.seed)
        # split_idx = split_idx_lst[run]
        split_idx = {'train': train_mask, 'valid': val_mask, 'test': test_mask}
        train_idx = split_idx['train'].to(device)
        model.reset_parameters()
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
        for epoch in range(args.epochs):
            #         Training part
            model.train()
            optimizer.zero_grad()
            out, x = model(data)
            out = F.log_softmax(out, dim=1)
            loss_sth = 0
            if(args.alpha > 0):
                loss_sth = smooth_loss(x, n_idxs, e_idxs, edge_index, epoch) * args.alpha
            loss_cls = criterion(out[train_idx], data.y[train_idx])
            loss = loss_cls + loss_sth
            loss.backward(retain_graph=True)
            optimizer.step()
    #         if args.method == 'HNHN':
    #             scheduler.step()
    #         Evaluation part
            result = evaluate(model, data, split_idx, eval_func)
            logger.add_result(run, result[:3])
            train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out = result
            train_acc_tensor[run, epoch] = result[0]
            val_acc_tensor[run, epoch] = result[1]
            test_acc_tensor[run, epoch] = result[2]
            smooth_loss_tensor[run, epoch] = loss_sth
            if(args.alpha > 0):
                smooth_loss_tensor[run, epoch] = loss_sth.cpu()
            if valid_loss.item() < best_val_loss:
                best_val_loss = valid_loss.item()
                best_model_state = model.state_dict()
                patience_counter = 0
                Z_adv = out.detach()
            else:
                patience_counter += 1
                if patience_counter > patience:
                    print(f'Early stopping at epoch {epoch}.')
                    break
            
    
            if epoch % args.display_step == 0 and args.display_step > 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Total Train Loss: {loss:.4f}, '
                      f'Smooth Train Loss: {loss_sth:.4f}, '
                      f'Cls Train Loss: {loss_cls:.4f}, '
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
    results['num_edges_perturbed'] = perturbations
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
        print('Actual |H_adv - H|_0:', (H_adv - H).abs().sum(),' ptb: ',perturbations)
    print(json.dumps(results,indent=4))
    if save:
        results.update(vars(args))
        evasion_dict.update(vars(args))
        
        save_to_csv(evasion_dict,filename=os.path.join(root,'evasion_results.csv'))
        save_to_csv(results,filename=os.path.join(root,'pois_results.csv'))
    