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
# import sys
# sys.path.append('../../')
from mla_utils import *
from modelzoo import SimpleHGNN,GradArgmax


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
    model =  hyperMLP_model(args)
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

def parse_arg():
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
    parser.add_argument('--surr_class', default='MeLA-D+LinHGNN', choices=['MeLA-D+LinHGNN', 'MeLA-D+HyperMLP'])
    parser.add_argument('--mode',type=str,default='attack',choices=['attack','defense'])
    parser.set_defaults(PMA=True)  # True: Use PMA. False: Use Deepsets.
    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(exclude_self=False)
    parser.set_defaults(GPR=False)
    parser.set_defaults(LearnMask=False)
    parser.set_defaults(HyperGCN_mediators=True)
    parser.set_defaults(HyperGCN_fast=True)
    parser.set_defaults(HCHA_symdegnorm=False)
    return parser 
# --- Main part of the training ---
# # Part 0: Parse arguments


"""

"""
def train(model,data,split_idx, n_idxs,e_idxs,device):
    train_idx = split_idx['train'].to(device)
    criterion = nn.NLLLoss()
    eval_func = eval_acc
    edge_index = data.edge_index
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
        loss.backward()
        optimizer.step()
#         if args.method == 'HNHN':
#             scheduler.step()
#         Evaluation part
        result = evaluate(model, data, split_idx, eval_func)
        logger.add_result(0, result[:3])
        train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out = result
        train_acc_tensor[0, epoch] = result[0]
        val_acc_tensor[0, epoch] = result[1]
        test_acc_tensor[0, epoch] = result[2]
        smooth_loss_tensor[0, epoch] = loss_sth
        if(args.alpha > 0):
            smooth_loss_tensor[0, epoch] = loss_sth.cpu()
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
    return model, Z_orig

def meta_laplacian_adversarial_train(root, H, X, y, data, unseen_advdata, target_model, split_idx, 
                                     budget=20, epsilon=0.05, T=20, eta_H=1e-2, eta_X=1e-2, alpha=4.0):
    """
    Meta Laplacian Adversarial Training (Defense)
    
    Trains the target model to be robust against structure and feature perturbations
    generated via Laplacian-guided meta-gradients.

    Args:
        H: Original incidence matrix (n x m)
        X: Original node features (n x d)
        y: Node labels
        data: PyG data object
        HG: Not used here but kept for compatibility
        target_model: The model being trained for robustness
        train_mask, val_mask, test_mask: Masks for dataset splits
    Returns:
        Trained target model
    """
    device = X.device
    clean_model = deepcopy(target_model)
    train_mask, val_mask, test_mask = split_idx['train'].to(device), split_idx['valid'].to(device), split_idx['test'].to(device)
    n, m = H.shape
    H = H.clone().detach()
    X = X.clone().detach()
    L_orig = lap(H)
    dv_orig = H @ torch.ones((m,), device=device)
    
    delta_H = (1e-3 * torch.randn_like(H)).requires_grad_()
    delta_X = (1e-3 * torch.randn_like(X)).requires_grad_()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(target_model.parameters(), lr=args.lr, weight_decay=args.wd)
    cleanmodel_poisoned = []
    robustmodel_poisoned = []
    start_tm = time.time()
    for t in tqdm(range(T), desc="Adversarial Training Iteration"):
        # Apply perturbations
        H_pert = torch.clamp(H + delta_H, 0, 1)
        X_pert = (X + delta_X).clamp(-1, 1)
        L_pert = lap(H_pert)

        # Update data object
        data_adv = deepcopy(data).to(device)
        data_adv.edge_index = incidence_to_edge_index(H_pert)
        data_adv.x = X_pert
        # print(torch.norm(data_adv.x-X).item())
        # target_model.eval()
        # clean_model.eval()
        # with torch.no_grad():
        #     logits_r, _ = target_model(data_adv)
        #     robustmodel_poisoned.append(accuracy(logits_r[test_mask], y[test_mask]).item() * 100)
        #     logits_c, _ = clean_model(data_adv)
        #     cleanmodel_poisoned.append(accuracy(logits_c[test_mask], y[test_mask]).item() * 100)
        #     print(cleanmodel_poisoned[-1],' ',robustmodel_poisoned[-1])
        _, acc_robust, acc_vanilla, _ = evaluate_robustness(data,unseen_advdata,y,target_model,clean_model,split_idx)
        cleanmodel_poisoned.append(acc_vanilla)
        robustmodel_poisoned.append(acc_robust)
        # Train target model on perturbed data
        for epoch in range(args.num_epochs_sur):
            target_model.train()
            optimizer.zero_grad()
            logits, _ = target_model(data_adv)
            cls_loss = criterion(logits[train_mask], y[train_mask])
            cls_loss.backward(retain_graph = True)
            optimizer.step()
            # if epoch % 100 == 0:
            #     result = evaluate(target_model, data_adv, split_idx, eval_acc)     
            #     print(f'Epoch: {epoch:02d}, '
            #         f'Cls Train Loss: {cls_loss:.4f}, '
            #         f'Valid Loss: {result[4]:.4f}, '
            #         f'Test  Loss: {result[5]:.4f}, '
            #         f'Train Acc: {100 * result[0]:.2f}%, '
            #         f'Valid Acc: {100 * result[1]:.2f}%, '
            #         f'Test  Acc: {100 * result[2]:.2f}%')      

        # Evaluate model on clean data
        target_model.eval()
        with torch.no_grad():
            logits_clean, _ = target_model(data)
        #     # logits_adv, _ = target_model(data_adv)
        #     acc_val = accuracy(logits_clean[val_mask], y[val_mask]).item()*100
        #     acc_test = accuracy(logits_clean[test_mask], y[test_mask]).item()*100
            # print('On Clean: acc_val: ',acc_val,' acc_test: ',acc_test)

        # Recompute forward pass on perturbed data *with gradients*
        target_model.train()  # Ensure gradients are tracked
        logits_adv, _ = target_model(data_adv)

        # Compute meta loss
        Z = logits_clean.detach()
        delta_L = (L_pert - L_orig) @ Z
        lap_dist = torch.norm(delta_L, p=2).mean()
        
        H_temp = H_pert.detach()
        dv_temp = H_temp @ torch.ones((m,), device=device)
        degree_penalty = ((dv_temp - dv_orig) ** 2).mean()

        cls_loss = F.cross_entropy(logits_adv[train_mask], y[train_mask])
        loss_meta = lap_dist + alpha * cls_loss + degree_penalty

        # Compute gradients for perturbations
        grads = torch.autograd.grad(loss_meta, [delta_H, delta_X])

        # Gradient ascent to update delta_H and delta_X
        with torch.no_grad():
            delta_H += eta_H * grads[0].sign()
            delta_X += eta_X * grads[1].sign()

            # Project delta_H to respect sparsity budget
            flat = delta_H.abs().flatten()
            topk = torch.topk(flat, k=min(budget, delta_H.numel())).indices
            delta_H_new = torch.zeros_like(delta_H)
            delta_H_new.view(-1)[topk] = delta_H.view(-1)[topk]

            # Remove bad nodes/edges
            H_temp = torch.clamp(H + delta_H_new, 0, 1)
            row_degrees = H_temp.sum(dim=1)
            col_degrees = H_temp.sum(dim=0)
            bad_nodes = (row_degrees < 1).nonzero(as_tuple=True)[0]
            bad_edges = (col_degrees < 2).nonzero(as_tuple=True)[0]
            delta_H_new[bad_nodes, :] = 0
            delta_H_new[:, bad_edges] = 0
            delta_H.copy_(delta_H_new)

            # Clamp feature perturbation
        delta_X = delta_X.clamp(-epsilon, epsilon)
        
    tm = time.time() - start_tm 
        # print(f"Iter {t:02d} | Val Acc: {acc_val:.3f} | Test Acc: {acc_test:.3f} | Meta Loss: {loss_meta.item():.4f}")

    return target_model, tm, cleanmodel_poisoned, robustmodel_poisoned

def evaluate_robustness(data_clean, data_adv, y, robust_model, vanilla_model, split_idx):
    device = data_adv.x.device

    train_mask, val_mask, test_mask = (
        split_idx["train"].to(device),
        split_idx["valid"].to(device),
        split_idx["test"].to(device),
    )

        # --- On CLEAN graph ---
    robust_model.eval()
    # vanilla_model.eval()
    with torch.no_grad():
        logits_r_clean, _ = robust_model(data_clean)
        # logits_v_clean, _ = vanilla_model(data_clean)

    # -- ON adv ---
    acc_robust_clean = accuracy(logits_r_clean[test_mask], y[test_mask]).item() * 100
    # acc_vanilla_clean = accuracy(logits_v_clean[test_mask], y[test_mask]).item() * 100

    # Eval on robust model
    robust_model.eval()
    with torch.no_grad():
        logits_r, _ = robust_model(data_adv)
    acc_robust = accuracy(logits_r[test_mask], y[test_mask]).item() * 100

    # Eval on vanilla model
    vanilla_model.eval()
    with torch.no_grad():
        logits_v, _ = vanilla_model(data_adv)
    acc_vanilla = accuracy(logits_v[test_mask], y[test_mask]).item() * 100

    # print(" Robust Model Accuracy on Clean Graph: {:.2f}%".format(acc_robust_clean))
    # print(" Robust Model Accuracy on Perturbed Graph: {:.2f}%".format(acc_robust))
    # print(" Vanilla Model Accuracy on Perturbed Graph: {:.2f}%".format(acc_vanilla))
    # print(" Accuracy Gain: {:.2f}%".format(acc_robust - acc_vanilla))

    return acc_robust_clean, acc_robust, acc_vanilla, acc_robust - acc_vanilla

if __name__ == '__main__':
    parser = parse_arg()
    #     Use the line below for .py file
    args = parser.parse_args()
    print(args)
    # # Part 1: Load data
    
    # root='./hypergraphMLP_newsplit'
    root='./hypergraphMLP_Melad'
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
    if args.cuda in [0,1,2,3,4,5,6,7]:
        device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
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
    else:
        dataset = args.dname
    args.__setattr__('dataset',dataset)
    split_idx = rand_train_test_idx(data.y)
    train_mask, val_mask, test_mask = split_idx['train'], split_idx['valid'], split_idx['test']
    print(sum(train_mask)*100/len(train_mask))
    # # Part 2: Load model
    
    
    model = parse_method(args)
    # put things to device

    
    model, data = model.to(device), data.to(device)
    data_copy = data.clone()
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
    # assert args.runs == 1, "Only one run is supported for now"
    # for run in range(args.runs):
    setup_seed(args.seed)
    # split_idx = split_idx_lst[run]
    split_idx = {'train': train_mask, 'valid': val_mask, 'test': test_mask}
    model, Z_orig = train(model,data,split_idx, n_idxs,e_idxs,device)
    # logger.print_statistics(run)

    ### Save results ###
    from copy import deepcopy
    clean_model = deepcopy(model)
    best_val, best_test = logger.print_statistics()
    H = torch.Tensor(ConstructH(data)).to(device)
    X = data.x.to(device)
    n = data.n_x 
    e = data.num_hyperedges
    y = data.y.to(device)
    
    # print('train_mask:', train_mask.sum()/len(train_mask))
    perturbations = int(args.ptb_rate * e)
    args.__setattr__('model', args.method)
    # Load H_adv, X_adv 
    # try:
    H_adv = np.load(os.path.join(root, args.model+"_"+args.attack+"_"+args.dataset+"_"+str(args.seed)+ '_H_adv.npz'))['arr_0']
    H_adv = torch.from_numpy(H_adv).to(device)
    X_adv = np.load(os.path.join(root, args.model+"_"+args.attack+"_"+args.dataset+"_"+str(args.seed)+ '_X_adv.npz'))['arr_0']
    X_adv = torch.from_numpy(X_adv).to(device)
    path = os.path.join(root,args.model+"_"+args.attack+"_"+args.dataset+"_"+str(args.seed)+ "_data.pth")
    # print(path)
    data_adv = torch.load(path,weights_only=False).to(device)
    # except Exception as e:
    #     print("Error loading adversarial data: ",e)
    #     import sys 
    #     sys.exit(1)
    robust_model,tm, cleanmodel_poisoned, robustmodel_poisoned = \
        meta_laplacian_adversarial_train(root, H, X, y, data, data_adv, deepcopy(model), \
                split_idx, budget=perturbations, epsilon=args.epsilon, T=args.T, \
                eta_H=args.eta_H, eta_X=args.eta_X, alpha=args.mla_alpha)
    # torch.save(robust_model.state_dict(), 'robust_model.pth')
    acc_robust_clean, acc_robust, acc_vanilla, acc_gain = \
        evaluate_robustness(data, data_adv, y, robust_model, clean_model, split_idx)
    args.__setattr__('clean_acc', acc_robust_clean)
    args.__setattr__('adv_acc_rob', acc_robust)
    args.__setattr__('adv_acc_base', acc_vanilla)
    args.__setattr__('rob_gain', acc_gain)
    args.__setattr__('time',tm)
    res = vars(args)
    save_to_csv(res,filename=os.path.join(root,'adv_results.csv'))
    cleanmodel_poisoned = np.array(cleanmodel_poisoned)
    robustmodel_poisoned = np.array(robustmodel_poisoned)
    # from matplotlib import pyplot as plt
    # plt.plot(cleanmodel_poisoned,label='clean model')
    # plt.plot(robustmodel_poisoned,label='robust model')
    # plt.legen()
    # plt.show()
    # Save accuracy trajectories.  
    np.savez(os.path.join(root, args.model+"_"+args.attack+"_"+args.dataset+"_"+str(args.seed)+ '_clean_pois.npz'),cleanmodel_poisoned)
    np.savez(os.path.join(root, args.model+"_"+args.attack+"_"+args.dataset+"_"+str(args.seed)+ '_robust_pois.npz'),robustmodel_poisoned)
    
    
