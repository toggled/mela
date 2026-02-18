import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm 
import os
import pandas as pd 
def plot_results(args,results,root,plot_acc_traj=True):
    loss_meta_trajectory, acc_drop_trajectory, lap_shift_trajectory, lap_dist_trajectory, \
        cls_loss_trajectory, deg_penalty_trajectory,feature_shift_trajectory,surrogate_test_trajectory,\
             target_test_trajectory = results 
    os.makedirs(os.path.join(root,str(args.seed)), exist_ok=True)
    prefix = os.path.join(root,str(args.seed), args.dataset+'_'+args.model+"_"+args.attack+'_'+str(args.ptb_rate))
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(loss_meta_trajectory)
    plt.title('Meta Loss over Iterations')
    plt.xlabel('Iteration')

    plt.subplot(2, 2, 2)
    plt.plot(acc_drop_trajectory)
    plt.title('Accuracy Drop over Iterations')
    plt.xlabel('Iteration')

    plt.subplot(2, 2, 3)
    plt.plot(lap_shift_trajectory)
    plt.title('Laplacian Shift over Iterations')
    plt.xlabel('Iteration')
    plt.subplot(2, 2, 4)
    plt.plot(feature_shift_trajectory)
    plt.title('Feature Shift (L2) over Iterations')
    plt.xlabel('Iteration')
    plt.tight_layout()
    plt.savefig(prefix+'_laplacian_feature_shift.png')

    # Plot individual loss components
    plt.figure(figsize=(10, 4))
    plt.yscale('log')
    plt.plot(lap_dist_trajectory, label='Laplacian Loss')
    plt.plot(cls_loss_trajectory, label='Classification Loss')
    plt.plot(deg_penalty_trajectory, label='Degree Penalty')
    plt.title('Meta Loss Components over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(prefix+'_loss_components.png')
    
    # Plot the accuracy trajectories for both models: surrogate and target during attackers iteration
    if plot_acc_traj:
        # Plot the accuracy trajectories for both models: surrogate and target during attackers iteration
        epochs = range(1, args.T + 1)
        # Create the plot
        plt.figure(figsize=(10, 4))
        # Plot the accuracy trajectories for both models
        print(len(surrogate_test_trajectory),len(target_test_trajectory),len(epochs))
        plt.plot(epochs, surrogate_test_trajectory, label='Surrogate Model', color='blue', marker='o')
        plt.plot(epochs, target_test_trajectory, label='Target Model', color='red', marker='x')
        # Add labels and title
        plt.xlabel('Attack Epoch (t)')
        plt.ylabel('Accuracy (Test Set)')
        # plt.title('Epoch-by-Epoch Test Set Accuracy Comparison')
        plt.legend()
        plt.savefig(prefix+'_accuracy_trajectory.png')


def embedding_shift(Z1, Z2):
    return torch.norm(Z1 - Z2).item()

def lap(H):
    de = H.sum(dim=0).clamp(min=1e-6)
    De_inv = torch.diag(1.0 / de)
    dv = H @ torch.ones(H.shape[1], device=H.device)
    Dv_inv_sqrt = torch.diag(1.0 / dv.clamp(min=1e-6).sqrt())
    return torch.eye(H.shape[0], device=H.device) - Dv_inv_sqrt @ H @ De_inv @ H.T @ Dv_inv_sqrt
# Measure the Frobenius norm difference between Laplacians of original and perturbed incidence matrices
# This evaluates the impact of perturbations on the hypergraph structure
def laplacian_diff(H1, H2):
    return torch.norm(lap(H1) - lap(H2)).item()

# Visualize original and adversarial embeddings using t-SNE for interpretability
# Helps detect how much the attack shifted the embedding geometry
def visualize_tsne(args, root, Z1, Z2, title):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 1, 1)
    Z = torch.cat([Z1, Z2], dim=0).cpu().numpy()
    tsne = TSNE(n_components=2)
    Z_2d = tsne.fit_transform(Z)
    plt.scatter(Z_2d[:Z1.shape[0], 0], Z_2d[:Z1.shape[0], 1], c='k', label='original', alpha=0.6, s = 2)
    plt.scatter(Z_2d[Z1.shape[0]:, 0], Z_2d[Z1.shape[0]:, 1], c='red', label='adversarial', alpha=0.6, s = 2)
    plt.legend()
    # plt.title(title)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(root,args.dataset+'_'+args.model+'_'+str(args.ptb_rate)+'_tsne.png'))

# Quantify how stealthy the attack is based on L0 structural change and L∞ feature deviation
def measure_stealthiness(H, H_adv, X, X_adv):
    h_l0 = torch.sum((H - H_adv).abs() > 1e-6).item()
    x_delta = torch.norm((X - X_adv), p=float('inf')).item()

    degree_orig = H.sum(dim=1)
    degree_adv = H_adv.sum(dim=1)
    deg_shift_inf = torch.norm(degree_orig - degree_adv, p=float('inf')).item()
    deg_shift_l1 = torch.norm(degree_orig - degree_adv, p=1).item()
    deg_shift_l2 = ((degree_orig - degree_adv)**2).mean().item()

    edge_card_orig = H.sum(dim=0)
    edge_card_adv = H_adv.sum(dim=0)
    edge_card_shift_inf = torch.norm(edge_card_orig - edge_card_adv, p=float('inf')).item()
    edge_card_shift_l1 = torch.norm(edge_card_orig - edge_card_adv, p=1).item()
    edge_card_shift_l2 = ((edge_card_orig - edge_card_adv)**2).mean().item()

    return h_l0, x_delta, deg_shift_l1, edge_card_shift_l1,deg_shift_l2, edge_card_shift_l2, deg_shift_inf, edge_card_shift_inf

# Evaluate how well the attack generalizes across different models
# Returns the output logits of each model when run on the adversarially perturbed inputs
def evaluate_transferability(H_adv, X_adv, model_list):
    return [model(X_adv,H_adv).detach() for model in model_list]

# Check how much semantic meaning of node features has changed
# Measured via average cosine similarity between original and perturbed features
def semantic_feature_change(X, X_adv):
    cosine = F.cosine_similarity(X, X_adv, dim=1)
    return 1.0 - cosine.mean().item()

@torch.no_grad()
def classification_drop_pois(model, model_pois, H, X, H_adv, X_adv, labels, W_e = None):
    if W_e is None:
        logits_orig = model(X, H)
        logits_adv = model_pois(X_adv,H_adv)
    else:
        logits_orig = model(X, H, W_e)
        logits_adv = model_pois(X_adv,H_adv,W_e)
    acc_orig = (logits_orig.argmax(dim=1) == labels).float().mean().item()
    acc_adv = (logits_adv.argmax(dim=1) == labels).float().mean().item()
    return acc_orig, acc_adv, (acc_orig - acc_adv)/acc_orig

def save_npz(root, seed, results):
    root = os.path.join(root, str(seed))
    os.makedirs(root,exist_ok=True)
    loss_meta_trajectory, acc_drop_trajectory, lap_shift_trajectory, lap_dist_trajectory, cls_loss_trajectory, \
               deg_penalty_trajectory,feature_shift_trajectory,surrogate_test_trajectory, target_test_trajectory = results
    np.savez(os.path.join(root, 'loss_meta_trajectory.npz'),loss_meta_trajectory)
    np.savez(os.path.join(root, 'acc_drop_trajectory.npz'), acc_drop_trajectory)
    np.savez(os.path.join(root, 'lap_shift_trajectory.npz'), lap_shift_trajectory)
    np.savez(os.path.join(root, 'lap_dist_trajectory.npz'), lap_dist_trajectory)
    np.savez(os.path.join(root, 'cls_loss_trajectory.npz'), cls_loss_trajectory)
    np.savez(os.path.join(root, 'deg_penalty_trajectory.npz'), deg_penalty_trajectory)
    np.savez(os.path.join(root, 'feature_shift_trajectory.npz'), feature_shift_trajectory)
    np.savez(os.path.join(root, 'surrogate_test_trajectory.npz'), surrogate_test_trajectory)
    np.savez(os.path.join(root, 'target_test_trajectory.npz'), target_test_trajectory)

def save_to_csv(results, filename='results.csv'):
    print('saving csv file: ',filename)
    # Convert results into a DataFrame for better handling
    df = pd.DataFrame(results, columns=sorted(list(results.keys())), index=[0])
    # print(df.columns) 
    # Append to the CSV file, creating a new one if it doesn't exist
    if os.path.isfile(filename):
        existing_df = pd.read_csv(filename)
        if not existing_df.columns.equals(df.columns):
            # print('Existing columns:', existing_df.columns)
            raise ValueError("Column mismatch between results and existing CSV file.")
    df.to_csv(filename, mode='a', header=not os.path.isfile(filename), index=False)

def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def compute_statistics(H,H_adv,Z_orig,Z_adv,X,X_adv,train_mask,val_mask,test_mask,y):
    results = {}
     # Laplacian Frobenius norm change
    results['laplacian_norm'] = laplacian_diff(H, H_adv)

    # Embedding shift (ΔZ Fro norm)
    results['embedding_shift'] = embedding_shift(Z_orig, Z_adv)

    # Stealthiness measures
    h_l0, x_linf, deg_shift_l1, edge_card_shift_l1, deg_shift_l2, edge_card_shift_l2, deg_shift_linf, edge_card_shift_linf = measure_stealthiness(H, H_adv, X, X_adv)
    results.update({
        'h_l0': h_l0,
        'x_linf': x_linf,
        'deg_shift_l1': deg_shift_l1,
        'edge_card_shift_l1': edge_card_shift_l1,
        'deg_shift_l2': deg_shift_l2,
        'edge_card_shift_l2': edge_card_shift_l2,
        'deg_shift_linf': deg_shift_linf,
        'edge_card_shift_linf': edge_card_shift_linf
    })
    # Semantic change in features
    results['semantic_change'] = semantic_feature_change(X, X_adv)

    # Classification accuracy before and after attack
    labels = y
    logits_orig = Z_orig
    logits_adv = Z_adv

    acc_orig_test = accuracy(logits_orig[test_mask], labels[test_mask])
    acc_adv_test = accuracy(logits_adv[test_mask], labels[test_mask])
    acc_orig_val = accuracy(logits_orig[val_mask], labels[val_mask])
    acc_adv_val = accuracy(logits_adv[val_mask], labels[val_mask])
    acc_orig_train = accuracy(logits_orig[train_mask], labels[train_mask])
    acc_adv_train = accuracy(logits_adv[train_mask], labels[train_mask])

    acc_dict = {
        'clean_train': acc_orig_train.item(), 'clean_test': acc_orig_test.item(), 'clean_val': acc_orig_val.item(),
        'adv_train': acc_adv_train.item(), 'adv_test': acc_adv_test.item(), 'adv_val': acc_adv_val.item()}
    results.update(acc_dict)

    acc_drop = (acc_dict['clean_test'] - acc_dict['adv_test']) / acc_dict['clean_test']
    results['acc_drop%'] = acc_drop * 100
    return results 

def evasion_setting_evaluate(args, H, X, y, Z_orig, H_adv, X_adv, H_adv_HG, model, poisoned_model,train_mask,val_mask,test_mask, draw=False):
    print('================ Evasion setting =================')
    if args.model in ['hypergcn']:
        Z_adv = poisoned_model(X_adv, H_adv_HG).detach()
    else:
        Z_adv = poisoned_model(X_adv, H_adv).detach()
    
    results = compute_statistics(H,H_adv,Z_orig,Z_adv,X,X_adv,train_mask,val_mask,test_mask,y)
    if args.verbose: 
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
    
    # Return the results dictionary
    return results

def evasion_setting_evaluate_hyperMLP(args, H, X, y, data, Z_orig, H_adv, X_adv, H_adv_HG, model, poisoned_model,train_mask,val_mask,test_mask, draw=False, verbose = False):
    print('================ Evasion setting =================')
    if args.model in ['hypergcn']:
        raise ValueError('HyperMLP does not support hypergcn model.')
    else:
        poisoned_model.eval()
        with torch.no_grad():
            Z_adv, _ = poisoned_model(data)
            Z_adv = Z_adv.detach()
            print('Z_adv.shape: ',Z_adv.shape)
    
    results = compute_statistics(H,H_adv,Z_orig,Z_adv,X,X_adv,train_mask,val_mask,test_mask,y)
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
    
    # Return the results dictionary
    return results

def canonical_boundary_matrix(H_bin):
    B1 = torch.zeros_like(H_bin)  # Initialize B1 with zeros (same shape as H_bin)
    
    for j in range(H_bin.shape[1]):  # Iterate over each hyperedge (columns of H_bin)
        nodes = torch.where(H_bin[:, j] > 0)[0]  # Get indices of nodes in the j-th hyperedge
        
        if len(nodes) >= 2:  # Only consider hyperedges with more than one node
            min_idx = nodes[0]  # The first node in the sorted list (smallest index)
            for i in nodes[1:]:  # Iterate over the remaining nodes
                B1[i, j] = 1  # Assign positive orientation to nodes after the first one
            B1[min_idx, j] = -1  # Assign negative orientation to the first node in sorted order
            
    return B1

def hodge_laplacian_L0(B1):
    return B1 @ B1.T