#!/bin/bash
# ulimit -v $((100 * 1024 * 1024))  # 100 GB in KB
dataset="cora"
# Define two arrays
t=0.25
seeds=(11 12)
# for j in "${seeds[@]}"; do
#     python ablation.py --method HGNN --dname $dataset \
#     --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 \
#     --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 \
#     --seed $j --attack mla --epsilon 0 --cuda 0 --train_prop $t & # All 3 coefficients 
#     python ablation.py --method HGNN --dname $dataset \
#     --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 \
#     --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 \
#     --seed $j --attack mla --alpha 0.0 --epsilon 0 --cuda 1 --train_prop $t
#     python ablation.py --method HGNN --dname $dataset \
#     --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 \
#     --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 \
#     --seed $j --attack mla --beta 0.0 --epsilon 0 --cuda 0 --train_prop $t & 
#     # python ablation.py --seed $j --epsilon 0.005  --num_epochs 1000 --eta_H 0.001 --eta_X 0.001 --T 30 --patience 150 --num_epochs_sur 50 --gamma 0
#     python ablation.py --method HGNN --dname $dataset \
#     --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 \
#     --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 \
#     --seed $j --attack mla --alpha 0.0 --beta 0.0 --epsilon 0 --cuda 1 --train_prop $t
#     # python ablation.py --seed $j --epsilon 0.005  --num_epochs 1000 --eta_H 0.001 --eta_X 0.001 --T 30 --patience 150 --num_epochs_sur 50 --alpha 0 --gamma 0
#     # python ablation.py --seed $j --epsilon 0.005  --num_epochs 1000 --eta_H 0.001 --eta_X 0.001 --T 30 --patience 150 --num_epochs_sur 50 --beta 1 --gamma 1
# done
for j in "${seeds[@]}"; do
    python ablation.py --method HGNN --dname $dataset \
        --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 \
        --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 \
        --seed $j --attack mla --alpha 0 --beta 0 --epsilon 0 --cuda 1 --train_prop $t &

    # python ablation.py --method HGNN --dname $dataset \
    #     --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 \
    #     --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 \
    #     --seed $j --attack mla --alpha 0.1 --beta 2.0 --epsilon 0 --cuda 1 --train_prop $t

    # python ablation.py --method HGNN --dname $dataset \
    #     --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 \
    #     --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 \
    #     --seed $j --attack mla --alpha 1.0 --beta 2.0 --epsilon 0 --cuda 1 --train_prop $t &

    # python ablation.py --method HGNN --dname $dataset \
    #     --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 \
    #     --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 \
    #     --seed $j --attack mla --alpha 2.0 --beta 2.0 --epsilon 0 --cuda 1 --train_prop $t


    # python ablation.py --method HGNN --dname $dataset \
    #     --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 \
    #     --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 \
    #     --seed $j --attack mla --alpha 4.0 --beta 2.0 --epsilon 0 --cuda 1 --train_prop $t &

    # python ablation.py --method HGNN --dname $dataset \
    #     --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 \
    #     --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 \
    #     --seed $j --attack mla --alpha 10.0 --beta 2.0 --epsilon 0 --cuda 1 --train_prop $t
done
# for j in "${seeds[@]}"; do
#     python ablation.py --method HGNN --dname $dataset \
#         --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 \
#         --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 \
#         --seed $j --attack mla --alpha 0 --beta 0.01 --epsilon 0 --cuda 1 --train_prop $t &
#     python ablation.py --method HGNN --dname $dataset \
#         --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 \
#         --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 \
#         --seed $j --attack mla --alpha 0 --beta 0.1 --epsilon 0 --cuda 1 --train_prop $t 
#     python ablation.py --method HGNN --dname $dataset \
#         --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 \
#         --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 \
#         --seed $j --attack mla --alpha 0 --beta 1 --epsilon 0 --cuda 1 --train_prop $t &
#     python ablation.py --method HGNN --dname $dataset \
#         --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 \
#         --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 \
#         --seed $j --attack mla --alpha 0 --beta 2 --epsilon 0 --cuda 1 --train_prop $t 
#     python ablation.py --method HGNN --dname $dataset \
#         --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 \
#         --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 \
#         --seed $j --attack mla --alpha 0 --beta 4 --epsilon 0 --cuda 1 --train_prop $t & 
#     python ablation.py --method HGNN --dname $dataset \
#         --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 \
#         --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 \
#         --seed $j --attack mla --alpha 0 --beta 10 --epsilon 0 --cuda 1 --train_prop $t
# done
# seeds=(11)
# for j in "${seeds[@]}"; do
#     python ablation.py --seed $j --epsilon 0.005  --num_epochs 1000 --eta_H 0.001 --eta_X 0.001 --T 80 --patience 150 --num_epochs_sur 80 --sur_class unigin
    # python ablation.py --seed $j --epsilon 0.005  --num_epochs 1000 --eta_H 0.001 --eta_X 0.001 --T 80 --patience 150 --num_epochs_sur 80
    # python ablation.py --seed $j --epsilon 0.005  --num_epochs 1000 --eta_H 0.001 --eta_X 0.001 --T 80 --patience 150 --num_epochs_sur 80 --hodge 
    # python ablation.py --seed $j --epsilon 0.005  --num_epochs 1000 --eta_H 0.001 --eta_X 0.001 --T 80 --patience 150 --num_epochs_sur 80 --sur_class hgnn
# done

# python ablation.py --method AllSetTransformer --dname $dataset \
# --heads 4 --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 256  \
#    --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0  \
#    --seed $j --attack mla --epsilon 0 --cuda 0 --train_prop $t --ptb_rate 0.2

# python ablation.py --method HyperGCN --dname $dataset   \
#   --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512  \
#      --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0  \
#         --seed $j --attack mla --epsilon 0 --cuda 0 --train_prop $t --ptb_rate 0.3 --alpha 0.0