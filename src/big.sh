#!/bin/bash
seeds=(11 13 15)
for j in "${seeds[@]}"; do
    python train_minibatch.py --method HGNN --dname coauthor_dblp --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 128 --wd 0.0 --epochs 500 --runs 1 --lr 0.001 --perturb_type replace --perturb_prop 0 \
    --epsilon 0.5 --ptb_rate 0.2 --alpha 16 --batch_size 1024 --patience 4 --num_epochs_sur 20 --T 20 --seed $j --cuda 0 &
    python train_minibatch.py --method HGNN --dname coauthor_dblp --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 128 --wd 0.0 --epochs 500 --runs 1 --lr 0.001 --perturb_type replace --perturb_prop 0 \
    --epsilon 0.5 --ptb_rate 0.2 --alpha 16 --batch_size 1024 --patience 4 --num_epochs_sur 20 --T 20 --seed $((j+1)) --cuda 1
done