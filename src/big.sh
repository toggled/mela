seeds=(11 12 13 14 15 16 17 18 19 20)
for j in "${seeds[@]}"; do
    python train_minibatch.py --method HGNN --dname coauthor_dblp --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 128 --wd 0.0 --epochs 500 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 --patience 4 --num_epochs_sur 30 --T 10 --seed $j
end