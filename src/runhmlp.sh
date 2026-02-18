array1=(cora citeseer coauthor_cora)
seeds=(11 12 13 14 15)
for dataset in "${array1[@]}"; do
    if [ "$dataset" == "cora" ] || [ "$dataset" == "citeseer" ]; then
        L="L2"
    else
        L="MSE"
    fi
    for j in "${seeds[@]}"; do
        # python train_hypermlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 100 --runs 1 --cuda 0 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack Rand-feat
        # python train_hypermlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 100 --runs 1 --cuda 0 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack Rand-flip 
        python train_hypermlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 100 --runs 1 --cuda 0 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack mla --T 30 --loss $L 
        python train_hypermlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 100 --runs 1 --cuda 1 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack mla_fgsm 
        # python train_hypermlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 100 --runs 1 --cuda 0 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack gradargmax
    done
done 
eps=0.05
seeds=(11)
for j in "${seeds[@]}"; do
    # python train_hypermlp.py --method MLP --dname zoo --All_num_layers 3 --feature_noise 0.0 --MLP_hidden 64 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --alpha 0.05 --attack mla --dropout 0.1 --seed $j --epsilon $eps --eta_H 1 --eta_X 1 # 16.7%, 8.3%
    # python train_hypermlp.py --method MLP --dname zoo --All_num_layers 3 --feature_noise 0.0 --MLP_hidden 64 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --alpha 0.05 --attack mla_fgsm --dropout 0.1 --seed $j --epsilon $eps # 33.3%, 8.3%
    # python train_hypermlp.py --method MLP --dname zoo --All_num_layers 3 --feature_noise 0.0 --MLP_hidden 64 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --alpha 0.05 --attack Rand-flip --dropout 0.1 
    # python train_hypermlp.py --method MLP --dname zoo --All_num_layers 3 --feature_noise 0.0 --MLP_hidden 64 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --alpha 0.05 --attack Rand-feat --dropout 0.1 
    # python train_hypermlp.py --method MLP --dname zoo --All_num_layers 3 --feature_noise 0.0 --MLP_hidden 64 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --alpha 0.05 --attack gradargmax --dropout 0.1 
done 

# python train.py --method MLP --dname zoo --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 64 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --alpha 0.05 --attack mla_fgsm --dropout 0.1 --seed 11 --attack mla_fgsm
# python train.py --method MLP --dname NTU2012 --All_num_layers 3 --feature_noise 0.0 --MLP_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.01 --alpha 0.05 --attack mla --seed 11 --eta_H 0.1 --eta_X 0.1
# python train.py --method MLP --dname NTU2012 --All_num_layers 3 --feature_noise 0.0 --MLP_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.01 --alpha 0.05 --attack mla_fgsm --seed 11 --eta_H 0.1 --eta_X 0.1

# eps=0.05
# seeds=(11 12 13 14 15)
# for j in "${seeds[@]}"; do
#     # python train.py --method MLP --dname NTU2012 --All_num_layers 3 --feature_noise 0.0 --MLP_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.01 --alpha 0.05 --attack mla --seed $j --eta_H 0.1 --eta_X 0.1 --epsilon $eps & # 53.14%, 11%
#     # python train.py --method MLP --dname NTU2012 --All_num_layers 3 --feature_noise 0.0 --MLP_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.01 --alpha 0.05 --attack mla_fgsm --seed $j --eta_H 0.1 --eta_X 0.1 --epsilon $eps  # 94%, 3.14%

#     python train_hypermlp.py --method MLP --dname zoo --All_num_layers 3 --feature_noise 0.0 --MLP_hidden 64 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --alpha 0.05 --attack mla --dropout 0.1 --seed $j --epsilon $eps --eta_H 1 --eta_X 1 & # 16.7%, 8.3%
#     python train_hypermlp.py --method MLP --dname zoo --All_num_layers 3 --feature_noise 0.0 --MLP_hidden 64 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --alpha 0.05 --attack mla_fgsm --dropout 0.1 --seed $j --epsilon $eps # 33.3%, 8.3%
#     python train_hypermlp.py --method MLP --dname zoo --All_num_layers 3 --feature_noise 0.0 --MLP_hidden 64 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --alpha 0.05 --attack Rand-flip --dropout 0.1 & 
#     python train_hypermlp.py --method MLP --dname zoo --All_num_layers 3 --feature_noise 0.0 --MLP_hidden 64 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --alpha 0.05 --attack Rand-feat --dropout 0.1 &
#     python train.py --method MLP --dname zoo --All_num_layers 3 --feature_noise 0.0 --MLP_hidden 64 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --alpha 0.05 --attack gradargmax --dropout 0.1 
# done 

# array1=(cora citeseer)
# for dataset in "${array1[@]}"; do
#     for j in "${seeds[@]}"; do
#         python train.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 500 --runs 1 --cuda 0 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack mla_fgsm
#     done
# done 

# python advtrain.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack mla_fgsm --surr_class MeLA-D+HyperMLP
# python advtrain.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack mla_fgsm --surr_class 
# python advtrain.py --method MLP --dname cora --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 100 --runs 1 --cuda 0 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed 11 --display_step 100 --attack Rand-feat --mode defense --surr_class MeLA-D+HyperMLP --num_epochs_sur 200 --T 12
