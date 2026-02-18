array1=(coauthor_cora)
seeds=(11 12 13 14 15)
for dataset in "${array1[@]}"; do
    for j in "${seeds[@]}"; do  
        python advtrain_hmlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 100 --runs 1 --cuda 0 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack Rand-feat --mode defense --surr_class MeLA-D+HyperMLP --num_epochs_sur 200 --T 15 --lr 0.0005 &
        python advtrain_hmlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 100 --runs 1 --cuda 1 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack Rand-flip --mode defense --surr_class MeLA-D+HyperMLP --num_epochs_sur 200 --T 15 --lr 0.0001 &
        python advtrain_hmlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 100 --runs 1 --cuda 0 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack gradargmax --mode defense --surr_class MeLA-D+HyperMLP --num_epochs_sur 200 --T 12 --lr 0.0001 &
        python advtrain_hmlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 400 --runs 1 --cuda 1 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack mla --mode defense --surr_class MeLA-D+HyperMLP --num_epochs_sur 400 --T 15 &
        python advtrain_hmlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 400 --runs 1 --cuda 0 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack mla_fgsm --mode defense --surr_class MeLA-D+HyperMLP --num_epochs_sur 400 --T 15
    done
done

array1=(cora)
for dataset in "${array1[@]}"; do
    for j in "${seeds[@]}"; do 
        python advtrain_hmlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 400 --runs 1 --cuda 0 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack Rand-feat --mode defense --surr_class MeLA-D+HyperMLP --num_epochs_sur 400 --T 15 &
        python advtrain_hmlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 100 --runs 1 --cuda 1 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack Rand-flip --mode defense --surr_class MeLA-D+HyperMLP --num_epochs_sur 200 --T 15 --lr 0.0001 &
        python advtrain_hmlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 10 --runs 1 --cuda 0 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack mla --mode defense --surr_class MeLA-D+HyperMLP --num_epochs_sur 40 --T 15 &
        python advtrain_hmlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 100 --runs 1 --cuda 1 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack gradargmax --mode defense --surr_class MeLA-D+HyperMLP --num_epochs_sur 200 --T 12 --lr 0.0001 & 
        python advtrain_hmlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 400 --runs 1 --cuda 0 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack mla_fgsm --mode defense --surr_class MeLA-D+HyperMLP --num_epochs_sur 400 --T 15
    done
done
array1=(citeseer)
for dataset in "${array1[@]}"; do
    for j in "${seeds[@]}"; do 
        python advtrain_hmlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 400 --runs 1 --cuda 0 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack Rand-feat --mode defense --surr_class MeLA-D+HyperMLP --num_epochs_sur 400 --T 10 --lr 0.003 &
        python advtrain_hmlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 100 --runs 1 --cuda 1  --alpha 0.005 --ptb_rate 0.2 --dropout 0.0 --seed $j --display_step 100 --attack Rand-flip --mode defense --surr_class MeLA-D+HyperMLP --num_epochs_sur 180 --T 15 --lr 0.001 &
        python advtrain_hmlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 100 --runs 1 --cuda 0 --alpha 0.005 --ptb_rate 0.2 --dropout 0.0 --seed $j --display_step 200 --attack gradargmax --mode defense --surr_class MeLA-D+HyperMLP --num_epochs_sur 200 --T 12 --lr 0.001 &
        python advtrain_hmlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 400 --runs 1 --cuda 1 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack mla --mode defense --surr_class MeLA-D+HyperMLP --num_epochs_sur 400 --T 15 &
        python advtrain_hmlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 400 --runs 1 --cuda 0 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack mla_fgsm --mode defense --surr_class MeLA-D+HyperMLP --num_epochs_sur 400 --T 15
    done
done
