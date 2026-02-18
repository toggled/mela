# array1=(zoo)
# array1=(cora citeseer)
# array1=(coauthor_cora)
array1=(cora citeseer coauthor_cora)
seeds=(12 13 14 15)
for dataset in "${array1[@]}"; do
    for j in "${seeds[@]}"; do
        python advtrain.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 200 --runs 1 --cuda 2 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --num_epochs_sur 300 --attack mla_fgsm --T 20 & 
        python advtrain.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 200 --runs 1 --cuda 2 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --num_epochs_sur 300 --attack Rand-feat --T 20 &
        python advtrain.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 200 --runs 1 --cuda 2 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --num_epochs_sur 300 --attack Rand-flip --T 20 &
        python advtrain.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 200 --runs 1 --cuda 2 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --num_epochs_sur 300 --attack mla --T 20  &
        python advtrain.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 200 --runs 1 --cuda 2 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --num_epochs_sur 300 --attack gradargmax --T 20
    done
done
# for dataset in "${array1[@]}"; do
#     for j in "${seeds[@]}"; do
#         python advtrain.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 200 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --num_epochs_sur 300 --attack mla --T 20 --cuda 0 --alpha 0
#         python advtrain.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 200 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --num_epochs_sur 300 --attack mla --T 20 --cuda 0 --alpha 1       
#     done
# done
# ICML
# for dataset in "${array1[@]}"; do
#     for j in "${seeds[@]}"; do
#         # python advtrain.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 200 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --K_inner 5 --attack mla --T 20 --cuda 0 --alpha 0
#         python advtrain.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 200 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j \
#         --K_inner 100 --attack mla --T 30 --cuda 0 --alpha 1  --beta 1 --gamma 1  &
#         python advtrain.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 200 --runs 1 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j \
#         --K_inner 100 --attack mla \
#         --alpha 1  --beta 1 --gamma 1 --cuda 1 --T 30 &
        
#         # python advtrain.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 200 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j \
#         # --K_inner 5 --attack mla_fgsm --T 50 --cuda 0 --alpha 4  --beta 1 --gamma 4.0  &
        
#         # python advtrain.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 200 --runs 1 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j \
#         # --K_inner 50 --attack mla_pgd \
#         # --alpha 1  --beta 1 --gamma 1 --cuda 1 --T 20 &
#     done
# done