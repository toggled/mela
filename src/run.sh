array1=("coauthor_cora" "cora" "citeseer")
seeds=(11 12)
METHOD="HGNN"
L="L2"
t=0.5
ALPHA=0.1
# ALPHA=32 # Only for MLP on Cora-CA
# for dataset in "${array1[@]}"; do
#     for j in "${seeds[@]}"; do
#         # python train_hypermlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 100 --runs 1 --cuda 0 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack Rand-feat
#         # python train_hypermlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 100 --runs 1 --cuda 0 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack Rand-flip 
#         # python train_hypermlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 100 --runs 1 --cuda 0 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --display_step 100 --attack gradargmax

#         python train_hypermlp.py --method MLP --dname $dataset --All_num_layers 2 \
#             --feature_noise 0.0 --MLP_hidden 256 --wd 0.0 --epochs 100 --runs 1 --cuda 0 \
#             --lr 0.001 --hyperMLP_alpha 0.005 --ptb_rate 0.2 --patience 50 --dropout 0.1 \
#             --seed $j --attack mla_fgsm --alpha $ALPHA --beta 1 --gamma 4.0 --T 30 --loss $L &
#         python train_hypermlp.py --method MLP --dname $dataset --All_num_layers 2 \
#             --feature_noise 0.0 --MLP_hidden 256 --wd 0.0 --epochs 100 --runs 1 --cuda 1 \
#             --lr 0.001 --hyperMLP_alpha 0.005 --ptb_rate 0.2 --patience 50 --dropout 0.1 \
#             --seed $j --attack mla_pgd --alpha $ALPHA --beta 1 --gamma 4.0 --T 30 --loss $L 
#     done 
# done
for dataset in "${array1[@]}"; do
    if [[ "$METHOD" == "HGNN" ]]; then
        if [[ "$dataset" == "citeseer" || "$dataset" == "co_citeseer" ]]; then
            T=50
        elif [[ "$dataset" == "coauthor_cora" ]]; then
            T=20
        else
            T=30
        fi

    # elif [[ "$METHOD" == "AllSetTransformer" ]]; 
    #     T = 30

#         if [[ "$dataset" == "citeseer" || "$dataset" == "co_citeseer" ]]; then
#             ALPHA=0.0
#         elif [[ "$dataset" == "cora" || "$dataset" == "co_cora" ]]; then
#             ALPHA=0.01
#         elif [[ "$dataset" == "coauthor_cora" || "$dataset" == "coauth-cora" ]]; then
#             ALPHA=0.01
#         else
#             ALPHA=0.0
#         fi

    else
        T=30
    fi

#     echo "[INFO] METHOD=$METHOD  DNAME=$dataset  ALPHA=$ALPHA"

    for j in "${seeds[@]}"; do
        python train.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --attack mla_fgsm --alpha $ALPHA --beta 1 --gamma 4.0 --epsilon 0.05 --cuda 0 --train_prop $t --loss $L &
        python train.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --attack Rand-feat --train_prop $t &
        python train.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --attack Rand-flip --train_prop $t 
        python train.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --attack mla_pgd --alpha $ALPHA --beta 1 --gamma 4.0 --epsilon 0.05 --cuda 0 --train_prop $t --loss $L --T $T
        python train.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 1 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --attack gradargmax --train_prop $t &

        # # python train_hypermlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 100 --runs 1 --cuda 3 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --attack Rand-feat --train_prop $t &
        # python train_hypermlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 100 --runs 1 --cuda 0 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --attack Rand-flip --train_prop $t 
        # # python train_hypermlp.py --method MLP --dname $dataset --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 100 --runs 1 --cuda 2 --lr 0.001 --alpha 0.005 --ptb_rate 0.2 --dropout 0.1 --seed $j --attack gradargmax --train_prop $t 

    done
done 
# METHOD="AllSetTransformer"
# for dataset in "${array1[@]}"; do
# #     if [[ "$METHOD" == "HGNN" ]]; then

# #         if [[ "$dataset" == "coauthor_cora" || "$dataset" == "coauth-cora" ]]; then
# #             ALPHA=2.0
# #         elif [[ "$dataset" == "citeseer" || "$dataset" == "co_citeseer" ]]; then
# #             ALPHA=2.0
# #         elif [[ "$dataset" == "cora" || "$dataset" == "co_cora" ]]; then
# #             ALPHA=0.01
# #         else
# #             ALPHA=0.0
# #         fi

# #     elif [[ "$METHOD" == "AllSetTransformer" ]]; then

# #         if [[ "$dataset" == "citeseer" || "$dataset" == "co_citeseer" ]]; then
# #             ALPHA=0.0
# #         elif [[ "$dataset" == "cora" || "$dataset" == "co_cora" ]]; then
# #             ALPHA=0.01
# #         elif [[ "$dataset" == "coauthor_cora" || "$dataset" == "coauth-cora" ]]; then
# #             ALPHA=0.01
# #         else
# #             ALPHA=0.0
# #         fi

# #     else
# #         echo "ERROR: Unsupported METHOD = $METHOD"
# #         exit 1
# #     fi

# #     echo "[INFO] METHOD=$METHOD  DNAME=$dataset  ALPHA=$ALPHA"

#     for j in "${seeds[@]}"; do
#         # python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack mla_fgsm \
#         #      --alpha $ALPHA --beta 1 --gamma 4.0 --epsilon 0.05 --cuda 0 --train_prop $t --loss $L &
#         # python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack Rand-feat --display_step 20 
#         # python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack Rand-flip --display_step 20 &
#         # python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack mla_pgd \
#         #     --alpha $ALPHA  --beta 1 --gamma 4.0 --epsilon 0.05 --cuda 1 --train_prop $t --loss $L --T 30
#         python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack gradargmax 
#     done
# done
# array1=('zoo')
# for dataset in "${array1[@]}"; do
#     for j in "${seeds[@]}"; do
#         python train.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.01 --perturb_type replace --perturb_prop 0 --seed $j --attack mla_fgsm 
#         python train.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.01 --perturb_type replace --perturb_prop 0 --seed $j --attack mla
#         python train.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.01 --perturb_type replace --perturb_prop 0 --seed $j --attack Rand-feat
#         python train.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.01 --perturb_type replace --perturb_prop 0 --seed $j --attack Rand-flip 
#         python train.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.01 --perturb_type replace --perturb_prop 0 --seed $j --attack gradargmax 
#     done
# done 

# for dataset in "${array1[@]}"; do
#     for j in "${seeds[@]}"; do
#         python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack mla_fgsm --display_step 20 &
#         python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack Rand-feat --display_step 20
#         python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack Rand-flip --display_step 20 &
#         python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack mla --display_step 20 &
#         python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack gradargmax --display_step 20
#     done
# done

# array1=(cora citeseer)
# seeds=(15)
# for dataset in "${array1[@]}"; do
#     for j in "${seeds[@]}"; do
#         #python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack mla_fgsm --display_step 20 &
#         #python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack Rand-feat --display_step 20 
#         #python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack Rand-flip --display_step 20 &
#         #python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack mla --display_step 20 &
#         python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack gradargmax --display_step 20
#     done
# done

# array1=(coauthor_cora cora citeseer)
# seeds=(11 12 13 14 15)
# for dataset in "${array1[@]}"; do
#     for j in "${seeds[@]}"; do
#         python train.py --method AllDeepSets --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack mla_fgsm --display_step 20 &
#         python train.py --method AllDeepSets --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack Rand-feat --display_step 20
#         python train.py --method AllDeepSets --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack Rand-flip --display_step 20 &
#         python train.py --method AllDeepSets --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack mla --display_step 20 &
#         python train.py --method AllDeepSets --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack gradargmax --display_step 20
#     done
# done

# python train.py --method HGNN --dname house-committees-100 --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.01 --perturb_type replace --perturb_prop 0 --seed 11 --attack mla_fgsm 
# python train.py --method HGNN --dname 20newsW100 --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.1 --perturb_type replace --perturb_prop 0 --seed 11 --attack mla_fgsm