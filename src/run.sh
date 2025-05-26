array1=(coauthor_cora cora citeseer)
seeds=(11 12 13 14 15 16 17 18 19 20)
for dataset in "${array1[@]}"; do
    for j in "${seeds[@]}"; do
        python train.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --attack mla_fgsm &
        python train.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --attack Rand-feat &
        python train.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --attack Rand-flip &
        python train.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --attack mla --mla_alpha 4.0 &
        python train.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --attack gradargmax
    done
done 
for dataset in "${array1[@]}"; do
    for j in "${seeds[@]}"; do
        python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack mla_fgsm --display_step 20 &
        python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack Rand-feat --display_step 20 
        python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack Rand-flip --display_step 20 &
        python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack mla --display_step 20 &
        python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack gradargmax --display_step 20
    done
done
array1=('zoo')
for dataset in "${array1[@]}"; do
    for j in "${seeds[@]}"; do
        python train.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.01 --perturb_type replace --perturb_prop 0 --seed $j --attack mla_fgsm 
        python train.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.01 --perturb_type replace --perturb_prop 0 --seed $j --attack mla
        python train.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.01 --perturb_type replace --perturb_prop 0 --seed $j --attack Rand-feat
        python train.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.01 --perturb_type replace --perturb_prop 0 --seed $j --attack Rand-flip 
        python train.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.01 --perturb_type replace --perturb_prop 0 --seed $j --attack gradargmax 
    done
done 

for dataset in "${array1[@]}"; do
    for j in "${seeds[@]}"; do
        python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack mla_fgsm --display_step 20 &
        python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack Rand-feat --display_step 20
        python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack Rand-flip --display_step 20 &
        python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack mla --display_step 20 &
        python train.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs 1000 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0.0 --seed $j --attack gradargmax --display_step 20
    done
done

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