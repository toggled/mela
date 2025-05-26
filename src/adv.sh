array1=(zoo)
seeds=(11)
for dataset in "${array1[@]}"; do
    for j in "${seeds[@]}"; do
        python advtrain.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 200 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --num_epochs_sur 300 --attack mla_fgsm --T 20
        python advtrain.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 200 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --num_epochs_sur 300 --attack Rand-feat --T 20
        python advtrain.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 200 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --num_epochs_sur 300 --attack Rand-flip --T 20
        python advtrain.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 200 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --num_epochs_sur 300 --attack mla --T 20 
        python advtrain.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --epochs 200 --runs 1 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed $j --num_epochs_sur 300 --attack gradargmax --T 20
    done
done