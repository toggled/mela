## Overview

This code is for the NeurIPS paper titled ``MeLA-D: Adversarial Attacks and Robust Training for
Hypergraph Neural Networks via Meta Learning``

1. Running MeLA-FGSM and MeLA-D to attack various datasets: Execute on the terminal

```bash
bash run.sh # To run HGNN, AllSetTransformer 
bash runmlp.sh # To run HyperMLP
```
2. Running MeLA-D for adversarial training.

```bash
bash adv.sh # To run HGNN
bash adv_hmlp.sh # To run HyperMLP
```

3. Running minibatch MeLA-D to attack large-scale hypergraph:

```bash
bash big.sh
```
## Requirements

- Python 3.12.8
- torch 2.6.0+cu126
- torch geometric 2.6.1
- numpy 2.1.2

## Datasets
Please check the `data/` folder.