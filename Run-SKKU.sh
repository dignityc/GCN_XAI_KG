#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/log.out
#SBATCH --job-name=gpu
#SBATCH --gres=gpu:1

sampling_size=260000
batch_size=65536
lr=0.002

python GCN-SKKU.py --sampling_size $sampling_size --batch_size $batch_size --alpha 0.0 --lr $lr
python GCN-SKKU.py --sampling_size $sampling_size --batch_size $batch_size --alpha 0.2 --lr $lr
python GCN-SKKU.py --sampling_size $sampling_size --batch_size $batch_size --alpha 0.4 --lr $lr
python GCN-SKKU.py --sampling_size $sampling_size --batch_size $batch_size --alpha 0.6 --lr $lr
python GCN-SKKU.py --sampling_size $sampling_size --batch_size $batch_size --alpha 0.8 --lr $lr
python GCN-SKKU.py --sampling_size $sampling_size --batch_size $batch_size --alpha 1.0 --lr $lr

