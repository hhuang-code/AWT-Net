#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=60GB
#SBATCH --time=48:00:00
#SBATCH --job-name='sonn'
#SBATCH -p nvidia

#SBATCH --mail-type=END
#SBATCH --mail-user=hh1811@nyu.edu

cd /scratch/hh1811/projects/AWT-Net/classification

exp_name='exp_name'

data_path='../data/ScanObjectNN/h5_files'

dropout=(0.1 0.4)

comment='comment'

$(which python) sonn_main.py --exp_name ${exp_name} --data_path ${data_path} --n_classes 15 --comment ${comment} \
--optim_type sgd --sched_type cos --batch_size 48 --sync_bn --dropout ${dropout[*]} \
--nodes 1 --gpus 2 --n_rank 0