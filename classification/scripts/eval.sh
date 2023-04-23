#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=60GB
#SBATCH --time=48:00:00
#SBATCH --job-name='modelnet'
#SBATCH -p nvidia

#SBATCH --mail-type=END
#SBATCH --mail-user=hh1811@nyu.edu

cd /scratch/hh1811/projects/AWT-Net/classification

exp_name='pretrained/modelnet40'

data_path='../data/modelnet40_ply_hdf5_2048'

comment='comment'

$(which python) main.py --exp_name ${exp_name} --data_path ${data_path} --comment ${comment} --batch_size 48 \
--nodes 1 --gpus 2 --n_rank 0 --sync_bn --eval