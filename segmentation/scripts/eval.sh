#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=60GB
#SBATCH --time=48:00:00
#SBATCH --job-name='shapenetpart'
#SBATCH -p nvidia
#SBATCH --reservation=YiFang2

#SBATCH --mail-type=END
#SBATCH --mail-user=hh1811@nyu.edu

cd /scratch/hh1811/projects/AWT-Net/segmentation

exp_name='pretrained/shapenet'

data_path='../data/shapenetcore_partanno_segmentation_benchmark_v0_normal'

comment='comment'

$(which python) main.py --exp_name ${exp_name} --data_path ${data_path} --comment ${comment} --batch_size 16 \
--nodes 1 --gpus 4 --n_rank 0 --sync_bn --eval --model_type 'insiou'

