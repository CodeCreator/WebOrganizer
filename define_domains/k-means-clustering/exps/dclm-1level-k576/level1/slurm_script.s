#!/bin/bash

#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=kmeans_level1
#SBATCH --time=1-0
#SBATCH --mem=800G
#SBATCH --cpus-per-task=32
#SBATCH --partition=pli-c

EXPDIR=/scratch/gpfs/awettig/delve/k-means-clustering/exps/dclm-1level-k576/level1
cd /scratch/gpfs/awettig/delve/k-means-clustering/scripts

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

PYTHONPATH=.. \
srun -N 4 --unbuffered --output="$EXPDIR"/logs/%j_%t_log.out --error="$EXPDIR"/logs/%j_%t_log.err  torchrun \
  --nnodes=4 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$master_addr:56321 \
    run_distributed_kmeans.py \
    --use_torchrun \
    --data_path /scratch/gpfs/PLI/awettig/dclm/dclm-pool-1b-1x/deduplicated/embeds \
    --n_clusters 576 \
    --n_iters 50 \
    --chunk_size 347222 \
    --dtype float32 \
    --high_precision float32 \
    --checkpoint_period 10000 \
    --exp_dir $EXPDIR \
    --n_steps 1 \
    --sample_size 1 \
    --do_not_sort_clusters \
    --held_out_shards 100 \
    --sampling_strategy r
