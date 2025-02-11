#!/bin/bash
#SBATCH -J prompt_classify
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH -N 1 -c 12 --mem=40G --gres=gpu:8
#SBATCH -t 0-12
#SBATCH -a 0-3

config=${CONFIG:-configs/topics.yaml}  # defines taxonomy and instructions
model=${MODEL:-405B-FP8}  # Llama model to use
size=${SIZE:-10K}  # how many samples to process (across job array)
seed=${SEED:-43}   # random seed for order of categories and few-shot examples


# Convert size to number of samples to process
if [[ $size == *M ]]; then
    max_index=$((${size%M} * 1000000))
elif [[ $size = *K ]]; then
    max_index=$((${size%K} * 1000))
else
    max_index=$size
fi

# Get number of available GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    num_gpus=$(nvidia-smi -L | wc -l)
else
    num_gpus=$(jq -n "[$CUDA_VISIBLE_DEVICES] | length")
fi
num_nodes=${NUM_NODES:-${SLURM_JOB_NUM_NODES:-1}}
port=56421

export OUTLINES_CACHE_DIR=/tmp/outlines  # Fixes some job issues with outlines cache on shared filesystem

if [ $num_nodes -gt 1 ]; then
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    srun bash -c 'HOST_IP=$(hostname -i) python -m sglang.launch_server \
        --model-path "'Meta-Llama-3.1-${model%-FP8}-Instruct${model#*B})'" \
        --port "'$port'" \
        --tp "'$(($num_gpus * $num_nodes))'" \
        --nnodes "'$num_nodes'" \
        --node-rank "$SLURM_NODEID" \
        --nccl-init "'$master_addr':'$port'"' &
else
    python -m sglang.launch_server \
        --model-path Meta-Llama-3.1-${model%-FP8}-Instruct${model#*B} \
        --port $port \
        --tp $num_gpus &
        # --enable-torch-compile \
        # --dtype bfloat16  \
fi

config_name=$(basename $config)
config_name=${config_name%.yaml}

python prompt_classify.py datasets/dclm-refinedweb-sample1M.jsonl datasets/dclm-sample${size}-${config_name}-${model}-seed${seed} \
    --config_path ${config} \
    --num_threads 1 \
    --batch_size 1000 \
    --port $port \
    --slurm_array \
    --index_range 0 $max_index \
    --randomize_seed $seed \
    $@


kill -9 $(jobs -p)
