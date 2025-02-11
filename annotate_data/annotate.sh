#!/bin/bash
#SBATCH -J annotate
#SBATCH -N 1 -c 9 --gres=gpu:1 --mem=72G
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH -t 0-24
#SBATCH -a 0-31

# Point to the root data directory
# Each job will iterate through a subset of files in $DATA_ROOT/$DOCUMENTS_DIR
# and annotate them with quality scores and domains
data_root=${DATA_ROOT:-}
documents_dir=${DOCUMENTS_DIR:-"documents"}

# Use WORKER/NUM_WORKERS env variables, slurm array variables or default to 0/1
num_workers=${NUM_WORKERS:-${SLURM_ARRAY_TASK_COUNT:-1}}
worker=${WORKER:-${SLURM_ARRAY_TASK_ID:-0}}

files=( $(ls -1 "$data_root/$documents_dir" | .jsonl.zst ) )
num_files=${#files[@]}

# Iterate through files for this work
for id in $(jq -n "range($worker; $num_files; $num_workers)"); do
    file=${files[$id]}
    output_file=${file%%.*}

    # Tokenize data and compute length
    python tokens.py \
        $data_root/$documents_dir/$file \
        $data_root/tokens/$output_file

    # Compute DCLM-fasttext scores
    python fasttext.py \
        $data_root/$documents_dir/$file \
        $data_root/scores_dclm-fasttext/$output_file \
        --model_path <path_to_dclm_fasttext_model>


   # ^ The two scripts above do not make use of a GPU and should be run separately
   # Everything below is accelerated a lot with GPUs

    # Compute FineWeb-Edu scores
    python edu.py \
        $data_root/$documents_dir/$file \
        $data_root/scores_fineweb-edu/$output_file \
        --model_name HuggingFaceTB/fineweb-edu-classifier

    # Compute Topic and Format domains
    python domains.py \
        $data_root/$documents_dir/$file \
        $data_root/domains_topics/$output_file \
        --model_name WebOrganizer/WebOrganizer-TopicClassifier
    python domains.py \
        $data_root/$documents_dir/$file \
        $data_root/domains_formats/$output_file \
        --model_name WebOrganizer/WebOrganizer-FormatClassifier

    # # For annotating kmeans clusters
    # python embed.py \
    #     $data_root/$documents_dir/$file \
    #     $data_root/embeds/$output_file
    # python clusters.py \
    #     $data_root/embeds/${output_file}.npy \
    #     $data_root/domains_clusters-k24/$output_file \
    #     --clustering_folder ../define_domains/k-means-clustering/exps/dclm-k24
done
