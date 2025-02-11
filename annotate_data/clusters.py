from tqdm import tqdm
import numpy as np
import torch
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

from datatools.process import process, ProcessOptions
from datatools.load import load, LoadOptions
from simple_parsing import ArgumentParser


@dataclass
class ScriptOptions:
    clustering_folder: Path
    batch_size: int = 4192
    device: str = "cpu"


def assign_clusters(dataset, indices, process_id, options):
    centroids_paths = sorted(options.clustering_folder.glob("level*/centroids.npy"))

    centroids_by_level = [torch.tensor(np.load(centroids_path)).to(options.device) for centroids_path in centroids_paths]

    for i in tqdm(range(0, len(dataset), options.batch_size), disable = process_id != 0):
        batch = [dataset[j] for j in range(i, min(i + options.batch_size, len(dataset)))]
        embeddings = torch.tensor(np.stack(batch)).to(options.device)

        assignments_by_level = []

        for centroids in centroids_by_level:
            # Compute distances
            distances = torch.cdist(embeddings, centroids)

            # Get cluster assignments
            cluster_ids = torch.argmin(distances, dim=1)
            assignments_by_level.append(cluster_ids.cpu().numpy())

            embeddings = centroids[cluster_ids]

        for cluster_id_by_level in zip(*assignments_by_level):
            yield {f"level{i+1}": cluster_id for i, cluster_id in enumerate(cluster_id_by_level)}


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("inputs", type=Path, nargs="+", help="Input embeds paths")
    parser.add_argument("output", type=Path, help="Output dataset path")

    parser.add_arguments(ScriptOptions, dest="script_options")
    parser.add_arguments(LoadOptions, dest="load_options")
    parser.add_arguments(ProcessOptions, dest="process_options")

    args = parser.parse_args()
    args.process_options.ndarray = True

    print("Arguments:", args)
    dataset = load(*args.inputs, options=args.load_options)
    N = len(dataset)
    print(f"Loaded dataset with {N} samples")

    process(dataset, partial(assign_clusters, options=args.script_options), args.output, args.process_options)