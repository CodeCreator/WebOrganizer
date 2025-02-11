import time
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass

from simple_parsing import ArgumentParser, field
from simple_parsing.helpers import Serializable
from typing import Callable, Dict, Optional, List, Any, Tuple
from collections.abc import Iterable

from functools import partial
from copy import copy
import json

from multiprocessing import Pool
from datatools import load, process, identity_fn, ProcessOptions

from pathlib import Path

from contextlib import contextmanager


@dataclass
class DatasetOptions(Serializable):
    """This script requires a strict folder structure where the root folder has
    subfolders for documents, tokens, annotations, domains, each with an equal number of shards.
    Example:
    ```
        base_corpus/
            documents/
                - CC_shard_00000000_processed.jsonl.zst
                - CC_shard_00000001_processed.jsonl.zst
            tokens/
                - CC_shard_00000000_processed.npy
                - CC_shard_00000001_processed.npy
            some_annotations/
                - CC_shard_00000000_processed.npy
                - CC_shard_00000001_processed.npy
    ```
    We will write a new folder with a similar structure,
    first writing a subfolder `indices` which can be used to reconstruct
    the rest of the selected data, and then all the other data folders listed in `data_dirs`.
    """
    input_base: Path = field(positional=True, help="Path to the input folder")
    output_base: Path = field(positional=True, help="Path to the output folder")

    data_dirs: List[Path] = field(default_factory=lambda: ["documents"], help="Relative paths to data folders")
    indices_dir: Path = field(default="indices", help="Relative to the output folder file containing indices")

    num_proc: int = field(default=8, help="Number of processes to use for parallel processing", alias="-w")
    held_out_shards: int = field(default=100, help="Hold out the last n shards for evaluation. By default, we use 100 shards.")
    argument_file: Path = field(default="args.yaml", help="Relative path for saving the arguments for this script")

    tokens_dir: Path = field(default="tokens", help="Subfolder with number of tokens per document")
    token_suffix: str = field(default=".npy", help="Extension of the annotation files")

    annotations_dir: Optional[Path] = field(default=None, help="Relative containing annotations")
    annotation_suffix: str = field(default=".npy", help="Extension of the annotation files")

    domains_dir: List[Path] = field(help="Relative paths to the input folders containing domains", default=None)
    domain_suffix: List[str] = field(default_factory=lambda: [".npy"], help="Extension of the domain files")
    ref_distribution: Optional[Path] = field(help="Path to the reference distribution file", default=None)



@dataclass
class SelectOptions(Serializable):
    num_tokens: Optional[int] = None

    invert: bool = False # Perform bottom-k since top-k selection
    from_median: bool = False # Select based on distance to annotation median

    threshold: Optional[float] = None # Select based on absolute threshold of annotations
    quantile_threshold: Optional[float] = None # Select based on quantile threshold of annotations

    do_sample: bool = False # Use top-k-gumbel sampling
    temperature: float = 1.0 # Temperature for sampling
    normalize: bool = False # Normalize mean and standard deviation annotations before applying temperature
    seed: int = 42

    multi_epoch: bool = False # Repeat samples if necessary


@contextmanager
def timer(action_name: str):
    start = time.time()
    yield
    print(f"{action_name} in {time.time() - start:.2f}s")


def argsort_first_k(arr: np.ndarray, k: int):
    """Equivalent to np.argsort(arr)[:k], but faster for large array and small k."""
    if k >= len(arr) * 0.5:
        return np.argsort(arr)[:k]
    indices = np.argpartition(arr, k)[:k]
    return indices[np.argsort(arr[indices])]


def select_indices_by_num_tokens(annotations: np.ndarray, tokens: np.ndarray, num_tokens: int, token_margin: int = 10_000_000):
    if num_tokens == 0:
        return np.array([], dtype=np.int64)

    total_tokens = np.sum(tokens)

    proposed_num_docs = np.ceil(len(annotations) * (num_tokens + token_margin) / total_tokens)
    proposed_num_docs = min(proposed_num_docs, len(annotations))

    indices = argsort_first_k(-annotations, int(proposed_num_docs))
    cum_tokens = np.cumsum(tokens[indices])

    if cum_tokens[-1] < num_tokens:
        if total_tokens < num_tokens:
            raise ValueError("Insufficient tokens")
        else:
            return select_indices_by_num_tokens(annotations, tokens, num_tokens, 2*token_margin)

    cutoff = np.argmax(cum_tokens >= num_tokens)
    return indices[:cutoff+1]


def select_indices_per_domain(annotations: np.ndarray, tokens: np.ndarray, options: SelectOptions):
    if options.invert:
        annotations = -annotations

    if options.threshold is not None or options.quantile_threshold is not None:
        threshold = (
            options.threshold
            if options.threshold is not None else
            np.quantile(annotations, options.quantile_threshold)
        )
        annotations[annotations < threshold] = -np.inf
        annotations[annotations >= threshold] = 0

    if options.from_median:
        annotations = np.abs(annotations - np.median(annotations))

    if options.do_sample:
        if options.normalize:
            annotations = (annotations - annotations.mean()) / annotations.std()

        annotations /= options.temperature

        annotations += np.random.gumbel(size=len(annotations)) # Use topk-gumbel trick

    assert options.num_tokens is not None
    total_num_tokens = np.sum(tokens)
    num_epochs = options.num_tokens // total_num_tokens

    if not options.multi_epoch and num_epochs >= 1:
        print(f" > Short of {options.num_tokens - total_num_tokens} tokens")
        return np.arange(len(annotations))

    epochs = [np.arange(len(annotations)) for _ in range(num_epochs)]
    epochs.append(select_indices_by_num_tokens(annotations, tokens, options.num_tokens - num_epochs * total_num_tokens))
    return np.concatenate(epochs)


def select_indices_for_domain(domain_id: Any, metadata_df: pd.DataFrame, target_domain_proportion: Dict[Any, float], options: SelectOptions):
    domain_columns = sorted([c for c in metadata_df.columns if isinstance(c, tuple) and c[0] == "domains"])
    domain_indices = metadata_df.index[np.all(metadata_df[domain_columns] == domain_id, axis=1)].to_numpy()

    np.random.seed(options.seed + hash(domain_id) % 2**24)

    options_ = copy(options)
    if options_.num_tokens is not None:
        options_.num_tokens = round(options.num_tokens * target_domain_proportion[domain_id])

    indices = select_indices_per_domain(
        metadata_df["annotations"].loc[domain_indices].to_numpy(),
        metadata_df["tokens"].loc[domain_indices].to_numpy(),
        options_)

    print(f"Domain {domain_id}:\t Selected {len(indices): 8} out of {len(domain_indices)} ({len(indices) / len(domain_indices):.2%})")
    return domain_indices[indices]


def write_indices(shard_name_and_index_range: Dict, subset_indices: Optional[np.ndarray], options: DatasetOptions):
    name = shard_name_and_index_range["shard_name"]

    index_start, index_end = shard_name_and_index_range["index_range"]
    file_indices = np.sort(subset_indices[(subset_indices >= index_start) & (subset_indices < index_end)]) - index_start

    indices_path = options.output_base / options.indices_dir / (name + ".npy")
    indices_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(indices_path, file_indices)


def write_data(shard_name: str, options: DatasetOptions):
    file_indices = np.load(options.output_base / options.indices_dir / (shard_name + ".npy"))

    for data_dir in options.data_dirs:
        (options.output_base / data_dir).mkdir(parents=True, exist_ok=True)
        data_files = sorted(list((options.input_base / data_dir).glob(f"{shard_name}*")))
        assert len(data_files) > 0, f"No files found for {shard_name} in {data_dir}"
        for file in data_files:
            process_options = ProcessOptions(indices=file_indices, log_level=logging.WARNING)

            output_name = file.name
            if file.name.endswith(".zst"):
                process_options.compression="zst"
                output_name = output_name[:-4]

            if output_name.endswith(".npy"):
                process_options.ndarray = True
                output_name = output_name[:-4]
            elif output_name.endswith(".jsonl"):
                process_options.jsonl = True
                output_name = output_name[:-6]

            process(load(file),
                    partial(identity_fn, disable=True),
                    options.output_base / data_dir / output_name,
                    process_options)


def load_dataframe(shard_name: Path,
                   options: DatasetOptions,
                   reference: bool = False):
    base_dir = options.input_base if not reference else options.ref_base
    token_path = base_dir / options.tokens_dir / (shard_name + options.token_suffix)
    if reference and not token_path.is_file():
        return pd.DataFrame()

    df = pd.DataFrame({
        "tokens": np.load(token_path)
    })

    if options.annotations_dir is not None and not reference:
        df["annotations"] = np.load(base_dir / options.annotations_dir / (shard_name + options.annotation_suffix))

    if options.domains_dir is not None:
        for i, domains_dir in enumerate(options.domains_dir):
            domains = np.load(base_dir / domains_dir / (shard_name + options.domain_suffix[i]))
            df[("domains", i)] = domains  # Use integer column names for domains

    return df


def compute_global_indices(shard_names: List[str], pool, options: DatasetOptions, filter_options: SelectOptions):
    with timer("Loading metadata"):
        metadata_dfs = pool.map(
            partial(load_dataframe, options=options),
            shard_names
        )
        shard_sizes = [len(df) for df in metadata_dfs]
        metadata_df = pd.concat(metadata_dfs, ignore_index=True)

    if options.annotations_dir is None:
        metadata_df["annotations"] = np.zeros(len(metadata_df), dtype=np.float32)

    if options.ref_distribution is not None:
        with open(options.ref_distribution, "r") as f:
            kvs = json.load(f)
        target_domain_proportion = {
            (tuple(entry["domain"]) if isinstance(entry["domain"], list) else entry["domain"]): entry["weight"]
            for entry in kvs
        }
    else:
        metadata_df[0] = np.zeros(len(metadata_df), dtype=np.int64)
        target_domain_proportion = {0: 1.0}

    with timer("Selecting indices"):
        subset_indices = np.concatenate(
            list(map(
                partial(select_indices_for_domain,
                        metadata_df=metadata_df,
                        target_domain_proportion=target_domain_proportion,
                        options=filter_options),
                sorted(list(target_domain_proportion.keys()))
            ))
        )

    num_token_total = metadata_df.loc[subset_indices]["tokens"].sum()
    print(f"Overall: Selected {num_token_total} tokens")

    return subset_indices, shard_sizes, num_token_total


def compute_global_indices_with_retries(shard_names: List[str], pool, options: DatasetOptions, filter_options: SelectOptions):
    original_num_tokens = filter_options.num_tokens

    subset_indices, shard_sizes, num_token_total = compute_global_indices(shard_names, pool, options, filter_options)

    while num_token_total < 0.99 * original_num_tokens:
        filter_options.num_tokens = int(filter_options.num_tokens * original_num_tokens / num_token_total)
        print(f" > Adjusting target number of tokens to {filter_options.num_tokens} and retrying", flush=True)
        subset_indices, shard_sizes, num_token_total = compute_global_indices(shard_names, pool, options, filter_options)

    return subset_indices, shard_sizes


def select(options: DatasetOptions, filter_options: SelectOptions):
    token_paths = sorted((options.input_base / options.tokens_dir).glob(f"*{options.token_suffix}"))

    if options.held_out_shards > 0:
        token_paths = token_paths[:-options.held_out_shards]

    shard_names = [
        str(path.name)[:len(str(path.name)) - len(options.token_suffix)]
        for path in token_paths
    ]
    with Pool(processes=options.num_proc) as pool:
        if all((options.output_base / options.indices_dir / (shard_name + ".npy")).is_file() for shard_name in shard_names):
            print("WARNING: Using existing indices!")
        else:
            subset_indices, shard_sizes = compute_global_indices_with_retries(shard_names, pool, options, filter_options)

            with timer("Writing indices"):
                boundaries = [0] + np.cumsum(shard_sizes).tolist()
                pool.map(
                    partial(write_indices, options=options, subset_indices=subset_indices),
                    ({"shard_name": shard_names[i], "index_range": (boundaries[i], boundaries[i + 1])} for i in range(len(shard_names)))
                )

        with timer("Writing subsets"):
            pool.map(
                partial(write_data, options=options),
                shard_names
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="Resample labels to desired distribution")

    parser.add_arguments(DatasetOptions, dest="select_options")
    parser.add_arguments(SelectOptions, dest="filter_options")

    args = parser.parse_args()
    print("Arguments", args)

    select(args.select_options, args.filter_options)

    argument_path = args.select_options.output_base / args.select_options.argument_file
    if not argument_path.is_file():
        @dataclass
        class AllOptions(Serializable):
            select_options: DatasetOptions
            filter_options: SelectOptions

        AllOptions(select_options=args.select_options, filter_options=args.filter_options).save_yaml(argument_path)
    else:
        print(f"Argument file {argument_path} already exists, skipping writing arguments")