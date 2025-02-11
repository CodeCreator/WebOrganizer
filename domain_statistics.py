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
            some_domain_annotation/
                - CC_shard_00000000_processed.npy
                - CC_shard_00000001_processed.npy
    ```
    """
    input_base: Path = field(positional=True, help="Path to the input folder containing labels")

    tokens_dir: Path = field(default="token_annotations", help="Relative to the output folder containing tokens")

    domains_dir: List[Path] = field(help="Relative paths to the input folders containing domains", default=None)

    domain_suffix: List[str] = field(default_factory=lambda: [".npy"], help="Extension of the domain files")
    token_suffix: str = field(default=".npy", help="Extension of the annotation files")
    num_proc: int = field(default=8, help="Number of processes to use", alias="-w")


def load_dataframe(shard_name: Path,
                   options: DatasetOptions):
    token_path = options.input_base / options.tokens_dir / (shard_name + options.token_suffix)

    df = pd.DataFrame({
        "tokens": np.load(token_path)
    })

    if options.domains_dir is not None:
        for i, domains_dir in enumerate(options.domains_dir):
            domains = np.load(options.input_base / domains_dir / (shard_name + options.domain_suffix[i]))
            df[("domains", i)] = domains  # Use integer column names for domains

    return df


def generate_statistics(options: DatasetOptions):
    token_paths = sorted((options.input_base / options.tokens_dir).glob(f"*{options.token_suffix}"))
    shard_names = [
        str(path.name)[:len(str(path.name)) - len(options.token_suffix)]
        for path in token_paths
    ]

    with Pool(processes=options.num_proc) as pool:
        metadata_dfs = pool.map(
            partial(load_dataframe, options=options),
            shard_names
        )
        metadata_df = pd.concat(metadata_dfs, ignore_index=True)
        domain_columns = sorted([c for c in metadata_df.columns if isinstance(c, tuple) and c[0] == "domains"])
        df_by_domain = metadata_df.groupby(domain_columns)["tokens"].aggregate(["sum", "count"])
        total_tokens = df_by_domain["sum"].sum()
        summary = [
            {"domain": k, "tokens": v["sum"], "documents": v["count"], "weight": v["sum"] / total_tokens}
            for k, v in df_by_domain.to_dict("index").items()
        ]
        file_name = "domain_statistics/" + "_".join(d.name.removesuffix("_annotations").removeprefix("domains_") for d in options.domains_dir) + ".json"

        print("Statistics:")
        for s in summary:
            print(s)

        (options.input_base / file_name).parent.mkdir(parents=True, exist_ok=True)

        with (options.input_base / file_name).open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved statistics to {file_name}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(DatasetOptions, dest="options")
    args = parser.parse_args()
    generate_statistics(args.options)
