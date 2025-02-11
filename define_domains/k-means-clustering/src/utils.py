# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import logging
from pathlib import Path

import numpy as np
import torch

from streaming import LocalDataset


class MultiMemMap:
    def __init__(self, path: str, held_out_shards: int = 0):
        """
        Parameters:
            X: memmap to a numy array, or an array
            indices: array, indices representing the slice
        """
        paths = sorted(Path(path).glob("*.npy"))
        self.shards = [
            np.load(path, mmap_mode="r")
            for path in paths[:len(paths)-held_out_shards]
        ]
        self.lengths = [
            len(shard) for shard in self.shards
        ]
        self.cum_lengths = np.cumsum(self.lengths)
        self.dim = self.shards[0].shape[-1]
        self.dtype = self.shards[0].dtype

    def __getitem__(self, ids):
        if isinstance(ids, int):
            return self.__getitem__([ids])[0]
        ids = np.arange(len(self))[ids]

        shard_idx = np.searchsorted(self.cum_lengths, ids, side='right')
        results = np.zeros((len(shard_idx), self.dim), dtype=self.dtype)

        for shard_id in np.unique(shard_idx):
            ids_mask = shard_idx == shard_id
            results[ids_mask] = self.shards[shard_id][ids[ids_mask] - self.cum_lengths[shard_id]]
        return results

    def __len__(self):
        return self.cum_lengths[-1]

    @property
    def shape(self):
        return (self.cum_lengths[-1], self.dim)

    def numpy(self):
        return self.__getitem__(slice(0, len(self)))

    def to_tensor(self, dtype, device):
        return torch.tensor(self.numpy(), device=device, dtype=dtype)


class MDSPseudoMemMap(LocalDataset):
    def __init__(self, path: str, field="embedding"):
        """
        Parameters:
            X: memmap to a numy array, or an array
            indices: array, indices representing the slice
        """
        super().__init__(path)
        self.field = field

    def __getitem__(self, ids):
        result = super().__getitem__(ids)
        if isinstance(result, dict):
            return result[self.field]
        elif isinstance(result[0], dict):
            return np.stack([r[self.field] for r in result])
        else:
            return np.stack(result)

    @property
    def shape(self):
        return (len(self), len(self[0]))

    def numpy(self):
        return self.__getitem__(slice(0, len(self)))

    def to_tensor(self, dtype, device):
        return torch.tensor(self.numpy(), device=device, dtype=dtype)



def create_clusters_from_cluster_assignment(
    cluster_assignment: np.array,
    num_clusters: int,
    return_object_array: bool = True,
):
    """
    Build clusters from cluster assignment.
    """
    ID = np.argsort(cluster_assignment)
    sorted_cluster_assigment = cluster_assignment[ID]
    index_split = np.searchsorted(sorted_cluster_assigment, list(range(num_clusters)))
    clusters = np.split(ID, index_split[1:])
    if return_object_array:
        return np.array(clusters, dtype=object)
    else:
        return clusters


def find_all_checkpoints(save_dir, pattern):
    """
    Parameters:
        pattern: str
            checkpoint name format <filename>_%d.<file extension>,
            e.g., kmpp_checkpoint_%d.pth
    """
    save_dir = Path(save_dir)
    ckpt_list = [str(el.stem) for el in save_dir.glob(pattern.replace("%d", "*"))]
    ckpt_list = [int(el.split("_")[-1]) for el in ckpt_list]
    ckpt_list = sorted(ckpt_list)
    return [Path(save_dir, pattern % el) for el in ckpt_list]


def get_last_valid_checkpoint(save_dir, pattern):
    """
    Find path to the last checkpoint.
    """
    ckpt_list = find_all_checkpoints(save_dir, pattern)
    for ckpt_path in ckpt_list[::-1]:
        try:
            if ".pth" in pattern:
                _ = torch.load(ckpt_path, map_location="cpu")
            elif ".npy" in pattern:
                _ = np.load(ckpt_path)
            else:
                raise ValueError("Pattern not recognized!")
            return ckpt_path
        except Exception:
            continue
    return None


def _delete_old_checkpoint(
    save_dir, current_iter, checkpoint_period, max_num_checkpoints, pattern
):
    Path(
        save_dir, pattern % (current_iter - checkpoint_period * max_num_checkpoints)
    ).unlink(missing_ok=True)


def setup_logging(
    *,
    name: str = None,
    level: int = logging.INFO,
    capture_warnings: bool = True,
) -> None:
    """
    Basic setting for logger.
    """
    logging.captureWarnings(capture_warnings)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        return

    fmt_prefix = (
        "%(levelname).1s%(asctime)s %(process)s %(name)s %(filename)s:%(lineno)s] "
    )
    fmt_message = "%(message)s"
    fmt = fmt_prefix + fmt_message
    datefmt = "%Y%m%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)

    logger.propagate = False
    logger.addHandler(handler)
    return
