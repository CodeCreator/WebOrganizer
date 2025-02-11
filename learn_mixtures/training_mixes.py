import numpy as np
from dataclasses import dataclass

from simple_parsing import ArgumentParser, field
from simple_parsing.helpers import Serializable
from typing import Callable, Dict, Optional, List, Any, Tuple

import json

from tqdm import tqdm

from pathlib import Path


@dataclass
class ScriptOptions(Serializable):
    prior_distribution_file: Path = field(positional=True, metadata={"help": "Path to the input distribution file"})
    output_folder: Path = field(positional=True, metadata={"help": "Output directory to save the generated mixes"})

    seed: int = 42

    prior_temperature: float = 2.0

    min_weight: Optional[int] = 2e-4     # For statistical significance, include at least 200k tokens per domain at the 1B scale
    max_epochs: Optional[int] = 20       # Set this to 20 because when we subsample 15% tokens for 1xC, it will be 3 epochs

    min_dirichlet: float = 0.1
    max_dirichlet: float = 5.0

    num_mixes: int = 512

    min_total_variation_distance: float = 0.05


def generate_mix(prior: np.ndarray, options: ScriptOptions) -> np.ndarray:
    # sample dirichlet parameter in log space
    alpha = np.exp(np.random.uniform(np.log(options.min_dirichlet), np.log(options.max_dirichlet)))
    mix = np.random.dirichlet(alpha * prior)

    # round small components to zero
    mix[mix < options.min_weight] = 0
    mix = mix / mix.sum()

    return mix


def is_valid(proposed_mix: np.ndarray, mixes: np.ndarray, prior: np.ndarray, options: ScriptOptions) -> bool:
    max_epoch = (proposed_mix[prior > 0] / prior[prior > 0]).max()
    if max_epoch > options.max_epochs:
        return False

    if np.any(np.abs(mixes - proposed_mix).max(axis=-1) < options.min_total_variation_distance):
        print("INFO: Rejecting proposed mix due to close proximity to existing mixes")
        return False

    return True


def generate_mixes(prior: np.ndarray, options: ScriptOptions):
    np.random.seed(options.seed)

    # apply temperature to prior
    prior = (prior ** (1/options.prior_temperature))
    prior = prior / prior.sum()

    mixes = np.zeros((0, len(prior)))

    update_bar = tqdm(total=options.num_mixes, desc="Generating mixes")
    while len(mixes) < options.num_mixes:
        proposed_mix = generate_mix(prior, options)
        if is_valid(proposed_mix, mixes, prior, options):
            mixes = np.vstack([mixes, proposed_mix])
            update_bar.update(1)

    return mixes


def generate_and_write_mixes(options: ScriptOptions):
    with open(options.prior_distribution_file, 'r') as f:
        prior_distribution_info = json.load(f)

    prior = np.array([row["weight"] for row in prior_distribution_info])
    prior = prior / prior.sum()

    mixes = generate_mixes(prior, options)

    output_folder = options.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    for i, mix in enumerate(mixes):
        with (output_folder / f"random{i}.json").open("w") as f:
            obj = [
                {"domain": row["domain"], "weight": weight.item()}
                for row, weight in zip(prior_distribution_info, mix)
            ]
            json.dump(obj, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ScriptOptions, dest="options")
    args = parser.parse_args()
    generate_and_write_mixes(args.options)