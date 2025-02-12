import json
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import pickle
import os
from pathlib import Path
import argparse
from itertools import product
import numpy as np

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--inputs', type=Path, required=True, nargs='+')
    argparser.add_argument('--output', type=Path, required=True)

    argparser.add_argument('--corpus_distribution', type=Path, default=None)
    # We roughly select 30 out of 200 domains to be present in the mixture.
    argparser.add_argument('--selection_fraction', type=float, default=30/200)

    args = argparser.parse_args()

    inputs = args.inputs

    domain_lists = []
    for file in inputs:
        with file.open() as f:
            domain_lists.append(json.load(f))

    # Assume independence
    independent_domain_list = []
    for domain_objs in product(*domain_lists):
        combined_obj = {}
        for key in domain_objs[0]:
            if key == "weight":
                combined_obj[key] = np.prod(list(domain_obj[key] for domain_obj in domain_objs)).item()
            else:
                combined_obj[key] = list(domain_obj[key] for domain_obj in domain_objs)
        independent_domain_list.append(combined_obj)

    if args.corpus_distribution is not None:
        corpus_distribution = json.load(args.corpus_distribution.open())
        for domain in independent_domain_list:
            ref_domain = next((d for d in corpus_distribution if d["domain"] == domain["domain"]), None)
            if ref_domain is None:
                raise ValueError(f"Domain {domain['domain']} not found in reference distribution")

            domain["weight"] = min(domain["weight"], ref_domain["weight"] / args.selection_fraction)

        total_weight = sum(domain["weight"] for domain in independent_domain_list)
        for domain in independent_domain_list:
            domain["weight"] = domain["weight"] / total_weight

    with args.output.open('w') as f:
        json.dump(independent_domain_list, f, indent=2)

