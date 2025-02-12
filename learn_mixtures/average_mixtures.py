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

    args = argparser.parse_args()
    print(args)

    inputs = args.inputs

    domain_lists = []
    for file in inputs:
        with file.open() as f:
            domain_lists.append(json.load(f))


    output_distribution = []
    for entry in domain_lists[0]:
        domain = entry["domain"]
        entries = [next(e for e in domain_list if e["domain"] == domain) for domain_list in domain_lists]

        output_entry = {
            "domain": domain,
            "weight": sum([entry["weight"] for entry in entries]) / len(entries),
        }

        output_distribution.append(output_entry)

    with args.output.open('w') as f:
        json.dump(output_distribution, f, indent=2)

