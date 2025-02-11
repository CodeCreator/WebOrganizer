import os
from typing import Dict, List, Callable
from functools import partial

from tqdm import tqdm

from pathlib import Path

from simple_parsing import ArgumentParser, field
from datatools.process import process, ProcessOptions
from datatools.load import load, LoadOptions
from transformers import AutoTokenizer
import numpy as np

def predict_fn(dataset, indices, process_id, tokenizer_name, text_field="text"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    for i in tqdm(range(len(dataset)), disable=(process_id != 0)):
        text = dataset[i][text_field]
        tokens = tokenizer.encode(text)
        num_tokens = len(tokens) + 1  # + 1 for <bos>
        yield {
            "": num_tokens,
            # "bin": np.clip(np.log2(num_tokens).astype(int), 6, 11)
        }


if __name__ == "__main__":
    parser = ArgumentParser(add_config_path_arg=True)

    parser.add_argument("inputs", type=Path, nargs="+", help="Input dataset paths")
    parser.add_argument("output", type=Path, help="Output dataset path")

    parser.add_argument("--text_field", type=str, default="text", help="Name of the field containing the text to classify")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b", help="Path to the FastText model")

    parser.add_arguments(LoadOptions, dest="load_options")
    parser.add_arguments(ProcessOptions, dest="process_options")

    args = parser.parse_args()
    args.process_options.ndarray = True

    print("Arguments:", args)
    dataset = load(*args.inputs, options=args.load_options)
    N = len(dataset)
    print(f"Loaded dataset with {N} samples")


    process(
        dataset,
        partial(
            predict_fn,
            text_field=args.text_field,
            tokenizer_name=args.tokenizer
        ),
        args.output, args.process_options
    )
