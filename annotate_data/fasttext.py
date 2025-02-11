import os
from typing import Dict, List, Callable
from functools import partial

import fasttext
from tqdm import tqdm

from pathlib import Path

from simple_parsing import ArgumentParser, field
from datatools.process import process, ProcessOptions
from datatools.load import load, LoadOptions


def classify_fasttext_hq_prob(model: fasttext.FastText._FastText, content: str):
    # Clean the input text by joining all lines into a single string
    text = " ".join(content.strip().splitlines())

    # Make the prediction
    pred = model.predict(text)

    # Extract the predicted label and its probability
    (pred_label, pred_prob) = pred
    pred_label = pred_label[0]
    hq_prob = pred_prob[0]

    # If the predicted label is 'CC', adjust the probability of it being 'Wikipedia'
    if pred_label == "__label__cc":
        hq_prob = 1 - hq_prob

    # Return the output
    return hq_prob


def predict_fn(dataset, indices, process_id, model_path, text_field="text"):
    model = fasttext.load_model(model_path)

    for i in tqdm(range(len(dataset)), disable=(process_id != 0)):
        text = dataset[i][text_field]
        hq_prob = classify_fasttext_hq_prob(model, text)
        yield {"": hq_prob}


if __name__ == "__main__":
    parser = ArgumentParser(add_config_path_arg=True)

    parser.add_argument("inputs", type=Path, nargs="+", help="Input dataset paths")
    parser.add_argument("output", type=Path, help="Output dataset path")


    parser.add_argument("--text_field", type=str, default="text", help="Name of the field containing the text to classify")
    parser.add_argument("--model_path", type=str, default="fasttext_oh_eli5.bin", help="Path to the FastText model")

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
            model_path=args.model_path
        ),
        args.output, args.process_options
    )
