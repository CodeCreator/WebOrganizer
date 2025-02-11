from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tqdm import tqdm

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.utils.data import DataLoader
import numpy

from datatools.process import process, ProcessOptions
from datatools.load import load, LoadOptions
from simple_parsing import ArgumentParser, field
from typing import Dict, Any

@dataclass
class EmbedOptions:
    model_name: str = "HuggingFaceTB/fineweb-edu-classifier"
    batch_size: int = 128
    num_dataloader_workers: int = 8
    max_length: int = 512
    input_template: str = "{text}"


class DataCollator:
    def __init__(self, tokenizer, options):
        self.tokenizer = tokenizer
        self.options = options

    @torch.no_grad()
    def __call__(self, features):
        documents = [self.options.input_template.format(**f) for f in features]
        return self.tokenizer(documents, padding=True, truncation=True, return_tensors='pt', max_length=self.options.max_length)


def load_model_and_tokenizer(options):
    tokenizer = AutoTokenizer.from_pretrained(options.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(options.model_name)
    return model, tokenizer



@torch.inference_mode()
def predict_fn(subset, indices, process_id, options):

    model, tokenizer = load_model_and_tokenizer(options)
    model.to(torch.bfloat16)
    model.cuda()
    model.eval()

    data_loader = DataLoader(subset,
                             batch_size=options.batch_size,
                             collate_fn=DataCollator(tokenizer,  options),
                             num_workers=options.num_dataloader_workers,
                             prefetch_factor=4,
                             pin_memory=True,
                             shuffle=False)

    for batch in tqdm(data_loader, disable=(process_id != 0)):
        for key in batch:
            batch[key] = batch[key].cuda()

        model_output = model(**batch)

        scores = model_output.logits.squeeze(-1).float().cpu().detach().numpy()

        for seq_score in scores:
            yield {
                "score": seq_score.item(),
                "int_score": int(round(max(0, min(seq_score.item(), 5))))
            }


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("inputs", type=Path, nargs="+", help="Input dataset paths")
    parser.add_argument("output", type=Path, help="Output dataset path")

    parser.add_arguments(EmbedOptions, dest="embed_options")
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
        partial(predict_fn, options=args.embed_options),
        args.output,
        args.process_options
    )
