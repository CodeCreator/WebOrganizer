from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader
import numpy

from datatools.process import process, ProcessOptions
from datatools.load import load, LoadOptions
from simple_parsing import ArgumentParser, field
from typing import Dict, Any


@dataclass
class PerplexityOptions:
    model_name: str = "EleutherAI/pythia-160m"
    batch_size: int = 32
    num_dataloader_workers: int = 8
    max_length: int = 2048


class DataCollator:
    def __init__(self, max_length):
        self.max_length = max_length

    @torch.no_grad()
    def __call__(self, features):
        bsz = len(features)
        seqs = [features[i]["input_ids"] for i in range(bsz)]
        max_length = min(max(len(seq) for seq in seqs), self.max_length)

        input_ids = torch.zeros(bsz, max_length, dtype=torch.long)
        attention_mask = torch.zeros(bsz, max_length, dtype=torch.long)

        for i, seq in enumerate(seqs):
            seq = seq[:max_length]
            input_ids[i, :len(seq)] = torch.tensor(seq)
            attention_mask[i, :len(seq)] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }


@torch.inference_mode()
def predict_fn(subset, indices, process_id, options):
    model = AutoModelForCausalLM.from_pretrained(options.model_name, attn_implementation="flash_attention_2")
    model.to(torch.bfloat16)
    model.cuda()
    model.eval()

    data_loader = DataLoader(subset,
                             batch_size=options.batch_size,
                             collate_fn=DataCollator(options.max_length),
                             num_workers=options.num_dataloader_workers,
                             prefetch_factor=4,
                             pin_memory=True,
                             shuffle=False)

    for batch in tqdm(data_loader, disable=(process_id != 0)):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
        labels = torch.where(attention_mask == 1, input_ids, torch.zeros_like(input_ids) - 100)[:, 1:]
        seq_lens = attention_mask.sum(1)

        seq_losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels, reduction='none').sum(1) / seq_lens

        for seq_loss in seq_losses :
            yield {"": seq_loss.cpu().numpy()}


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("inputs", type=Path, nargs="+", help="Input dataset paths")
    parser.add_argument("output", type=Path, help="Output dataset path")

    parser.add_arguments(PerplexityOptions, dest="embed_options")
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
