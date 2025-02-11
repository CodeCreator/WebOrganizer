from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer
import torch
from torch.utils.data import DataLoader
import numpy

from datatools.process import process, ProcessOptions
from datatools.load import load, LoadOptions
from simple_parsing import ArgumentParser, field
from typing import Dict, Any

@dataclass
class EmbedOptions:
    model_name: str = "Alibaba-NLP/gte-base-en-v1.5"
    batch_size: int = 128
    num_dataloader_workers: int = 8
    pooling_strategy: str = "cls"
    normalize_embeddings: bool = True
    max_length: int = 8192
    input_template: str = "{text}"


class DataCollator:
    def __init__(self, tokenizer, options):
        self.tokenizer = tokenizer
        self.options = options

    @torch.no_grad()
    def __call__(self, features):
        documents = [self.options.input_template.format(**f) for f in features]
        return self.tokenizer(documents, padding=True, truncation=True, return_tensors='pt', max_length=self.options.max_length)


@torch.inference_mode()
def pooling(model_output, attention_mask, pooling_strategy):
    if pooling_strategy == "cls":
        return model_output.last_hidden_state[:, 0].float()
    elif pooling_strategy == "mean":
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def load_model_and_tokenizer(options):
    if options.model_name.startswith("nomic-ai/"):
        try:
            from contrastors.models.encoder.modeling_nomic_bert import NomicBertModel
        except:
            raise ImportError("Could not import NomicBertModel. Please install the https://github.com/nomic-ai/contrastors in this folder")

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = NomicBertModel.from_pretrained('nomic-ai/nomic-embed-text-v1', add_pooling_layer=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(options.model_name)
        model = AutoModel.from_pretrained(options.model_name, trust_remote_code=True)
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
        embeddings = pooling(model_output, batch['attention_mask'], options.pooling_strategy)

        if options.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        for embed in embeddings:
            yield {"": embed.cpu().numpy()}


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
