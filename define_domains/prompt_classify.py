"""
Usage:
OUTLINES_CACHE_DIR=/tmp/outlines python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --port 30000
python prompt_classify  <input datasets>  <output dataset>  --config_path <config>
"""

import sglang as sgl

from tqdm import tqdm
import numpy as np
import torch

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Dict, Any
import time

from simple_parsing import ArgumentParser, field
from simple_parsing.helpers import Serializable
from datatools.process import process, ProcessOptions
from datatools.load import load, LoadOptions
from retry import retry
from urllib.error import URLError


@dataclass
class PromptConfig(Serializable):
    system_template: Optional[str] = None
    template: Optional[str] = None
    choices: Optional[List[str]] = None
    demonstrations: List[Dict[str, str]] = field(cmd=False, default=None)
    labels: List[str] = field(default_factory=lambda: list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

    response_prefix: Optional[str] = ""
    truncation: int = 50_000  # Truncate the input text to this character length

    randomize_choices: bool = True
    randomize_demonstrations: bool = True
    randomize_seed: int = 42


def get_permutation(index: int, prompt_config: PromptConfig) -> np.ndarray:
    if not prompt_config.randomize_choices:
        return np.arange(len(prompt_config.choices))
    else:
        np.random.seed(index + prompt_config.randomize_seed)
        permutation = np.random.permutation(len(prompt_config.choices))
        return permutation


def get_demonstration_permutation(index: int, prompt_config: PromptConfig) -> np.ndarray:
    if not prompt_config.randomize_demonstrations:
        return np.arange(len(prompt_config.demonstrations))
    else:
        np.random.seed(index + prompt_config.randomize_seed + 1)
        permutation = np.random.permutation(len(prompt_config.demonstrations))
        return permutation


@sgl.function
def classify(s, item: Dict[str, Any], index: int, prompt_config: PromptConfig):
    permutation = get_permutation(index, prompt_config)
    labels = prompt_config.labels[:len(prompt_config.choices)]
    choices = "\n".join(f"{labels[j]}: {prompt_config.choices[i]}" for j, i in enumerate(permutation))

    kwargs = item.copy()
    if len(kwargs["text"]) > prompt_config.truncation:
        kwargs["text"] = kwargs["text"][:prompt_config.truncation] + "... (truncated)"
    kwargs["choices"] = choices

    if prompt_config.system_template is not None:
        s += sgl.system(prompt_config.system_template.format(**kwargs))
    prompt = prompt_config.template.format(**kwargs)

    if prompt_config.demonstrations is not None:
        demonstration_permutation = get_demonstration_permutation(index, prompt_config)
        for j in demonstration_permutation:
            demonstration = prompt_config.demonstrations[j]

            kwargs = demonstration.copy()
            if len(kwargs["text"]) > prompt_config.truncation:
                kwargs["text"] = kwargs["text"][:prompt_config.truncation] + "... (truncated)"
            kwargs["choices"] = choices

            label_index = next(i for i, v in enumerate(prompt_config.choices) if v.startswith(demonstration["choice"]))
            permuted_label_index = np.where(permutation == label_index)[0][0]
            label = labels[permuted_label_index]

            s += sgl.user(prompt_config.template.format(**kwargs))
            if "explanation" in demonstration:
                s += sgl.assistant(prompt_config.response_prefix + label + ": " + demonstration["explanation"])
            else:
                s += sgl.assistant(prompt_config.response_prefix + label)


    s += sgl.user(prompt)
    s += sgl.assistant(prompt_config.response_prefix + sgl.gen("choice", choices=labels))


@retry(URLError, tries=360, backoff=1, delay=5)
def set_default_backend(port=30000):
    sgl.set_default_backend(sgl.RuntimeEndpoint(f"http://localhost:{port}"))


def predict_fn(dataset,
               indices,
               process_id,
               prompt_config,
               num_threads=1,
               batch_size=1000,
               port=30000):
    set_default_backend(port)

    start_time = time.time()

    for batch_start in range(0, len(dataset), batch_size):
        batch_range = list(range(batch_start, min(batch_start + batch_size, len(dataset))))
        print(f"Processing batch {batch_range[0]} - {batch_range[-1]}")

        states = classify.run_batch([
            {"item": dataset[i], "index": indices[i], "prompt_config": prompt_config}
            for i in batch_range
        ], num_threads=num_threads, progress_bar=True)

        # Check for corruption of inference server
        for state in states:
            meta_info = state.get_meta_info("choice")

            assert meta_info is not None and meta_info["normalized_prompt_logprobs"] is not None
            assert all(
                len(answer_tokens) > 1 for answer_tokens in meta_info["input_token_logprobs"]
            ), f"All answers should have at least 2 tokens in {meta_info['input_token_logprobs']}"


        for i, state in zip(batch_range, states):
            demonstration_permutation = get_demonstration_permutation(indices[i], prompt_config)
            permutation = get_permutation(indices[i], prompt_config)
            meta_info = state.get_meta_info("choice")

            # We re-compute answer logprobs, as the first token is the preceding token
            # that is the same for all answers
            permuted_choice_loss = np.array([
                sum(logprob for logprob, token_id, _ in answer_tokens[1:]) / (len(answer_tokens) - 1)
                for answer_tokens in meta_info["input_token_logprobs"]
            ])

            choice_loss = np.zeros_like(permuted_choice_loss)
            choice_loss[permutation] = permuted_choice_loss

            scores = choice_loss - np.max(choice_loss)
            scores = scores - np.log(np.exp(scores).sum())
            probs = np.exp(scores)

            prediction = np.argmax(probs)

            yield {
                **dataset[i],
                "choice_loss": choice_loss,
                "choice_probs": probs,
                "top_choice": prompt_config.choices[prediction],
                "top_choice_index": prediction,
                "top_choice_prob": probs[prediction],
                "label_permutation": permutation,
                "fewshot_permutation": demonstration_permutation,
            }

    print(f"Time taken: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("inputs", type=Path, nargs="+", help="Input dataset paths")
    parser.add_argument("output", type=Path, help="Output dataset path")

    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of threads to use")
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of threads to use")
    parser.add_argument("--port", type=int, default=30000, help="Number of threads to use")
    parser.add_argument("--randomize_seed", default=None, type=int, help="Seed for randomization")

    parser.add_arguments(LoadOptions, dest="load_options")
    parser.add_arguments(ProcessOptions, dest="process_options")

    args = parser.parse_args()
    prompt_config = PromptConfig.load_yaml(args.config_path)

    if args.randomize_seed is not None:
        prompt_config.randomize_seed = args.randomize_seed

    args.prompt_config = prompt_config


    print("Arguments:", args)
    dataset = load(*args.inputs, options=args.load_options)
    N = len(dataset)
    print(f"Loaded dataset with {N} samples")

    process(
        dataset,
        partial(
            predict_fn,
            prompt_config=prompt_config,
            num_threads=args.num_threads,
            batch_size=args.batch_size,
            port=args.port
        ),
        args.output, args.process_options
    )
