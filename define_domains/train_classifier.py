import logging
import os
import sys
import torch
import datasets
import transformers
import numpy as np
import pickle

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    set_seed,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    PrinterCallback
)

import time

from datatools import load

from transformers.trainer_utils import get_last_checkpoint
from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict, Tuple


logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments(TrainingArguments):
    model_name: Optional[str] = field(
        default="Alibaba-NLP/gte-base-en-v1.5",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code"})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    train_dataset: str = field(default_factory=list, metadata={"help": "Path to training datasets"})
    validation_dataset: List[str] = field(default_factory=list, metadata={"help": "Paths to validation datasets"})
    test_dataset: str = field(default=None, metadata={"help": "Path to test dataset"})

    max_length: int = field(default=8192, metadata={"help": "Max length of input sequences"})

    label_field: List[str] = field(default_factory=lambda: ["labels"], metadata={"help": "Field name for text. Supply back-off labels with multiple"})

    template: str = field(default="{text}", metadata={"help": "Template for the text input"})
    unpad_inputs: bool = field(default=False, metadata={"help": "Unpad inputs"})
    use_memory_efficient_attention: bool = field(default=False, metadata={"help": "Use memory efficient attention"})

    label_temperature: float = field(default=1.0, metadata={"help": "Temperature applied to soft labels"})

    freeze_encoder: bool = field(default=False, metadata={"help": "Freeze the encoder"})


class LogCallback(TrainerCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = None
        self.last_log_time = None
        self.is_training = False

        self.first_step_of_run = 0

    def on_step_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self.last_log_time is None:
            self.last_step = 0
            self.start_time = time.time()
            self.last_log_time = self.start_time
            self.first_step_of_run = state.global_step
            self.last_tokens_seen = state.num_input_tokens_seen

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)

        if state.is_world_process_zero:
            if self.is_training:
                current_time = time.time()
                time_diff = current_time - self.last_log_time

                self.last_log_time = current_time
                steps_completed = max(state.global_step, 1)

                steps_since_first = max(1, state.global_step - self.first_step_of_run)
                self.last_step = state.global_step

                tokens_seen_since_last = state.num_input_tokens_seen - self.last_tokens_seen
                self.last_tokens_seen = state.num_input_tokens_seen

                remaining_steps = state.max_steps - steps_completed
                pct_completed = (steps_completed / state.max_steps) * 100
                time_since_start = current_time - self.start_time
                remaining_time = (time_since_start / steps_since_first) * remaining_steps

                gpu_mem_free, _ = torch.cuda.mem_get_info(device=args.device)

                update = {
                    "completed": f"{pct_completed:.2f}% ({steps_completed:_} / {state.max_steps:_})",
                    "remaining time": self.format_duration(remaining_time),
                    "throughput": f"{tokens_seen_since_last / time_diff:.2f}",
                    "gpu_mem_free": f"{gpu_mem_free / 1024 / 1024:.0f}MB",
                }

                logger.info(str({**logs, **update}))
            else:
                logger.info(str(logs))

    def on_train_begin(self, args, state, control, **kwargs):
        args.include_num_input_tokens_seen = True

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.is_training = True

    def on_prediction_step(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.is_training = False

    @staticmethod
    def format_duration(seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"


def get_labels(item, label_fields):
    for label_fields in label_fields:
        if label_fields in item:
            return item[label_fields]


class DataCollator:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = self.args.max_length

    @torch.no_grad()
    def __call__(self, items) -> Dict[str, Any]:
        batch = self.tokenizer(
            [self.args.template.format(**item) for item in items],
            truncation=True, return_tensors="pt", padding=True, max_length=self.max_length
        )

        if isinstance(get_labels(items[0], self.args.label_field), np.ndarray):
            labels = torch.tensor(np.stack([get_labels(item, self.args.label_field) for item in items]))
            labels = labels ** (1/self.args.label_temperature)
            labels = labels / labels.sum(dim=-1, keepdim=True)
        else:
            labels = torch.tensor([get_labels(item, self.args.label_field) for item in items])

        batch["labels"] = labels
        return batch


class SoftClassificationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            self.remove_callback(PrinterCallback)
            self.add_callback(LogCallback)
            # self.add_callback(SIGUSR1Callback(self))
        except ValueError:
            logger.warn("Couldn't remove PrinterCallback")

        self.data_collator = DataCollator(self.args, self.tokenizer)
        self.compute_metrics = self.compute_soft_classification_metrics

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        outputs = model(**inputs)

        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss

    def compute_soft_classification_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        top_label = np.argmax(labels, axis=-1)

        conf50_mask = (np.max(labels, axis=-1) > 0.5)
        conf75_mask = (np.max(labels, axis=-1) > 0.75)

        correct = (predictions == top_label)

        metrics = {
            "accuracy": correct.mean().item(),
            "accuracy_conf50": correct[conf50_mask].mean().item(),
            "proportion_conf50": conf50_mask.mean().item(),
            "accuracy_conf75": correct[conf75_mask].mean().item(),
            "proportion_conf75": conf75_mask.mean().item(),
        }

        for i in range(logits.shape[-1]):
            metrics[f"accuracy__{i}"] = (predictions == i)[top_label == i].mean().item()
            metrics[f"accuracy_conf50__{i}"] = (predictions == i)[(top_label == i) & conf50_mask].mean().item()
            metrics[f"accuracy_conf75__{i}"] = (predictions == i)[(top_label == i) & conf75_mask].mean().item()

        metrics["accuracy_label_average"] = sum([metrics[f"accuracy__{i}"] for i in range(logits.shape[-1])]) / logits.shape[-1]
        metrics["accuracy_label_average_conf50"] = sum([metrics[f"accuracy_conf50__{i}"] for i in range(logits.shape[-1])]) / logits.shape[-1]
        metrics["accuracy_label_average_conf75"] = sum([metrics[f"accuracy_conf75__{i}"] for i in range(logits.shape[-1])]) / logits.shape[-1]

        metrics["accuracy_label_min"] = min([metrics[f"accuracy__{i}"] for i in range(logits.shape[-1])])
        metrics["accuracy_label_min_conf50"] = min([metrics[f"accuracy_conf50__{i}"] for i in range(logits.shape[-1])])
        metrics["accuracy_label_min_conf75"] = min([metrics[f"accuracy_conf75__{i}"] for i in range(logits.shape[-1])])

        return metrics


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    parser = HfArgumentParser(ScriptArguments)
    args, = parser.parse_args_into_dataclasses()
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {args.local_rank}, device: {args.device}, n_gpu: {args.n_gpu}"
        + f"distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {args}")
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is not None and args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


    # Set seed before initializing model.
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
    )
    config = AutoConfig.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )

    if args.do_train:
        train_dataset = load(args.train_dataset)

        config.num_labels = len(get_labels(train_dataset[0], args.label_field))

    if args.do_eval:
        eval_dataset = {
            x.split("/")[-1]: load(x)
            for x in args.validation_dataset
        }

    if args.do_predict:
        test_dataset = load(args.test_dataset)

    if args.use_memory_efficient_attention:
        config.use_memory_efficient_attention = True
    if args.unpad_inputs:
        config.unpad_inputs = True

    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = 0
        tokenizer.pad_token_id = 0

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        from_tf=bool(".ckpt" in args.model_name),
        config=config,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else None),
    )


    if args.freeze_encoder:
        assert args.model_name.startswith("Alibaba-NLP")
        for param in model.new.parameters():
            param.requires_grad = False

    logger.info(f"Model: {model}")


    # Initialize our Trainer
    trainer = SoftClassificationTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        tokenizer=tokenizer,
    )

    if trainer.is_fsdp_enabled:
        import functools
        from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

        def layer_policy_fn(module):
            return "layer" in module.__class__.__name__.lower()

        auto_wrap_policy = functools.partial(lambda_auto_wrap_policy,
                                             lambda_fn=layer_policy_fn)
        trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = auto_wrap_policy


    # Training
    if args.do_train:
        checkpoint = None
        if args.resume_from_checkpoint is not None:
            checkpoint = args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if args.do_predict:
        logger.info("*** Predict ***")
        output = trainer.predict(test_dataset)

        with open(os.path.join(args.output_dir, "predictions.pkl"), "wb") as f:
            pickle.dump((output.predictions, output.label_ids), f)

        trainer.log_metrics("pred", output.metrics)
        trainer.save_metrics("pred", output.metrics)


if __name__ == "__main__":
    main()
