import argparse

import numpy as np
import wandb
from datasets import load_dataset, load_metric
from environs import Env
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from transformers.models.my_fnet import MyFNetForSequenceClassification

GLUE_TASKS = [
    "cola",
    "mnli",
    "mnli-mm",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
    "wnli",
]

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def run_training(pretrained_model_path, task, batch_size):
    env = Env()
    env.read_env()
    write_hub_token = env("write_hub_token")

    actual_task = "mnli" if task == "mnli-mm" else task

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased", use_fast=True, use_auth_token=write_hub_token
    )
    sentence1_key, sentence2_key = task_to_keys[actual_task]

    def tokenize_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True)
        return tokenizer(
            examples[sentence1_key], examples[sentence2_key], truncation=True
        )

    dataset = load_dataset("glue", actual_task)
    encoded_dataset = dataset.map(tokenize_function, batched=True)

    if pretrained_model_path == "Joqsan/custom-fnet":
        ModelForSequenceClassification = MyFNetForSequenceClassification
    elif pretrained_model_path == "bert-base-uncased":
        ModelForSequenceClassification = BertForSequenceClassification
    else:
        ValueError("Non-valid checkpoint path")

    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
    
    model = ModelForSequenceClassification.from_pretrained(
        pretrained_model_path, num_labels=num_labels, use_auth_token=write_hub_token
    )

    model_name = pretrained_model_path.split("/")[-1]
    metric_name = (
        "pearson"
        if task == "stsb"
        else "matthews_correlation"
        if task == "cola"
        else "accuracy"
    )

    metric = load_metric("glue", actual_task)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    args = TrainingArguments(
        f"{model_name}-finetuned-{task}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        num_train_epochs=5,
        weight_decay=0.01,
        push_to_hub=True,
        hub_strategy="end",
        hub_token=write_hub_token,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to="wandb",
    )

    validation_key = (
        "validation_mismatched"
        if task == "mnli-mm"
        else "validation_matched"
        if task == "mnli"
        else "validation"
    )

    run = wandb.init(
        project=env("WANDB_PROJECT"),
        name=f"{model_name}-{actual_task}",
        reinit=True,
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    run.finish()


def get_parser():
    parser = argparse.ArgumentParser(
        description="Program to run fine-tuning on GLUE tasks"
    )

    parser.add_argument(
        "pretrained_model_path", choices=["bert-base-uncased", "Joqsan/custom-fnet"]
    )
    parser.add_argument("task", choices=GLUE_TASKS)
    parser.add_argument("--batch_size", type=int, default=16)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    run_training(args.pretrained_model_path, args.task, args.batch_size)


if __name__ == "__main__":
    main()
