#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 16:37:29 2023

@author: liyingqiu
"""

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DebertaModel
import evaluate
import numpy as np

seqeval = evaluate.load("seqeval")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base", add_prefix_space=True)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Some useful functions

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# Create mapping

id2label = {
    0: "O",
    1: "B-MethodName",
    2: "I-MethodName",
    3: "B-HyperparameterName",
    4: "I-HyperparameterName",
    5: "B-HyperparameterValue",
    6: "I-HyperparameterValue",
    7: "B-MetricName",
    8: "I-MetricName",
    9: "B-MetricValue",
    10: "I-MetricValue",
    11: "B-TaskName",
    12: "I-TaskName",
    13: "B-DatasetName",
    14: "I-DatasetName",
}

label2id = {
    "O": 0,
    "B-MethodName": 1,
    "I-MethodName": 2,
    "B-HyperparameterName": 3,
    "I-HyperparameterName": 4,
    "B-HyperparameterValue": 5,
    "I-HyperparameterValue": 6,
    "B-MetricName": 7,
    "I-MetricName": 8,
    "B-MetricValue": 9,
    "I-MetricValue": 10,
    "B-TaskName": 11,
    "I-TaskName": 12,
    "B-DatasetName": 13,
    "I-DatasetName": 14,
}

def read_conll2003(file_path):
    data = []
    with open(file_path, 'r') as f:
        lines = f.read().strip().split('\n\n')
        for i, sentence in enumerate(lines):
            tokens = []
            ner_tags = []
            for line in sentence.split('\n'):
                parts = line.split()
                if parts:
                    tokens.append(parts[0])
                    ner_tags.append(parts[-1])
            data.append({"id": str(i), "tokens": tokens, "ner_tags": ner_tags})
    return data

# Path to the CoNLL-2003 file
file_path = 'Annontated Data/Training Part 1.conll'
file_path = 'Annontated Data/TrainPlus.conll'

# Read the CoNLL-2003 file
train_data = read_conll2003(file_path)

file_path = 'Annontated Data/2023_acl_long_114.conll'
file_path = 'Annontated Data/TestPlus.conll'

# Read the CoNLL-2003 file
test_data = read_conll2003(file_path)

from datasets import Dataset, DatasetDict, ClassLabel, Sequence, Features

ner_tags_feature = Sequence(
    feature=ClassLabel(names=["O","B-MethodName","I-MethodName","B-HyperparameterName","I-HyperparameterName","B-HyperparameterValue","I-HyperparameterValue","B-MetricName","I-MetricName","B-MetricValue","I-MetricValue","B-TaskName","I-TaskName","B-DatasetName","I-DatasetName"], id=None),
    length=-1,
    id=None
)

train = Dataset.from_dict({
    "id": [example["id"] for example in train_data],
    "tokens": [example["tokens"] for example in train_data],
    "ner_tags": [example["ner_tags"] for example in train_data],
})


test = Dataset.from_dict({
    "id": [example["id"] for example in test_data],
    "tokens": [example["tokens"] for example in test_data],
    "ner_tags": [example["ner_tags"] for example in test_data],
})

dataset = DatasetDict({
    "train": train,
    "test": test
})

for split in dataset:
    dataset[split] = dataset[split].map(lambda example: {"ner_tags": [label2id[tag] for tag in example["ner_tags"]]})


tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
label_list = ["O","B-MethodName","I-MethodName","B-HyperparameterName","I-HyperparameterName","B-HyperparameterValue","I-HyperparameterValue","B-MetricName","I-MetricName","B-MetricValue","I-MetricValue","B-TaskName","I-TaskName","B-DatasetName","I-DatasetName"]


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {"f1_score": results["overall_f1"]}

model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=15, id2label=id2label, label2id=label2id
)

# model = AutoModelForTokenClassification.from_pretrained(
#     "bert-base-uncased", num_labels=15, id2label=id2label, label2id=label2id
# )

# model = AutoModelForTokenClassification.from_pretrained(
#     "roberta-base", num_labels=15, id2label=id2label, label2id=label2id
# )

# model = AutoModelForTokenClassification.from_pretrained(
#     "microsoft/deberta-base", num_labels=15, id2label=id2label, label2id=label2id
# )

training_args = TrainingArguments(
    output_dir="fine_tune_copy_test",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

















