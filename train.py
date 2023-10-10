import os
import sys
import json
import random
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import transformers
from transformers import AutoTokenizer, AutoModel
from transformers import HfArgumentParser, TrainingArguments

from datasets import Dataset

from trainer import SemanticTrainer
from model import SemanticModel

from utils import compute_metrics, save_metrics, subsampling
from arguments import ModelArguments, DataTrainingArguments


logger = logging.getLogger(__name__)

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# The default of training_args.log_level is passive, so we set log level at info here to have that default.
# transformers.utils.logging.set_verbosity_info()

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
# datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

logger.warning('***** Arguments *****')
logger.warning(model_args)
logger.warning(data_args)
logger.warning('***** Arguments *****\n\n')


path = {
    '20Newsgroups': 'data/20Newsgroups',
    'BGC': 'data/BGC',
    'WOS': 'data/WOS',
    'RCV1-v2-50%*1': 'data/RCV1-v2/50%*1',
    'RCV1-v2-25%*3': 'data/RCV1-v2/25%*3',
    'Reuters-21578-50%*1': 'data/Reuters-21578/50%*1',
    'Reuters-21578-25%*3': 'data/Reuters-21578/25%*3',
    'AAPD-50%*1': 'data/AAPD/50%*1',
    'AAPD-25%*3': 'data/AAPD/25%*3',
    'freecode-98%*1': 'data/freecode/98%*1',
    'EUR-Lex-80%*1': 'data/EUR-Lex/80%*1'
}

multi_label = True if data_args.dataset_name in {
    'RCV1-v2-50%*1', 'RCV1-v2-25%*3', 
    'Reuters-21578-50%*1', 'Reuters-21578-25%*3', 
    'AAPD-50%*1', 'AAPD-25%*3',
    'freecode-98%*1', 
    'EUR-Lex-80%*1'
} else False

def load_label_list(path):
    with open(os.path.join(path, 'label2id.json')) as f:
        label2id = json.load(f)
        label2id = {k: int(v) for k, v in label2id.items()}
        id2label = {v: k for k, v in label2id.items()}

    labels = list(label2id.keys())

    return label2id, id2label, labels


label2id, id2label, labels = load_label_list(path[data_args.dataset_name])


def load_data(path, mode):
    if data_args.with_example:
        with open(os.path.join(path, f"{mode}-with-example.json")) as f:
            file = json.load(f)
    else:
        with open(os.path.join(path, f"{mode}.json")) as f:
            file = json.load(f)


    if mode == 'train':
        file = subsampling(file, data_args.subsampling_rate)
        if not data_args.stratified_sampling:
            random.shuffle(file)
        # Pairwise
        global_idx = 0
        label_name = 'labels' if multi_label else 'label'
        pairs = []
        for line in file:
            pairs.append({
                'text': line['text'],
                'label': line[label_name][0] if isinstance(line[label_name], list) else line[label_name],
                'global_idx': global_idx
            })
            global_idx += 1

            pairs.append({
                'text': line['label_description'],
                'label': line[label_name][0] if isinstance(line[label_name], list) else line[label_name],
                'global_idx': global_idx
            })
            global_idx += 1

        dataset = Dataset.from_list(pairs)

    elif mode == 'test':
        dataset = Dataset.from_list(file)

    return dataset


train_ds = load_data(path[data_args.dataset_name], 'train')
test_ds = load_data(path[data_args.dataset_name], 'test')


tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
pretrained_model = AutoModel.from_pretrained(
    model_args.model_name_or_path,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)


def preprocess_data(examples):
    encoding = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=data_args.text_max_length)
    encoding['labels'] = examples['label']

    if model_args.centroid:
        encoding['global_idxs'] = examples['global_idx']

    return encoding


def preprocess_data_multi_label_dev(examples):
    encoding = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=data_args.text_max_length)
    encoding['labels'] = examples['labels']
    for i in range(len(encoding['labels'])):
        if len(encoding['labels'][i]) < len(labels):
            encoding['labels'][i] += [-1] * (len(labels) - len(encoding['labels'][i]))
    return encoding


train_tokenized_ds = train_ds.map(preprocess_data, batched=True, remove_columns=train_ds.column_names, num_proc=data_args.num_proc)
if multi_label:
    test_tokenized_ds = test_ds.map(preprocess_data_multi_label_dev, batched=True, remove_columns=test_ds.column_names, num_proc=data_args.num_proc)
else:
    test_tokenized_ds = test_ds.map(preprocess_data, batched=True, remove_columns=test_ds.column_names, num_proc=data_args.num_proc)

train_tokenized_ds.set_format("torch")
test_tokenized_ds.set_format("torch")

# Calculate the centroids if use centroid
if model_args.centroid:
    index_table = dict()
    centroids = dict()

    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    train_data_loader = DataLoader(train_tokenized_ds, batch_size=training_args.per_device_train_batch_size)

    pretrained_model.to(device)
    pretrained_model.eval()

    for batch in tqdm(train_data_loader, 'Calculate the centroids'):
        input_ids = batch['input_ids'].to(pretrained_model.device)
        with torch.no_grad():
            outputs = pretrained_model(input_ids)
        logits = outputs.last_hidden_state[:, 0]
        for i in range(len(batch['global_idxs'])):
            global_idx = batch['global_idxs'][i].item()
            label = batch['labels'][i].item()
            if label not in centroids:
                centroids[label] = logits[i].detach().cpu().unsqueeze(dim=0)
            else:
                centroids[label] = torch.cat((centroids[label], logits[i].detach().cpu().unsqueeze(dim=0)))
            index_table[global_idx] = (label, len(centroids[label]) - 1)

    pretrained_model.train()

else:
    index_table, centroids = None, None


model = SemanticModel(
    pretrained_model.config, 
    pretrained_model, 
    alpha=model_args.alpha, 
    beta=model_args.beta,
    gamma=model_args.gamma,
    index_table=index_table,
    centroids=centroids,
    momentum_rate=model_args.momentum_rate,
)

hyper_params={
    'model_name': model_args.model_name_or_path,
    'dataset_name': '-'.join(data_args.dataset_name.split('-')[:-1]),
    'subsampling_rate': data_args.subsampling_rate,
    'batch_size': training_args.per_device_train_batch_size,
    'epoch': training_args.num_train_epochs,
    'learning_rate': training_args.learning_rate,
    'text_max_length': data_args.text_max_length,
    'weight_decay': training_args.weight_decay,
    'dataset_state': data_args.dataset_name.split('-')[-1].replace('*', ' * '),
    'semantic_label': True,
    'stratified_sampling': data_args.stratified_sampling,
    'with_example': data_args.with_example,
    'alpha': model_args.alpha,
    'beta': model_args.beta,
    'gamma': model_args.gamma,
    'momentum_rate': model_args.momentum_rate,
    'cls_loss': model_args.cls_loss,
    'centroid': model_args.centroid,
}

trainer = SemanticTrainer(
    model,
    training_args,
    train_dataset=train_tokenized_ds,
    eval_dataset=test_tokenized_ds,
    compute_metrics=compute_metrics
)

trainer.train()

metrics = trainer.evaluate()

with open(f"efficiency_analysis/{data_args.dataset_name}-{'Roberta' if 'roberta' in model_args.model_name_or_path else 'Bert'}.log", 'w') as f:
    json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)

save_metrics('output/result.csv', {**metrics, **hyper_params})
