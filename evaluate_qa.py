import re
import numpy as np
import pandas as pd
import warnings
import nltk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import classification_report, f1_score
from datasets import Dataset, load_dataset
from dataloaders import ClimaText, SciDCC, CDPCities, ClimateStance, ClimateEng, ClimateInsurance, ClimateInsuranceMulti, CDPQA
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import itertools
import copy
from sklearn.metrics import ndcg_score
import argparse
import pickle

parser = argparse.ArgumentParser()

parser.add_argument("--task", default='CDPCitiesQA',
                    help="Tasks like CDPCitiesQA, CDPStatesQA")
parser.add_argument("--model", type=str, default='cross-encoder/ms-marco-MiniLM-L-12-v2',
                    help="huggingface model to test")
parser.add_argument("--tokenizer", type=str, default='cross-encoder/ms-marco-MiniLM-L-12-v2',
                    help="huggingface tokenizer to ttest")
parser.add_argument("--max-len", type=int, default=512,
                    help="huggingface model max length")
parser.add_argument("--bs", default=64, type=int,
                    help="batch size")
args = parser.parse_args()


def mrr_at_k(grouped_labels, grouped_scores, k):
    all_mrr_scores = []
    for labels, scores in zip(grouped_labels, grouped_scores):
        pred_scores_argsort = np.argsort(-np.array(scores))
        mrr_score = 0
        for rank, index in enumerate(pred_scores_argsort[0:k]):
            if labels[index]:
                mrr_score = 1 / (rank + 1)

        all_mrr_scores.append(mrr_score)

    mean_mrr = np.mean(all_mrr_scores)
    return mean_mrr


def create_batches(pairs, bs):
    all_questions = []
    all_answers = []
    for i in range(0, len(pairs), bs):
        questions, answers = list(zip(*pairs[i:i + bs]))
        all_questions.append(questions)
        all_answers.append(answers)
    batched_pairs = list(zip(all_questions, all_answers))
    return batched_pairs


def get_loader(task):

    if task == 'CDPCitiesQA':
        folder = './CDP/Cities/Cities Responses/'
    elif task == 'CDPStatesQA':
        folder = './CDP/States/'
    elif task == 'CDPCorpsQA':
        folder = './CDP/Corporations/Corporations Responses/Climate Change/'

    data_class = CDPQA(folder)
    dataset = data_class.load_dataset()
    print(data_class.class_weights)
    if 'train' in dataset:
        train_df, val_df, test_df = dataset['train'].to_pandas(),\
                                    dataset['val'].to_pandas(),\
                                    dataset['test'].to_pandas()
    else:
        train_df, val_df, test_df = dataset[0].to_pandas(), dataset[1].to_pandas(), dataset[2].to_pandas()

    return train_df, val_df, test_df, data_class


model = AutoModelForSequenceClassification.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

_, _, test_df, data_class = get_loader(args.task)

pairs = test_df[test_df['label'] == 1].groupby('answer').agg(list)[['question']].reset_index()
sampled_pairs = pairs.sample(frac=1.0, random_state=42)


all_questions = set(itertools.chain(*pairs['question'].values))


exploded_pairs = []
labels = []
for idx, (ans, ques) in tqdm(sampled_pairs.iterrows()):
    questions_copy = copy.deepcopy(all_questions)
    i = 1
    for q in set(ques):
        exploded_pairs.append((q, ans))
        labels.append(1)
        questions_copy.remove(q)
    for q in questions_copy:
        exploded_pairs.append((q, ans))
        labels.append(0)
    assert len(labels) % len(all_questions) == 0

batched_pairs = create_batches(exploded_pairs, args.bs)
model.cuda()
model.eval()
with torch.no_grad():
    scores = []
    for questions, answers in tqdm(batched_pairs):
        features = tokenizer(questions, answers, padding=True, truncation=True,
                             return_tensors="pt", max_length=args.max_len,
                             return_token_type_ids=True)
        features['input_ids'] = features['input_ids'].cuda()
        features['attention_mask'] = features['attention_mask'].cuda()
        features['token_type_ids'] = features['token_type_ids'].cuda()
        scores.append(model(**features).logits.cpu().numpy())

scores = np.vstack(scores).reshape(len(labels))
grouped_labels = []
grouped_scores = []
for i in range(0, len(labels), len(all_questions)):
    grouped_labels.append(labels[i:i+len(all_questions)])
    grouped_scores.append(scores[i:i+len(all_questions)])

results = {}
for k in (10, 100, 200, len(all_questions)):
    results[f'NDCG@{k}'] = ndcg_score(grouped_labels, grouped_scores, k=k)
    results[f'MRR@{k}'] = mrr_at_k(grouped_labels, grouped_scores, k)

print(results)
with open(f'test_results/qa_{args.task}_{args.model.replace("/","_")}.pickle', 'wb') as f:
    pickle.dump(results, f)
