import re
import numpy as np
import pandas as pd
import warnings
import nltk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import classification_report, f1_score
from datasets import Dataset, load_dataset
from dataloaders import CDPQA
from tqdm import tqdm
import itertools
import copy
from sklearn.metrics import ndcg_score
import argparse
import pickle
from rank_bm25 import BM25Okapi

parser = argparse.ArgumentParser()

parser.add_argument("--task", default='CDPCitiesQA',
                    help="Tasks like CDPCitiesQA, CDPStatesQA")
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


_, _, test_df, data_class = get_loader(args.task)
test_df = test_df[test_df['label'] == 1]
print(test_df.shape)
test_df = test_df.drop_duplicates(subset='answer')
print(test_df.shape)
pairs = test_df[test_df['label'] == 1].groupby('answer').agg(list)[['question']].reset_index()
sampled_pairs = pairs.sample(frac=1.0, random_state=42)

print(pairs.shape)
print(sampled_pairs.shape)
all_questions = list(set(itertools.chain(*pairs['question'].values)))

tokenized_corpus = [answer.split(" ") for answer in sampled_pairs['answer'].values]

bm25 = BM25Okapi(tokenized_corpus)

exploded_pairs = []
labels = []
grouped_labels = []
grouped_scores = []

for idx, (ans, gold_ques) in tqdm(sampled_pairs.iterrows()):
    if len(gold_ques) != 1:
        print(gold_ques)
        print(ans)
        raise Exception
    gold_idx = all_questions.index(gold_ques[0])
    assert gold_idx >= 0
    label_arr = []
    for index in range(0, len(all_questions)):
        if index == gold_idx:
            label_arr.append(1)
        else:
            label_arr.append(0)
    grouped_labels.append(label_arr)

for ques in tqdm(all_questions):
    score_arr = bm25.get_scores(ques.split(" "))
    grouped_scores.append(score_arr)

grouped_scores = np.vstack(grouped_scores).T
print(grouped_scores.shape)
results = {}
for k in (10, 100, 200, len(all_questions)):
    results[f'NDCG@{k}'] = ndcg_score(grouped_labels, grouped_scores, k=k)
    results[f'MRR@{k}'] = mrr_at_k(grouped_labels, grouped_scores, k)

print(results)
with open(f'test_results/qa_{args.task}_bm25.pickle', 'wb') as f:
    pickle.dump(results, f)
