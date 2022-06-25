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


model = AutoModelForSequenceClassification.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

_, _, test_df, data_class = get_loader(args.task)

pairs = test_df[test_df['label'] == 1].groupby('answer').agg(list)[['question']].reset_index()
sampled_pairs = pairs.sample(frac=1.0, random_state=42)


all_questions = list(set(itertools.chain(*pairs['question'].values)))

model.cuda()
model.eval()

extracts = pd.read_csv('/home/tlaud/pdf_extracts.txt', sep='|')
results = []
with torch.no_grad():
    for answer in tqdm(extracts['Text']):
        features = tokenizer(all_questions, np.repeat(answer, len(all_questions)).tolist(), padding=True, truncation=True,
                             return_tensors="pt", max_length=args.max_len,
                             return_token_type_ids=True)
        features['input_ids'] = features['input_ids'].cuda()
        features['attention_mask'] = features['attention_mask'].cuda()
        features['token_type_ids'] = features['token_type_ids'].cuda()
        outputs = model(**features).logits.cpu().numpy()
        outputs = outputs.flatten()
        #print(outputs.shape)
        ind = np.argpartition(outputs, -5)[-5:]
        print(ind)
        print(np.sort(-outputs)[:5])
        top5_idx = ind[np.argsort(outputs[ind])]
        print(top5_idx)
        top5_questions = []
        for idx in top5_idx:
            top5_questions.append(all_questions[idx])
        results.append(top5_questions)

with open(f'test_results/topq_{args.task}_{args.model.replace("/","_")}.pickle', 'wb') as f:
    pickle.dump(results, f)

