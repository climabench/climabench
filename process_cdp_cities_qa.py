import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas()

seed = 10
np.random.seed(seed)

def filter_org(row, orgs):
    return row["Organization"] in orgs


def make_gold_set(x, df):
    filtered = df[df['Organization'] == x['Organization']]
    filtered = filtered[filtered['id'] != x['id']]
    if len(filtered) < 5:
        answers = list(filtered['Text'].sample(frac=1).values)
    else:
        answers = list(filtered['Text'].sample(5).values)
    answers.append(x['Text'])
    return answers

def process_split(split):
    split['Options'] = split.progress_apply(make_gold_set, args=(split,), axis=1)
    split = split.explode(column=['Options'])
    split['Answer'] = (split['Text'] == split['Options']).astype(int)
    split = split.sample(frac=1).reset_index(drop=True)
    split = split[['Question Name', 'Options', 'Answer']]
    split.columns = ['question', 'answer', 'label']
    return split

data_folder = "CDP/Cities/Cities Responses/"

df = pd.read_csv(data_folder + 'combined.csv')[['id', 'Organization', 'Question Name', 'Text']]

orgs = list(df["Organization"].value_counts()[:420].keys())
np.random.shuffle(orgs)
print(orgs[:10])

train = df[df.apply(filter_org, args=(orgs[:280],), axis=1)]
val = df[df.apply(filter_org, args=(orgs[280:350],), axis=1)]
test = df[df.apply(filter_org, args=(orgs[350:],), axis=1)]
print(train.shape, val.shape, test.shape)
print(set(train['id']).intersection(set(val['id'])))
print(set(train['id']).intersection(set(test['id'])))
other = df[df.apply(lambda x: x['id'] not in train['id'], axis=1)]
other = other[other.apply(lambda x: x['id'] not in val['id'], axis=1)]
other = other[other.apply(lambda x: x['id'] not in test['id'], axis=1)]
print(other.shape)
train = pd.concat([train, other])

total = train.shape[0] + val.shape[0] + test.shape[0]
print(train.shape[0]/total, val.shape[0]/total, test.shape[0]/total)

train = process_split(train)
val = process_split(val)
test = process_split(test)


print("-------------------------------------------")
print(f'Train:\n{train["label"].value_counts()}')
print(f'Train total:\n{sum(train["label"].value_counts())}')
print("-------------------------------------------")
print(f'Val:\n{val["label"].value_counts()}')
print(f'Val total:\n{sum(val["label"].value_counts())}')
print("-------------------------------------------")
print(f'Test:\n{test["label"].value_counts()}')
print(f'Test total:\n{sum(test["label"].value_counts())}')
print("-------------------------------------------")

train.to_csv(f"{data_folder}train_qa.csv", index=False)
val.to_csv(f"{data_folder}val_qa.csv", index=False)
test.to_csv(f"{data_folder}test_qa.csv", index=False)
