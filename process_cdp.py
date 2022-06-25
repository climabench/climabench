import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

seed = 10
np.random.seed(seed)
data_folder = "./CDP/Cities/Cities Responses/"
parent_to_label = {
    "Emissions Reduction: City-wide": "Emissions",
    "Strategy": "Strategy",
    "Hazards and Adaptation": {
        "Climate Hazards": "Climate Hazards",
        "Adaptation": "Adaptation",
        "Social Risks": "Climate Hazards",
    },
    "Climate Hazards": "Climate Hazards",
    "Governance and Data Management": "Governance and Data Management",
    "Opportunities": "Opportunities",
    "Adaptation": "Adaptation",
    "City Wide Emissions": "Emissions",
    "Water": "Water",
    "Emissions Reduction: Local Government": "Emissions",
    "Local Government Emissions": "Emissions",
    "Local Government Operations GHG Emissions Data": "Emissions",
    "Introduction": None,
    "GHG Emissions Data": "Emissions",
    "Climate Hazards & Vulnerability": "Climate Hazards",
    "Water Security": "Water",
    "Emissions Reduction": "Emissions",
    "Climate Hazards and Vulnerability": "Climate Hazards",
    "City-wide Emissions": "Emissions",
    "Energy Data": "Energy",
}
section_to_label = {
    "Introduction": None,
    "Opportunities": "Opportunities",
    "Energy": "Energy",
    "Waste": "Waste",
    "Transport": "Transport",
    "Food": "Food",
    "Buildings": "Buildings",
    "Submit your response": None,
    "Urban Planning": "Strategy",
}


def filter_org(row, orgs):
    return row["Organization"] in orgs


def create_label(x):
    parent = x["Parent Section"]
    section = x["Section"]
    if type(parent) is str:
        if type(parent_to_label[parent]) is dict:
            return parent_to_label[parent][section]
        return parent_to_label[parent]
    return section_to_label[section]


combined = pd.read_csv(f"{data_folder}combined.csv")
print(combined.columns)
combined["Label"] = combined.apply(create_label, axis=1)
combined = combined[~combined["Label"].isna()]
print(combined.shape)
print(combined['Label'].value_counts())
orgs = list(combined["Organization"].value_counts()[:420].keys())
np.random.shuffle(orgs)
print(orgs[:10])

train = combined[combined.apply(filter_org, args=(orgs[:280],), axis=1)]
val = combined[combined.apply(filter_org, args=(orgs[280:350],), axis=1)]
test = combined[combined.apply(filter_org, args=(orgs[350:],), axis=1)]
print(train.shape, val.shape, test.shape)
print(set(train['id']).intersection(set(val['id'])))
print(set(train['id']).intersection(set(test['id'])))
other = combined[combined.apply(lambda x: x['id'] not in train['id'], axis=1)]
other = other[other.apply(lambda x: x['id'] not in val['id'], axis=1)]
other = other[other.apply(lambda x: x['id'] not in test['id'], axis=1)]
print(other.shape)
train = pd.concat([train, other])
print("-------------------------------------------")
print(f'Train:\n{train["Label"].value_counts()}')
print(f'Train total:\n{sum(train["Label"].value_counts())}')
print("-------------------------------------------")
print(f'Val:\n{val["Label"].value_counts()}')
print(f'Val total:\n{sum(val["Label"].value_counts())}')
print("-------------------------------------------")
print(f'Test:\n{test["Label"].value_counts()}')
print(f'Test total:\n{sum(test["Label"].value_counts())}')
print("-------------------------------------------")
train.to_csv(f"{data_folder}train.csv", index=False)
val.to_csv(f"{data_folder}val.csv", index=False)
test.to_csv(f"{data_folder}test.csv", index=False)
