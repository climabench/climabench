import pandas as pd
from langdetect import detect
from tqdm import tqdm
import re

tqdm.pandas()


def lang(x):
    try:
        return detect(x)
    except:
        print(x)
        return "na"


def clean(x):
    try:
        return re.sub(r'\d+. ', '', x)
    except:
        return x


years = ["2018", "2019", "2020", "2021"]
columns = [
    "Year Reported to CDP",
    "Organization",
    "Parent Section",
    "Section",
    "Question Name",
    "Row Name",
    "Response Answer",
    "Comments",
]
data_folder = "./CDP/Cities/Cities Responses/"
comments = pd.DataFrame()
responses = pd.DataFrame()

for y in years:
    df = pd.read_csv(f"{data_folder}{y}_Full_Cities_Dataset.csv")
    comment_df = df[~df["Comments"].isna()]
    response_df = df[df["Comments"].isna()]
    comments = pd.concat([comments, comment_df])
    responses = pd.concat([responses, response_df])
comments = comments.drop_duplicates(subset=["Comments"], keep="first")
comments["Lang"] = comments["Comments"].progress_apply(lang)
comments = comments[comments["Lang"] == "en"]
comments = comments[
    comments["Comments"].fillna("").apply(lambda x: len(x.split()) > 10)
]
comments = comments[columns]
comments.to_csv(f"{data_folder}filtered_city_comments.csv")


responses = responses[
    responses["Response Answer"].fillna("").apply(lambda x: len(x.split()) > 10)
]
responses = responses.drop_duplicates(subset=["Response Answer"], keep="first")
responses["Lang"] = responses["Response Answer"].progress_apply(lang)
responses = responses[responses["Lang"] == "en"]
responses = responses[columns]
responses.to_csv(f"{data_folder}filtered_city_responses.csv")

combined = pd.concat([responses, comments])
combined["Text"] = combined["Comments"].where(
    ~combined["Comments"].isna(), combined["Response Answer"]
)
combined["Parent Section"] = combined["Parent Section"].apply(clean)
combined["Section"] = combined["Section"].apply(clean)
combined = combined.reset_index(drop=True).reset_index().rename(columns={"index": "id"})
combined.to_csv(f"{data_folder}combined.csv", index=False)
