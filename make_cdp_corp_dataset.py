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


years = ["2018", "2019", "2020"]
columns = [
    "survey_year",
    "organization",
    "module_name",
    "column_name",
    "question_unique_reference",
    "response_value",
    "comments",
]
data_folder = "./CDP/Corporations/Corporations Responses/Climate Change/"
comments = pd.DataFrame()
responses = pd.DataFrame()

for y in years:
    df = pd.read_csv(f"{data_folder}{y}_Full_Climate_Change_Dataset.csv")
    comment_df = df[~df["comments"].isna()]
    response_df = df[df["comments"].isna()]
    comments = pd.concat([comments, comment_df])
    responses = pd.concat([responses, response_df])
comments = comments.drop_duplicates(subset=["comments"], keep="first")
comments["Lang"] = comments["comments"].progress_apply(lang)
comments = comments[comments["Lang"] == "en"]
comments = comments[
    comments["comments"].fillna("").apply(lambda x: len(x.split()) > 10)
]
comments = comments[columns]
comments.to_csv(f"{data_folder}filtered_city_comments.csv")


responses = responses[
    responses["response_value"].fillna("").apply(lambda x: len(x.split()) > 10)
]
responses = responses.drop_duplicates(subset=["response_value"], keep="first")
responses["Lang"] = responses["response_value"].progress_apply(lang)
responses = responses[responses["Lang"] == "en"]
responses = responses[columns]
responses.to_csv(f"{data_folder}filtered_city_responses.csv")

combined = pd.concat([responses, comments])
combined["Text"] = combined["comments"].where(
    ~combined["comments"].isna(), combined["response_value"]
)
combined["module_name"] = combined["module_name"].apply(clean)
combined["column_name"] = combined["column_name"].apply(clean)
combined = combined.reset_index(drop=True).reset_index().rename(columns={"index": "id"})
combined.to_csv(f"{data_folder}combined.csv", index=False)
