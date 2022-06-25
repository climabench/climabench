import pandas as pd
import numpy as np
from datasets import Dataset, Features, Value, load_dataset
from transformers import PreTrainedTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, List


class CGDataset:
    @staticmethod
    def _encode(dataset, tokenizer, max_length):
        def tokenize(batch):
            return tokenizer(
                batch["text"], padding="max_length", truncation=True, max_length=max_length,
            )

        dataset = dataset.map(tokenize, batched=True, batch_size=len(dataset))
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        return dataset


class SciDCC(CGDataset):
    def __init__(self, file: str):
        super().__init__()
        self.file = file
        self.num_labels = 20
        self.labels = None
        self.class_weights = None

    def prepare(self, tokenizer: PreTrainedTokenizer, max_length=512):
        train_dataset, val_dataset, test_dataset = self.load_dataset()
        return (
            self._encode(train_dataset, tokenizer, max_length),
            self._encode(val_dataset, tokenizer, max_length),
            self._encode(test_dataset, tokenizer, max_length),
        )

    @staticmethod
    def _convert_to_dataset(X, y):
        df = pd.DataFrame([X, y]).T
        df.columns = ["text", "label"]
        dataset = Dataset.from_pandas(df)
        return dataset

    def load_dataset(self):
        scidcc = pd.read_csv(self.file)
        scidcc["text"] = (
            scidcc["Title"] + " " + scidcc["Summary"] + " " + scidcc["Body"].fillna("")
        )
        i = iter(range(self.num_labels))
        self.labels = {k: next(i) for k in scidcc["Category"].value_counts().keys()}
        scidcc_y = scidcc["Category"].map(self.labels)

        X_train, X_test, y_train, y_test = train_test_split(
            scidcc["text"].values,
            scidcc_y,
            test_size=0.1,
            random_state=42,
            stratify=scidcc["Category"],
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.111, random_state=42, stratify=y_train
        )
        train_dataset = self._convert_to_dataset(X_train, y_train)
        val_dataset = self._convert_to_dataset(X_val, y_val)
        test_dataset = self._convert_to_dataset(X_test, y_test)

        weight = sum(y_train.value_counts()) / y_train.value_counts()
        weight = weight / sum(weight)
        self.class_weights = weight.values
        return train_dataset, val_dataset, test_dataset


class ClimaText(CGDataset):
    def __init__(self, files: Dict[str, List[str]]):
        super().__init__()
        assert "train" in files
        assert "val" in files
        assert "test" in files
        self.files = files
        self.num_labels = 2
        self.class_weights = None
        self.labels = {'Negative':0, 'Positive':1}

    def prepare(self, tokenizer, max_length=512):
        dataset = self.load_dataset()
        dataset = self._encode(dataset, tokenizer, max_length)
        return dataset["train"], dataset["val"], dataset["test"]

    def load_dataset(self):
        dataset = load_dataset(
            "csv", data_files=self.files, delimiter="\t"
        ).remove_columns(["id", "title", "paragraph"]).rename_column('sentence', 'text')
        label_counts = dataset['train'].to_pandas()['label'].value_counts()
        label_counts = label_counts[sorted(label_counts.keys())]
        weight = sum(label_counts) / label_counts
        weight = weight / sum(weight)
        self.class_weights = weight.values
        return dataset


class CDPCities(CGDataset):
    def __init__(self, folder: str):
        super().__init__()
        self.folder = folder
        self.num_labels = 12
        self.labels = None
        self.class_weights = None

    def prepare(self, tokenizer: PreTrainedTokenizer, max_length=512):
        train_dataset, val_dataset, test_dataset = self.load_dataset()
        return (
            self._encode(train_dataset, tokenizer, max_length),
            self._encode(val_dataset, tokenizer, max_length),
            self._encode(test_dataset, tokenizer, max_length),
        )

    @staticmethod
    def _convert_to_dataset(X, y):
        df = pd.concat([X[["id", "Text"]].reset_index(drop=True), pd.Series(y)], axis=1)
        df.columns = ["id", "text", "label"]
        print(df.head())
        dataset = Dataset.from_pandas(df)
        return dataset

    def load_dataset(self):
        train = pd.read_csv(f'{self.folder}/train.csv')
        val = pd.read_csv(f'{self.folder}/val.csv')
        test = pd.read_csv(f'{self.folder}/test.csv')
        i = iter(range(self.num_labels))
        self.labels = {k: next(i) for k in train['Label'].value_counts().keys()}

        X_train = train[["id", "Organization", "Text"]]
        y_train = train["Label"].map(self.labels)
        X_val = val[["id", "Organization", "Text"]]
        y_val = val["Label"].map(self.labels)
        X_test = test[["id", "Organization", "Text"]]
        y_test = test["Label"].map(self.labels)
        print(X_train.shape, X_val.shape, X_test.shape)

        weight = sum(y_train.value_counts()) / y_train.value_counts()
        weight = weight / sum(weight)
        self.class_weights = weight.values

        train_dataset = self._convert_to_dataset(X_train, y_train)
        val_dataset = self._convert_to_dataset(X_val, y_val)
        test_dataset = self._convert_to_dataset(X_test, y_test)
        return train_dataset, val_dataset, test_dataset


class CDPQA(CGDataset):
    def __init__(self, folder: str):
        super().__init__()
        self.folder = folder
        self.num_labels = 1
        self.labels = None
        self.class_weights = None

    def prepare(self, tokenizer: PreTrainedTokenizer, max_length=512):
        train_dataset, val_dataset, test_dataset = self.load_dataset()
        return (
            self._encode(train_dataset, tokenizer, max_length),
            self._encode(val_dataset, tokenizer, max_length),
            self._encode(test_dataset, tokenizer, max_length),
        )

    @staticmethod
    def _convert_to_dataset(df):
        return Dataset.from_pandas(df)

    @staticmethod
    def _encode(dataset, tokenizer, max_length=512):
        def tokenize(batch):
            return tokenizer(
                batch["question"], batch["answer"], padding=True, truncation=True, max_length=max_length,
                return_token_type_ids=True
            )

        dataset = dataset.map(tokenize, batched=True, batch_size=len(dataset))
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
        return dataset

    def load_dataset(self):
        train = pd.read_csv(f'{self.folder}/train_qa.csv')
        val = pd.read_csv(f'{self.folder}/val_qa.csv')
        test = pd.read_csv(f'{self.folder}/test_qa.csv')
        i = iter(range(2))
        self.labels = {k: next(i) for k in train['label'].value_counts().keys()}
        y_train = train['label']
        print(train.shape, val.shape, test.shape)
        weight = sum(y_train.value_counts()) / y_train.value_counts()
        weight = weight / sum(weight)
        self.class_weights = weight.values[1]
        if val.shape[0] > 50000:
            val = val.sample(50000, random_state=42)
        train_dataset = self._convert_to_dataset(train)
        val_dataset = self._convert_to_dataset(val)
        test_dataset = self._convert_to_dataset(test)
        return train_dataset, val_dataset, test_dataset


class ClimateStance(CGDataset):
    def __init__(self, files: Dict[str, List[str]]):
        super().__init__()
        assert "train" in files
        assert "val" in files
        assert "test" in files
        self.files = files
        self.num_labels = 3
        self.class_weights = None
        self.labels = None

    def prepare(self, tokenizer, max_length=512):
        dataset = self.load_dataset()
        dataset = self._encode(dataset, tokenizer, max_length)
        return dataset["train"], dataset["val"], dataset["test"]

    def load_dataset(self):
        dataset = load_dataset(
            "csv", data_files=self.files,
        )
        label_counts = dataset['train'].to_pandas()['label'].value_counts()
        label_counts = label_counts[sorted(label_counts.keys())]
        weight = sum(label_counts) / label_counts
        weight = weight / sum(weight)
        self.class_weights = weight.values
        return dataset


class ClimateEng(CGDataset):
    def __init__(self, files: Dict[str, List[str]]):
        super().__init__()
        assert "train" in files
        assert "val" in files
        assert "test" in files
        self.files = files
        self.num_labels = 5
        self.class_weights = None
        self.labels = None

    def prepare(self, tokenizer, max_length=512):
        dataset = self.load_dataset()
        dataset = self._encode(dataset, tokenizer, max_length)
        return dataset["train"], dataset["val"], dataset["test"]

    def load_dataset(self):
        dataset = load_dataset(
            "csv", data_files=self.files,
        )
        label_counts = dataset['train'].to_pandas()['label'].value_counts()
        label_counts = label_counts[sorted(label_counts.keys())]
        weight = sum(label_counts) / label_counts
        weight = weight / sum(weight)
        self.class_weights = weight.values
        return dataset


class ClimateInsurance(CGDataset):
    def __init__(self, files: Dict[str, List[str]]):
        super().__init__()
        assert "train" in files
        assert "val" in files
        assert "test" in files
        self.files = files
        self.num_labels = 2
        self.class_weights = None
        self.labels = {'N': 0, 'Y': 1}

    def prepare(self, tokenizer, max_length=512):
        dataset = self.load_dataset()
        dataset = self._encode(dataset, tokenizer, max_length)
        return dataset["train"], dataset["val"], dataset["test"]

    def load_dataset(self):
        dataset = load_dataset(
            "csv", data_files=self.files,
        )
        label_counts = dataset['train'].to_pandas()['label'].value_counts()
        label_counts = label_counts[sorted(label_counts.keys())]
        weight = sum(label_counts) / label_counts
        weight = weight / sum(weight)
        self.class_weights = weight.values
        return dataset


class ClimateInsuranceMulti(CGDataset):
    def __init__(self, files: Dict[str, List[str]]):
        super().__init__()
        assert "train" in files
        assert "val" in files
        assert "test" in files
        self.files = files
        self.num_labels = 8
        self.class_weights = None
        self.labels = None

    def prepare(self, tokenizer, max_length=512):
        dataset = self.load_dataset()
        dataset = self._encode(dataset, tokenizer, max_length)
        return dataset["train"], dataset["val"], dataset["test"]

    def load_dataset(self):
        dataset = load_dataset(
            "csv", data_files=self.files,
        )
        label_counts = dataset['train'].to_pandas()['label'].value_counts()
        label_counts = label_counts[sorted(label_counts.keys())]
        weight = sum(label_counts) / label_counts
        weight = weight / sum(weight)
        self.class_weights = weight.values
        return dataset


class ClimateFEVER(CGDataset):
    def __init__(self):
        super().__init__()
        self.num_labels = 3
        self.labels = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT_ENOUGH_INFO': 2}
        self.class_weights = None

    def prepare(self, tokenizer: PreTrainedTokenizer, max_length=512):
        train_dataset, val_dataset, test_dataset = self.load_dataset()
        return (
            self._encode(train_dataset, tokenizer, max_length),
            self._encode(val_dataset, tokenizer, max_length),
            self._encode(test_dataset, tokenizer, max_length),
        )

    @staticmethod
    def _convert_to_dataset(df):
        df = df[['claim','evidence', 'evidence_label']]
        df = df.rename(columns={"evidence_label": "label"})
        return Dataset.from_pandas(df)

    @staticmethod
    def _encode(dataset, tokenizer, max_length=512):
        def tokenize(batch):
            return tokenizer(
                batch["claim"], batch["evidence"], padding=True, truncation=True, max_length=max_length,
                return_token_type_ids=True
            )

        dataset = dataset.map(tokenize, batched=True, batch_size=len(dataset))
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
        return dataset

    def load_dataset(self):
        ds = load_dataset('climate_fever')
        ds = ds['test'].to_pandas()
        ds = ds.explode('evidences')
        evidences = pd.json_normalize(ds['evidences']).reset_index(drop=True)
        ds = pd.concat([ds.reset_index(drop=True), evidences], axis=1)
        train, test = train_test_split(ds, test_size=0.1, random_state=42,
                                       stratify=ds['evidence_label'])
        train, val = train_test_split(train, test_size=0.111, random_state=42,
                                      stratify=train['evidence_label'])
        y_train = train['evidence_label']
        print(train.shape, val.shape, test.shape)

        weight = sum(y_train.value_counts()) / pd.Series(dict(sorted(y_train.value_counts().items())))
        weight = weight / sum(weight)
        self.class_weights = weight.values
        train_dataset = self._convert_to_dataset(train)
        val_dataset = self._convert_to_dataset(val)
        test_dataset = self._convert_to_dataset(test)
        return train_dataset, val_dataset, test_dataset

