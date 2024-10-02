import re
import ray
from ray.data import Dataset
import numpy as np
from transformers import BertModel,BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple

from config import STOPWORDS

def load_data(dataset_loc: str, num_samples: int = None) -> Dataset:
    """Load data from source into a Ray Dataset.

    Args:
        dataset_loc (str): Location of the dataset.
        num_samples (int, optional): The number of samples to load. Defaults to None.

    Returns:
        Dataset: Our dataset represented by a Ray Dataset.
    """
    ds = ray.data.read_csv(dataset_loc)
    ds = ds.random_shuffle(seed=1234)
    ds = ray.data.from_items(ds.take(num_samples)) if num_samples else ds
    return ds

def stratify_split(
    ds: Dataset,
    stratify: str,
    test_size: float,
    shuffle: bool = True,
    seed: int = 1234,
) -> Tuple[Dataset, Dataset]:
    def _add_split(df):
        train,test=train_test_split(df,test_size=test_size,shuffle=True,random_state=1234)
        train["_split"]="train"
        test["_split"]="test"
        return pd.concat([train,test])

    def _get_split(df,split):
        return df[df['_split']==split].drop("_split",axis=1)

    #Distributed
    grouped=ds.groupby(stratify).map_groups(_add_split,batch_format="pandas")
    train_ds=grouped.map_batches(_get_split,fn_kwargs={"split":"train"},batch_format="pandas")
    test_ds=grouped.map_batches(_get_split,fn_kwargs={"split":"test"},batch_format="pandas")

    # Shuffle each split (required)
    train_ds = train_ds.random_shuffle(seed=seed)
    test_ds = test_ds.random_shuffle(seed=seed)

    return train_ds,test_ds

def clean_text(text, stopwords=STOPWORDS):
    """Clean raw text string."""
    # Lower
    text = text.lower()

    # Remove stopwords
    pattern = re.compile(r'\b(' + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub('', text)

    # Spacing and filters
    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)  # add spacing
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends
    text = re.sub(r"http\S+", "", text)  #  remove links

    return text


def tokenize(batch):
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    encoded_inputs = tokenizer(batch["text"].tolist(), return_tensors="np", padding="longest")
    return dict(ids=encoded_inputs["input_ids"], masks=encoded_inputs["attention_mask"], targets=np.array(batch["tag"]))


def preprocess(df, class_to_index):
    """Preprocess the data."""
    df["text"] = df.title + " " + df.description  # feature engineering
    df["text"] = df.text.apply(clean_text)  # clean text
    df = df.drop(columns=["id", "created_on", "title", "description"], errors="ignore")  # clean dataframe
    df = df[["text", "tag"]]  # rearrange columns
    df["tag"] = df["tag"].map(class_to_index)  # label encoding
    outputs = tokenize(df)
    return outputs


class CustomPreprocessor:
    def __init__(self,class_to_index={}):
        self.class_to_index=class_to_index or {}
        self.index_to_class={v:k for k,v in self.class_to_index.items()}


    def fit(self, ds: Dataset):
        if not self.class_to_index:
            # ds_arrow = ds.map_batches(lambda df: df, batch_format="pyarrow")
            # unique_tags=ds_arrow.unique(column="tag")
            tags = ds.unique(column="tag")
            self.class_to_index={k:v for k,v in zip(tags,range(len(tags)))}
            self.index_to_class={v:k for k,v in self.class_to_index.items()}
        return self
    
    def transform(self, ds: Dataset):
        return ds.map_batches(preprocess,fn_kwargs={"class_to_index":self.class_to_index},batch_format="pandas")