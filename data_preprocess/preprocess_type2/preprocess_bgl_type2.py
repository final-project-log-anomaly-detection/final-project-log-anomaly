import pandas as pd
import numpy as np

from gensim.models import Word2Vec, KeyedVectors
from nltk.tokenize import RegexpTokenizer

import os
import sys

if __name__ == '__main__':
    sys.path.append(os.path.abspath('data_preprocess'))

import pickle
import re
from utils import json_pretty_dump, word2VecContinueLearning, trainWord2VecModelType2, tokenizeData, convertWord2Vec, text_cleansing

seed = 42
np.random.seed(seed)


data_dir = "./data_preprocess/processed_type2/BGL_preprocessed_type2"

params = {
    "struct_file": "./Drain_result/BGL.log_structured.csv",
    "template_file": "./Drain_result/BGL.log_templates.csv",
}

os.makedirs(data_dir, exist_ok=True)


def preprocess_bgl(struct_file, template_file, **kwargs):
    struct_log = pd.read_csv(struct_file)
    template_log = pd.read_csv(template_file)

    struct_log["Label"] = struct_log["Label"].apply(lambda x: int(x != "-"))
    struct_log.sort_values("Time", inplace=True)

    struct_log[struct_log["Label"] == 1].Date.value_counts().sort_index()
    split_date = struct_log[struct_log.Label == 1].Date.values[0]

    train_set = struct_log[struct_log.Date < split_date]
    test_set = struct_log[struct_log.Date >= split_date]
    eventId_train = train_set.EventId.unique()
    eventId_test = test_set.EventId.unique()

    template_log_train = template_log[template_log["EventId"].isin(eventId_train)]
    template_log_test = template_log[template_log["EventId"].isin(eventId_test)]
    template_log_train["EventTemplateIdent_cleansed"] = template_log_train.EventTemplateIdent.map(text_cleansing)

    template_log_train_list = template_log_train["EventTemplateIdent_cleansed"].astype('str').tolist()

    tokenizer = RegexpTokenizer(r'[A-Z][a-z]+|\w+')
    token_train_list = [tokenizer.tokenize(sen) for sen in template_log_train_list]

    template_log_train["EventTemplateIdent_token"] = pd.Series(token_train_list)
    train_set["Token"] = train_set.EventId.map(
        lambda id: template_log_train[template_log_train.EventId == id].
        EventTemplateIdent_token.values[0]
    )

    trainWord2VecModelType2(token_train_list)

    print('Total rows: ', len(train_set) + len(test_set))

    print("# train sessions: {} ({:.2f}%)".format(len(train_set), 100*sum(train_set['Label'])/len(train_set)))
    print("# test sessions: {} ({:.2f}%)".format(len(test_set), 100*sum(test_set['Label'])/len(test_set)))

    with open(os.path.join(data_dir, "train_set.pkl"), "wb") as fw:
        pickle.dump(train_set, fw)
    with open(os.path.join(data_dir, "test_set.pkl"), "wb") as fw:
        pickle.dump(test_set, fw)
    with open(os.path.join(data_dir, "template_train_set.pkl"), "wb") as fw:
        pickle.dump(template_log_train, fw)
    with open(os.path.join(data_dir, "template_test_set.pkl"), "wb") as fw:
        pickle.dump(template_log_test, fw)

    print("finish to save .pkl file.")


if __name__ == '__main__':
    preprocess_bgl(**params)
