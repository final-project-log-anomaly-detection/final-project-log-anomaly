import numpy as np
import argparse
import pandas as pd

from gensim.models.word2vec import Word2Vec
import os
import pickle
from utils import json_pretty_dump, word2VecContinueLearning, trainWord2VecModel, tokenizeData, convertWord2Vec
from sklearn.model_selection import train_test_split

seed = 42
np.random.seed(seed)

parser = argparse.ArgumentParser()

parser.add_argument("--train_anomaly_ratio", default=0.0, type=float)

params = vars(parser.parse_args())

data_name = f'thunderbird_{params["train_anomaly_ratio"]}_tar'
data_dir = "data_preprocess/processed/thunderbird_preprocessed/"

params = {
    "log_file":  "/Users/thanadonlamsan/Documents/research project จบ/final_project_code/final-project-log-anomaly/Drain_result/thunderbird_small.log_structured.csv",
    "test_ratio": 0.2,
    # "random_sessions": True,  # shuffle sessions
    "train_anomaly_ratio": params["train_anomaly_ratio"],
    "train_word2Vec": True
}

data_dir = os.path.join(data_dir, data_name)
os.makedirs(data_dir, exist_ok=True)


def preprocess_thunderbird(log_file, test_ratio=None, train_anomaly_ratio=1, train_word2Vec=False, **kwargs):
    print("Loading ThunderBird logs from {}.".format(log_file))

    struct_log = pd.read_csv(log_file, engine="c", na_filter=False, memory_map=True)
    struct_log['Label'] = struct_log['Label'].map(lambda x: int(x != "-"))
    eventTemplateTokenTrain = []
    eventTemplateTokenTest = []
    eventVectors = []

    train_data, test_data = train_test_split(struct_log, test_size=test_ratio, random_state=42)
    train_data = train_data.loc[train_data['Label'] == 0]
    print(train_data)
    train_data = train_data.sort_values(by=['Date', 'Time'])
    test_data = test_data.sort_values(by=['Date', 'Time'])

    eventTemplateTokenTrain = tokenizeData(train_data)
    eventTemplateTokenTest = tokenizeData(test_data)

    if train_word2Vec:
        trainWord2VecModel(eventTemplateTokenTrain, 'word2vec_thunderbird')

    model = Word2Vec.load("word2vec_thunderbird.model")
    eventVectors = convertWord2Vec(eventTemplateTokenTrain, model)
    train_data['EventTemplateIdentToken'] = eventTemplateTokenTrain
    train_data['Vectors'] = eventVectors

    eventVectors = convertWord2Vec(eventTemplateTokenTest, model)
    test_data['EventTemplateIdentToken'] = eventTemplateTokenTest
    test_data['Vectors'] = eventVectors

    print("# train sessions: {} ({:.2f}%)".format(len(train_data), 100*sum(train_data['Label'])/len(train_data)))
    print("# test sessions: {} ({:.2f}%)".format(len(test_data), 100*sum(test_data['Label'])/len(test_data)))

    with open(os.path.join(data_dir, "session_train.pkl"), "wb") as fw:
        pickle.dump(train_data, fw)
    with open(os.path.join(data_dir, "session_test.pkl"), "wb") as fw:
        pickle.dump(test_data, fw)
    json_pretty_dump(params, os.path.join(data_dir, "data_desc.json"))


if __name__ == '__main__':
    preprocess_thunderbird(**params)
