import numpy as np
import argparse
import pandas as pd

from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import os
import pickle
from utils import json_pretty_dump
from sklearn.model_selection import train_test_split


seed = 42
np.random.seed(seed)

parser = argparse.ArgumentParser()

parser.add_argument("--train_anomaly_ratio", default=0.0, type=float)

params = vars(parser.parse_args())

data_name = f'bgl_{params["train_anomaly_ratio"]}_tar'
data_dir = "data_preprocess/processed/BGL_preprocessed/"

params = {
    "log_file":  "/Users/thanadonlamsan/Documents/research project จบ/final_project_code/final-project-log-anomaly/Drain_result/BGL_2.log_structured.csv",
    "test_ratio": 0.2,
    # "random_sessions": True,  # shuffle sessions
    "train_anomaly_ratio": params["train_anomaly_ratio"],
    "train_word2Vec": False
}

data_dir = os.path.join(data_dir, data_name)
os.makedirs(data_dir, exist_ok=True)


def preprocess_bgl(log_file, test_ratio=None, train_anomaly_ratio=1, train_word2Vec=False, **kwargs):
    print("Loading BGL logs from {}.".format(log_file))

    struct_log = pd.read_csv(log_file, engine="c", na_filter=False, memory_map=True)
    struct_log['Label'] = struct_log['Label'].map(lambda x: int(x != "-"))
    eventTemplateToken = []
    eventVectors = []

    for _, row in struct_log.iterrows():
        eventTemplateToken.append(str(row['EventTemplateIdent']).split())

    if train_word2Vec:
        trainWord2VecModel(eventTemplateToken)

    model = Word2Vec.load("word2vec_bgl.model")
    for token_list in eventTemplateToken:
        list_vector = []
        for word in token_list:
            # print(model.wv[word])
            list_vector.append(model.wv[word])
        eventVectors.append(list_vector)

    struct_log['EventTemplateIdentToken'] = eventTemplateToken
    struct_log['Vectors'] = eventVectors
    print('Total rows: ', len(struct_log))

    train_data, test_data = train_test_split(struct_log, test_size=test_ratio, random_state=42)
    train_data = train_data.loc[train_data['Label'] == 0]

    print("# train sessions: {} ({:.2f}%)".format(len(train_data), 100*sum(train_data['Label'])/len(train_data)))
    print("# test sessions: {} ({:.2f}%)".format(len(test_data), 100*sum(test_data['Label'])/len(test_data)))

    with open(os.path.join(data_dir, "session_train.pkl"), "wb") as fw:
        pickle.dump(train_data, fw)
    with open(os.path.join(data_dir, "session_test.pkl"), "wb") as fw:
        pickle.dump(test_data, fw)
    json_pretty_dump(params, os.path.join(data_dir, "data_desc.json"))


def trainWord2VecModel(eventTemplateToken):
    print("start train word2Vec model. . . . .")
    model = Word2Vec(vector_size=300, window=5, min_count=1, workers=4)
    model.build_vocab(eventTemplateToken)
    training_examples_count = model.corpus_count
    preTrained_model = KeyedVectors.load_word2vec_format(
        "./google_news/GoogleNews-vectors-negative300.bin",
        binary=True
    )
    model.build_vocab([list(preTrained_model.key_to_index.keys())], update=True)
    model.wv.vectors_lockf = np.ones(len(model.wv), dtype=np.float32)
    model.wv.intersect_word2vec_format("./google_news/GoogleNews-vectors-negative300.bin", binary=True, lockf=1.0)
    model.train(eventTemplateToken, total_examples=training_examples_count, epochs=5)
    model.save('word2vec_bgl.model')
    print('finish train word2Vec model . . . . . ^^')


def convertGloveWord2Vec():
    # _ = glove2word2vec("./glove/glove.6B.200d.txt", "./glove_word2vec/glove.6B.200d_word2vec.txt")
    pass


if __name__ == '__main__':
    preprocess_bgl(**params)
