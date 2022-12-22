import random
import json
from gensim.models import Word2Vec, KeyedVectors
import numpy as np


def decision(probability):
    return random.random() < probability


def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(
            obj,
            fw,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
            ensure_ascii=False,
        )


def trainWord2VecModel(eventTemplateToken, model_name):
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
    model.save(f'{model_name}.model')
    print('finish train word2Vec model . . . . . ^^')


def word2VecContinueLearning(eventTemplateToken, name):
    model = Word2Vec.load(name)
    model.train(eventTemplateToken, total_examples=1, epochs=5)
    model.save(name)
    print('training successful')


def convertWord2Vec(eventTemplateToken, model):
    eventVectors = []
    for token_list in eventTemplateToken:
        list_vector = []
        unknown_vocab = False
        for word in token_list:
            try:
                list_vector.append(model.wv[word])
            except:
                list_vector = []
                unknown_vocab = True
                break
        if unknown_vocab:
            eventVectors.append(None)
        else:
            eventVectors.append(list_vector)
    return eventVectors


def tokenizeData(data):
    list_data = []
    for _, row in data.iterrows():
        list_data.append(str(row['EventTemplateIdent']).split())

    return list_data
