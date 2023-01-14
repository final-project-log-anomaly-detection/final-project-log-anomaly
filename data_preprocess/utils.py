import random
import json
from gensim.models import Word2Vec, KeyedVectors
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import numpy as np
import re
import gensim
from packaging import version
import calendar


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
        "../../google_news/GoogleNews-vectors-negative300.bin",
        binary=True
    )
    model.build_vocab([list(preTrained_model.key_to_index.keys())], update=True)
    model.wv.vectors_lockf = np.ones(len(model.wv), dtype=np.float32)
    model.wv.intersect_word2vec_format("../../google_news/GoogleNews-vectors-negative300.bin", binary=True, lockf=1.0)
    model.train(eventTemplateToken, total_examples=training_examples_count, epochs=5)
    model.save('../../'+f'{model_name}.model')
    print('finish train word2Vec model . . . . . ^^')


def trainWord2VecModelType2(token_train_list, model_name):
    print("start train word2Vec model. . . . .")
    model = KeyedVectors.load_word2vec_format(
        "../../google_news/GoogleNews-vectors-negative300.bin",
        binary=True
    )

    if version.parse(gensim.__version__) < version.parse("4.0.0"):
        embedder = Word2Vec(size=300, min_count=1)
        embedder.build_vocab(token_train_list)
        total_examples = embedder.corpus_count
        embedder.build_vocab([list(model.vocab.keys())], update=True)

        embedder.intersect_word2vec_format("../../google_news/GoogleNews-vectors-negative300.bin", binary=True)

        embedder.train(token_train_list, total_examples=total_examples, epochs=embedder.iter)
    else:
        embedder = Word2Vec(vector_size=300, min_count=1)
        embedder.build_vocab(token_train_list)
        total_examples = embedder.corpus_count
        embedder.build_vocab([list(model.key_to_index.keys())], update=True)

        embedder.wv.vectors_lockf = np.ones(len(embedder.wv), dtype=np.float32)
        embedder.wv.intersect_word2vec_format("../../google_news/GoogleNews-vectors-negative300.bin", binary=True)

        embedder.train(token_train_list, total_examples=total_examples, epochs=10)

    # embedder.wv.save_word2vec_format("../BGL-fine-tune-embedder.txt", binary=False)
    embedder.save('../../'+f'{model_name}.model')

    print('finish train word2Vec model . . . . . ^^')


def word2VecContinueLearning(eventTemplateToken, name):
    model = Word2Vec.load(name)
    model.train(eventTemplateToken, total_examples=1, epochs=10)
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


def text_cleansing(text):
    regex_except_token = r'\B(?!<\w+>\B)[^\w\s]'
    regex_expect_words = r'[^\w<>]+'
    output = re.sub(regex_except_token, '', text)
    output = re.sub(regex_expect_words, ' ', output)
    return output


def parse_datetime(data):
    str_datetime = data['DateTime']
    str_month = str_datetime.split('/')[1]
    str_datetime = str_datetime.replace(str_month, str(list(calendar.month_abbr).index(str_month)))
    result_datetime = pd.to_datetime(str_datetime, format="%d/%m/%Y:%H:%M:%S +%f")
    return result_datetime


def parse_month(data):
    data = str(list(calendar.month_abbr).index(data))
    return data
