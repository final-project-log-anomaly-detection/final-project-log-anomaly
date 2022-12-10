# import tensorflow as tf

# text_dataset = tf.data.Dataset.from_tensor_slices(["foo", "bar", "baz"])
# max_features = 5000  # Maximum vocab size.
# max_len = 4  # Sequence length to pad the outputs to.

# # Create the layer.
# vectorize_layer = tf.keras.layers.TextVectorization(
#     max_tokens=max_features,
#     output_mode='int',
#     output_sequence_length=max_len)

# # Now that the vocab layer has been created, call `adapt` on the
# # text-only dataset to create the vocabulary. You don't have to batch,
# # but for large datasets this means we're not keeping spare copies of
# # the dataset.
# vectorize_layer.adapt(text_dataset.batch(64))

# # Create the model that uses the vectorize text layer
# model = tf.keras.models.Sequential()

# # Start by creating an explicit input layer. It needs to have a shape of
# # (1,) (because we need to guarantee that there is exactly one string
# # input per batch), and the dtype needs to be 'string'.
# model.add(tf.keras.Input(shape=(1,), dtype=tf.string))

# # The first layer in our model is the vectorization layer. After this
# # layer, we have a tensor of shape (batch_size, max_len) containing
# # vocab indices.
# model.add(vectorize_layer)

# # Now, the model can map strings to integers, and you can add an
# # embedding layer to map these integers to learned embeddings.
# input_data = [["foo qux bar"], ["qux baz"]]
# print(model.predict(input_data))

# from sklearn.feature_extraction.text import TfidfVectorizer
# corpus = [
#     'This is the first document.',
#     'This document is the second document.',
#     'And this is the third one.',
#     'Is this the first document?',
# ]
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(corpus)
# print(vectorizer.get_feature_names_out())


# print(X.shape)


# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.pipeline import Pipeline
# corpus = ['this is the first document',
#           'this document is the second document',
#           'and this is the third one',
#           'is this the first document']
# vocabulary = ['this', 'document', 'first', 'is', 'second', 'the',
#               'and', 'one']
# pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
#                  ('tfid', TfidfTransformer())]).fit(corpus)
# print(pipe)
# print(pipe['count'].transform(corpus).toarray())


# print(pipe['tfid'].idf_)


# print(pipe.transform(corpus).shape)

# from sklearn.feature_extraction.text import TfidfVectorizer
# corpus = [
#     'This is the first document.'
# ]
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(corpus)
# print(X)
# vectorizer.get_feature_names_out()


# print(X.shape)


# import torch
# import torch.nn as nn
# from gensim.test.utils import common_texts
# from gensim.models import Word2Vec
# model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
# model.save("word2vec.model")


# embedding = nn.Embedding(10, 3)
# print(embedding)

# input = torch.LongTensor()
# print(embedding(input))

# from gensim.models import Word2Vec
# import dataloader

# session_train, session_test = dataloader.load_sessions('data_preprocess/processed/HDFS/hdfs_0.0_tar/')
# d = []

# print(session_train)
# for k, v in session_train.items():
#     for val in v['templates']:
#         print(val)


# model = Word2Vec(d, min_count=1)

# print(model.wv.vocab)


# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import numpy as np

# x = {'text', 'the', 'leader', 'prime',
#      'natural', 'language'}

# # create the dict.
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(x)

# # number of unique words in dict.
# print("Number of unique words in dictionary=",
#       len(tokenizer.word_index))
# print("Dictionary is = ", tokenizer.word_index)


# import gensim.downloader
# from gensim.models.word2vec import Word2Vec
# from gensim.test.utils import common_texts
# text = gensim.downloader.load('text8')
# model = Word2Vec()
# print(common_texts)
# print(model.wv['hello'])


a = [1, 2, 3, 4, 5, 6, 'asd', 7]

for i in a:
    try:
        print(int(i))
    except:
        print('hello')
        print(i)
