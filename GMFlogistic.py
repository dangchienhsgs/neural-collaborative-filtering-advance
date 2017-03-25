'''
Created on Aug 9, 2016

@author: he8819197
'''
import numpy as np
import theano.tensor as T
import keras
import scipy
import random
from keras import backend as K
from keras import initializers
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Input, Dense, merge, Reshape, Merge, Flatten
from keras.layers.embeddings import Embedding
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
from Dataset import Dataset
from evaluate import evaluate_model
from time import time
import multiprocessing as mp
import sys
import math
from scipy.sparse import dok_matrix

from analysis.create_vector_item import read_dictionaries
from analysis.read_movie import read_vectors


def init_normal():
    return initializers.RandomNormal(mean=0.5)


def get_model(num_users, num_items, num_words, latent_dim, regs=[0, 0]):
    """
    Input of this model: id of user, id of item id of word
    :param num_users:
    :param num_items:
    :param num_words:
    :param latent_dim:
    :param regs:
    :return:
    """

    assert len(regs) == 2  # regularization for user and item, respectively

    print('Number of words %d' % num_words)
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    word_input = Input(shape=(1,), dtype='int32', name='word_input')

    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=latent_dim, name='user_embedding',
                                  embeddings_initializer=init_normal(), activity_regularizer=l2(regs[0]),
                                  input_length=1)
    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=latent_dim, name='item_embedding',
                                  embeddings_initializer=init_normal(), activity_regularizer=l2(regs[0]),
                                  input_length=1)
    MF_Embedding_Word = Embedding(input_dim=num_words, output_dim=latent_dim, name='word_embedding',
                                  embeddings_initializer=init_normal(), activity_regularizer=l2(regs[0]),
                                  input_length=1)

    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))
    word_latent = Flatten()(MF_Embedding_Word(word_input))

    # Element-wise product of user and item embeddings 
    predict_vector = merge([user_latent, item_latent], mode='mul')
    language_vector = merge([item_latent, word_latent], mode='mul')

    # Final prediction layer
    # prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(predict_vector)
    language_value = Dense(1, activation='sigmoid', init='lecun_uniform', name='language_value')(language_vector)

    model = Model(inputs=[user_input, item_input, word_input],
                  outputs=[prediction, language_value])

    return model


def get_train_instances(train, num_negatives, weight_negatives, user_weights):
    """
    :param train:
    :param num_negatives:
    :param weight_negatives:
    :param user_weights:
    :return:
    """

    user_input, item_input, labels, weights = [], [], [], []
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        weights.append(user_weights[u])
        # negative instances

        for t in range(num_negatives):
            j = np.random.randint(num_items)

            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
            weights.append(weight_negatives * user_weights[u])
    return user_input, item_input, labels, weights


if __name__ == '__main__':
    dataset_name = "ml-1m"
    num_factors = 10
    regs = [0, 0]
    num_negatives = 1
    weight_negatives = 1
    learner = "Adam"
    learning_rate = 0.001
    epochs = 100
    batch_size = 256
    verbose = 1
    dictionary = read_dictionaries()
    dict_size = len(dictionary)

    if len(sys.argv) > 3:
        dataset_name = sys.argv[1]
        num_factors = int(sys.argv[2])
        regs = eval(sys.argv[3])
        num_negatives = int(sys.argv[4])
        weight_negatives = float(sys.argv[5])
        learner = sys.argv[6]
        learning_rate = float(sys.argv[7])
        epochs = int(sys.argv[8])
        batch_size = int(sys.argv[9])
        verbose = int(sys.argv[10])

    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print(
        "GMF-logistic (%s) Settings: num_factors=%d, batch_size=%d, learning_rate=%.1e, num_neg=%d, weight_neg=%.2f, regs=%s, epochs=%d, verbose=%d"
        % (learner, num_factors, batch_size, learning_rate, num_negatives, weight_negatives, regs, epochs, verbose))

    # Loading data
    t1 = time()
    dataset = Dataset("Data/" + dataset_name)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    total_weight_per_user = train.nnz / float(num_users)
    train_csr, user_weights = train.tocsr(), []
    for u in range(num_users):
        user_weights.append(1)
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    print("Create model")
    model = get_model(num_users, num_items, dict_size, num_factors, regs)
    print("Finish create model")

    # ---------------- compile model ----------------------
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss=['binary_crossentropy', 'binary_crossentropy'],
                      loss_weights=[1, 0.1])
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss=['binary_crossentropy', 'binary_crossentropy'],
                      loss_weights=[1, 0.1])
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss=['binary_crossentropy', 'binary_crossentropy'],
                      loss_weights=[1, 0.1])
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss=['binary_crossentropy', 'binary_crossentropy'],
                      loss_weights=[1, 0.1])
    # print(model.summary())
    # -----------------------------------------------------

    # Init performance
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()

    mf_embedding_norm = np.linalg.norm(model.get_layer('user_embedding').get_weights()) + np.linalg.norm(
        model.get_layer('item_embedding').get_weights())

    p_norm = np.linalg.norm(model.get_layer('prediction').get_weights()[0])
    print(model.get_layer('user_embedding').get_weights())
    print(p_norm)

    print('Init: HR = %.4f, NDCG = %.4f\t MF_norm=%.1f, p_norm=%.2f' %
          (hr, ndcg, mf_embedding_norm, p_norm))

    # Train model
    loss_pre = sys.float_info.max
    best_hr, best_ndcg = 0, 0

    # -------------------
    # read vector of item
    items_vectors = read_vectors()
    # -------------------

    # create data
    user_input, item_input, labels, weights = get_train_instances(train, num_negatives, weight_negatives,
                                                                  user_weights)

    print(len(user_input))
    # get item vector
    users_train = []
    items_train = []
    words_train = []
    labels_train = []
    item_vector_value = []

    print("Create data")

    cache_item = {}

    # for not crawl item
    item_vector_empty = dok_matrix((1, items_vectors.shape[1]))
    item_vector_empty[0, random.randint(0, items_vectors.shape[1])] = 1

    for item in range(0, items_vectors.shape[0]):
        if item in items_vectors.keys():
            item_vec = items_vectors[item]
        else:
            item_vec = item_vector_empty

        t1 = item_vec.nonzero()[1]
        t2 = item_vec.values()

        cache_item[item] = [item_vec.nonzero()[1], item_vec.values()]

    for i in range(len(user_input)):
        user = user_input[i]
        item = item_input[i]

        if item in items_vectors.keys():
            item_vec = items_vectors[item]
        else:
            item_vec = item_vector_empty

        # add good item
        size = len(item_vec.nonzero()[1])
        users_train.extend([user] * size)
        items_train.extend([item] * size)
        labels_train.extend([labels[i]] * size)

        t = cache_item[item]
        words_train.extend(t[0])
        item_vector_value.extend(t[1])

        # add non word in item


        print(i, len(user_input), len(users_train), len(labels_train), len(items_train), len(words_train),
              len(item_vector_value))

    print("Data size ", len(user_input))
    for epoch in range(epochs):
        print("Epoch %d" % epoch)
        t1 = time()
        # Generate training instances
        print("Start fit data")

        # print("Before", model.get_layer('user_embedding').get_weights())
        hist = model.fit([np.array(users_train), np.array(items_train), np.array(words_train)],  # input
                         [np.array(labels_train), np.array(item_vector_value)],  # labels
                         batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
        print("Finish fit data")
        # print("After", model.get_layer('user_embedding').get_weights())

        t2 = time()

        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            print(hist.history.keys())
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]

            mf_embedding_norm = np.linalg.norm(model.get_layer('user_embedding').get_weights()) + np.linalg.norm(
                model.get_layer('item_embedding').get_weights())

            p_norm = np.linalg.norm(model.get_layer('prediction').get_weights()[0])
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s] MF_norm=%.1f, p_norm=%.2f'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2, mf_embedding_norm, p_norm))
            if hr > best_hr:
                best_hr = hr
                if hr > 0.6:
                    model.save_weights('pretrain/%s_GMF_%d_neg_%d_hr_%.4f_ndcg_%.4f.h5' % (
                        dataset_name, num_factors, num_negatives, hr, ndcg), overwrite=True)
            if ndcg > best_ndcg:
                best_ndcg = ndcg

    print("End. best HR = %.4f, best NDCG = %.4f" % (best_hr, best_ndcg))
