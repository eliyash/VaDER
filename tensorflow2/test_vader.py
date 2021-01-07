import json
import pickle
from pathlib import Path

import numpy as np
import os
import sys
import tensorflow as tf

from read_data import read_data, read_premade, DAYS_ORDERED
from vader import VADER

tf.config.run_functions_eagerly(False)
sys.path.append(os.getcwd())
sys.path.append('tensorflow2')


def prepare_data():
    # generating some simple random data [ns * 2 samples, nt - 1 time points, 2 variables]
    nt = 8
    ns = 200
    sigma = 0.5
    mu1 = -2
    mu2 = 2
    a1 = np.random.normal(mu1, sigma, ns)
    a2 = np.random.normal(mu2, sigma, ns)
    # one variable with two clusters
    x0 = np.outer(a1, 2 * np.append(np.arange(-nt/2, 0), 0.5 * np.arange(0, nt/2)[1:]))
    x1 = np.outer(a2, 0.5 * np.append(np.arange(-nt/2, 0), 2 * np.arange(0, nt/2)[1:]))
    x_train = np.concatenate((x0, x1), axis=0)
    y_train = np.repeat([0, 1], ns)
    # add another variable as a random permutation of the first one
    # resulting in four clusters in total
    ii = np.random.permutation(ns * 2)
    x_train = np.stack((x_train, x_train[ii, ]), axis=2)
    # we now get four clusters in total
    y_train = y_train * 2**0 + y_train[ii] * 2**1

    # normalize (better for fitting)
    for i in np.arange(x_train.shape[2]):
        x_train[:, :, i] = (x_train[:, :, i] - np.mean(x_train[:, :, i])) / np.std(x_train[:, :, i])

    # randomly re-order the samples
    ii = np.random.permutation(ns * 2)
    x_train = x_train[ii, :]
    y_train = y_train[ii]
    # Randomly set 50% of values to missing (0: missing, 1: present)
    # Note: All x_train[i,j] for which w_train[i,j] == 0 are treated as missing (i.e. their specific value is ignored)
    w_train = np.random.choice(2, x_train.shape)
    return x_train, y_train, w_train


def test1():
    save_path = os.path.join('test_vader', 'vader.ckpt')
    x_train, y_train, w_train = prepare_data()
    # json.dump(x_train, open("x_train.json", "wb"))
    # json.dump(y_train, open("y_train.json", "wb"))
    # json.dump(w_train, open("w_train.json", "wb"))
    pickle.dump(x_train, open("x_train.pickle", "wb"))
    pickle.dump(y_train, open("y_train.pickle", "wb"))
    pickle.dump(w_train, open("w_train.pickle", "wb"))
    # Note: y_train is used purely for monitoring performance when a ground truth clustering is available.
    # It can be omitted if no ground truth is available.
    vader = VADER(x_train=x_train, w_train=w_train, y_train=y_train, save_path=save_path, n_hidden=[12, 2], k=4,
                  learning_rate=1e-3, output_activation=None, recurrent=True, batch_size=16)
    # pre-train without latent loss
    vader.pre_fit(n_epoch=50, verbose=True)
    # train with latent loss
    vader.fit(n_epoch=50, verbose=True)
    # get the clusters
    c = vader.cluster(x_train)
    # get the re-constructions
    p = vader.predict(x_train)


def get_dete_for_seconed_test():
    # Run VaDER non-recurrently (ordinary VAE with GM prior)
    nt = int(8)
    ns = int(2e2)
    sigma = np.diag(np.repeat(2, nt))
    mu1 = np.repeat(-1, nt)
    mu2 = np.repeat(1, nt)
    a1 = np.random.multivariate_normal(mu1, sigma, ns)
    a2 = np.random.multivariate_normal(mu2, sigma, ns)
    x_train = np.concatenate((a1, a2), axis=0)
    y_train = np.repeat([0, 1], ns)
    ii = np.random.permutation(ns * 2)
    x_train = x_train[ii, :]
    y_train = y_train[ii]
    # normalize (better for fitting)
    x_train = (x_train - np.mean(x_train)) / np.std(x_train)
    return x_train, y_train


def test2():
    x_train, y_train = get_dete_for_seconed_test()
    vader = VADER(x_train=x_train, y_train=y_train, n_hidden=[12, 2], k=2, learning_rate=1e-3, output_activation=None,
                  recurrent=False, batch_size=16)
    # pre-train without latent loss
    vader.pre_fit(n_epoch=50, verbose=True)
    # train with latent loss
    vader.fit(n_epoch=50, verbose=True)
    # get the clusters
    c = vader.cluster(x_train)
    # get the re-constructions
    p = vader.predict(x_train)
    # compute the loss given the network
    l = vader.get_loss(x_train)
    # generate some samples
    g = vader.generate(10)
    # compute the loss given the network
    l = vader.get_loss(x_train)


def main():
    output_path = Path('../output/try2_exactly_7_times')
    output_path.mkdir(exist_ok=True)
    save_path = output_path / 'vader.ckpt'

    # w_train, x_train, names = read_premade(DAYS_ORDERED)
    w_train, x_train, names = read_data()
    x_train = (x_train - np.mean(x_train)) / np.std(x_train)

    vader = VADER(x_train=x_train, w_train=w_train, save_path=save_path, n_hidden=[128, 32], k=5,
                  learning_rate=1e-3, output_activation=None, recurrent=True, batch_size=8, alpha=0.1)
    # pre-train without latent loss
    vader.pre_fit(n_epoch=20, verbose=True)
    # train with latent loss
    vader.fit(n_epoch=100, verbose=True)
    # get the clusters
    c = vader.cluster(x_train, w_train)
    # get the re-constructions
    p = vader.predict(x_train)

    print(vader.get_clusters_on_x())


if __name__ == '__main__':
    main()
    # test1()
