import numpy as np
import pickle


# from sklearn.decomposition import PCA


def smallnorb_loader():
    f_x_train = open('datasets/small_norb/smallnorb_x_train_24300x2048.pkl', 'rb')
    x_train = pickle.load(f_x_train, encoding='bytes')
    f_y_train = open('datasets/small_norb/smallnorb_y_train.pkl', 'rb')
    y_train = pickle.load(f_y_train, encoding='bytes')

    n_values = np.max(y_train) + 1
    y_train = np.eye(n_values)[y_train]

    f_x_test = open('datasets/small_norb/smallnorb_x_test_24300x2048.pkl', 'rb')
    x_test = pickle.load(f_x_test, encoding='bytes')
    f_y_test = open('datasets/small_norb/smallnorb_y_test.pkl', 'rb')
    y_test = pickle.load(f_y_test, encoding='bytes')

    n_values = np.max(y_test) + 1
    y_test = np.eye(n_values)[y_test]

    return x_train, y_train, x_test, y_test


import os
import gzip
import numpy as np

# import matplotlib.pyplot as plt

'''
0:T-shirt/top
1:Trouser
2:Pullover
3:Dress
4:Coat
5:Sandal
6:Shirt
7:Sneaker
8:Bag
9:Ankle boot
'''


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels


def fashionmnist_loader():
    x_train, y_train = load_mnist('datasets/fashion_mnist')
    x_test, y_test = load_mnist('datasets/fashion_mnist', 't10k')
    y_tr = np.zeros((len(x_train), 10))
    for i in range(len(x_train)):
        y_tr[i][int(y_train[i])] = 1.0
    y_te = np.zeros((len(x_test), 10))
    for i in range(len(x_test)):
        y_te[i][int(y_test[i])] = 1.0
    return x_train / 255.0, y_tr, x_test / 255.0, y_te