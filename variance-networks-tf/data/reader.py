import os
import gzip
import random
import numpy as np
import cPickle as pickle
from data.downloader import download_mnist, download_cifar10, download_cifar100


def load(dataset):
    if dataset == 'mnist':
        return load_mnist()
    if dataset == 'cifar10':
        return load_cifar10()
    if dataset == 'cifar100':
        return load_cifar100()
    raise Exception('Load of %s not implemented yet' % dataset)


def load_mnist():
    """
    load_mnist taken from https://github.com/Lasagne/Lasagne/blob/master/examples/images.py
    :param base: base path to images dataset
    """

    def load_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28).transpose(0,2,3,1)
        return data / np.float32(255)

    def load_mnist_labels(filename):
        with gzip.open(filename, 'rb') as f:
            Y = np.frombuffer(f.read(), np.uint8, offset=8)
        return Y

    base = './data/mnist'
    if not os.path.exists(base):
        download_mnist()

    # We can now download and read the training and test set image and labels.
    X_train = load_mnist_images(base + '/train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(base + '/train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(base + '/t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(base + '/t10k-labels-idx1-ubyte.gz')

    return (X_train, y_train, X_test, y_test), X_train.shape[0], X_test.shape[0], (None, 28, 28, 1), 10


def load_cifar10():
    def load_CIFAR_batch(filename):
        with open(filename, 'rb') as f:
            datadict = pickle.load(f)
            Y = np.array(datadict['labels'])
            X = datadict['data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            return X, Y

    def load_CIFAR10(ROOT):
        xs, ys = [], []
        for b in range(1, 6):
            f = os.path.join(ROOT, 'data_batch_%d' % (b,))
            X, Y = load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)
        Xtr, Ytr = np.concatenate(xs), np.concatenate(ys)
        del X, Y
        Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
        return Xtr, Ytr, Xte, Yte

    base = './data/cifar10'
    if not os.path.exists(base):
        download_cifar10()

    # Load the raw CIFAR-10 data
    cifar10_dir = os.path.join(base, 'cifar-10-batches-py')
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    return (X_train, y_train, X_test, y_test), X_train.shape[0], X_test.shape[0], (None, 32, 32, 3), 10


def load_cifar100():
    def load_CIFAR_batch(filename, num):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = pickle.load(f)
            X = datadict['data']
            Y = datadict['coarse_labels']
            X = X.reshape(num, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            Y = np.array(Y)
            return X, Y

    base = './data/cifar100/'
    if not os.path.exists(base):
        download_cifar100()

    Xtr, Ytr = load_CIFAR_batch(os.path.join(base + '/cifar-100-python', 'train'), 50000)
    Xte, Yte = load_CIFAR_batch(os.path.join(base + '/cifar-100-python', 'test'), 10000)
    print Xtr.shape
    return (Xtr, Ytr, Xte, Yte), Xtr.shape[0], Xte.shape[0], (None, 32, 32, 3), 100


def batch_iterator_train_crop_flip(data, y, batchsize, shuffle=False, PIXELS=32, PAD_CROP=4):
    data = data.transpose((0, 3, 1, 2))
    n_samples = data.shape[0]
    indx = np.random.permutation(xrange(n_samples))
    for i in range((n_samples + batchsize - 1) // batchsize):
        sl = slice(i * batchsize, (i + 1) * batchsize)
        X_batch = data[indx[sl]]
        y_batch = y[indx[sl]]

        # pad and crop settings
        trans_1 = random.randint(0, (PAD_CROP*2))
        trans_2 = random.randint(0, (PAD_CROP*2))
        crop_x1 = trans_1
        crop_x2 = (PIXELS + trans_1)
        crop_y1 = trans_2
        crop_y2 = (PIXELS + trans_2)

        # flip left-right choice
        flip_lr = random.randint(0,1)

        # set empty copy to hold augmented images so that we don't overwrite
        X_batch_aug = np.copy(X_batch)

        # for each image in the batch do the augmentation
        for j in range(X_batch.shape[0]):
            # for each image channel
            for k in range(X_batch.shape[1]):
                # pad and crop images
                img_pad = np.pad(
                    X_batch_aug[j, k], pad_width=((PAD_CROP, PAD_CROP), (PAD_CROP, PAD_CROP)), mode='constant')
                X_batch_aug[j, k] = img_pad[crop_x1:crop_x2, crop_y1:crop_y2]

                # flip left-right if chosen
                if flip_lr == 1:
                    X_batch_aug[j, k] = np.fliplr(X_batch_aug[j,k])

        X_batch_aug = X_batch_aug.transpose((0, 2, 3, 1))
        yield X_batch_aug, y_batch
