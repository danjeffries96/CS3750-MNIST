import tensorflow as tf

import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, ClassifierMixin
from ext_test import kaggle_test


def augment_data(X, y):
    X_aug = []
    y_aug = []

    for x, label in zip(X, y):
        # include originals
        X_aug.append(x)
        y_aug.append(label)

        # random rotations up to -20/20 degrees
        # for degree in range(-60, 60, 5):
        #     rotated = tf.contrib.keras.preprocessing.image.random_rotation(
        #                 x, degree, row_axis=0, col_axis=1, channel_axis=2)
        #     X_aug.append(rotated)
        #     y_aug.append(label)

        #     # random shears up to 40% intensity        
        #     for sh in range(-40, 40, 5):
        #         sh /- 10
        #         sheared = tf.contrib.keras.preprocessing.image.random_shear(
        #                 rotated, sh, row_axis=0, col_axis=1, channel_axis=2)
        #         X_aug.append(sheared)
        #         y_aug.append(label)

        for _ in range(50):
            rot = np.random.randint(-60, 60) 
            sh = np.random.randint(-40, 40) / 10
            rotated = tf.contrib.keras.preprocessing.image.random_rotation(
                        x, rot, row_axis=0, col_axis=1, channel_axis=2)
            X_aug.append(rotated)
            y_aug.append(label)
            sheared = tf.contrib.keras.preprocessing.image.random_shear(
                    rotated, sh, row_axis=0, col_axis=1, channel_axis=2)
            X_aug.append(sheared)
            y_aug.append(label)

            
        # random shifts 20% left, right, up or down
        # for xsh in range(0, 5, 5):
        #     xsh /= 10
        #     for ysh in range(0, 5, 5):
        #         ysh /= 10
        #         shifted = tf.contrib.keras.preprocessing.image.random_shift(
        #                 x, xsh, ysh, row_axis=0, col_axis=1, channel_axis=2)
        #         X_aug.append(shifted)
        #         y_aug.append(label)

        #         # rotate and shift
        #         # for degree in range(0, 24, 4):
        #         #     rotated = tf.contrib.keras.preprocessing.image.random_rotation(
        #         #                 shifted, degree, row_axis=0, col_axis=1, channel_axis=2)
        #         #     X_aug.append(rotated)
        #         #     y_aug.append(label)

        # random zoom up to 20%
        zoomed = tf.contrib.keras.preprocessing.image.random_zoom(
                x, (0.9, 1.0), row_axis=0, col_axis=1, channel_axis=2)
        X_aug.append(zoomed)
        y_aug.append(label)
        
    return  np.array(X_aug), np.array(y_aug)

X = np.load("x_mnist1000.npy")
X = X.reshape((-1, 28, 28, 1))
y = np.load("y_mnist1000.npy")

np.random.seed(1)

X, y = augment_data(X, y)

indices = np.random.permutation(len(X))

valid_ind = indices[:2000]
test_ind = indices[2000:4000]
train_ind = indices[4000:]

X_train = X[train_ind]
y_train = y[train_ind]

X_validation = X[valid_ind]
y_validation = y[valid_ind]

X_test = X[test_ind]
y_test = y[test_ind]
X_test, y_test = kaggle_test()

def class_prevalence(v):
    return [sum(v == c) / len(v) for c in range(0, 10)]
print("Train cp:", class_prevalence(y_train))
print("Validation cp:", class_prevalence(y_validation))
print("Test cp:", class_prevalence(y_test))

# pad for lenet
pad_dims = ((0, 0), (2, 2), (2, 2), (0, 0))
X_train = np.pad(X_train, pad_dims, "constant")
X_validation = np.pad(X_validation, pad_dims, "constant")
X_test = np.pad(X_test, pad_dims, "constant")

N = len(X_train)

class CNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, verbose=1, optimizer_class=tf.train.AdamOptimizer,
            learning_rate=0.001, batch_size=100, activations=None,
            dropout_rate=0, logdir=None):
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate

        self.logdir = logdir
        self.writer = None
        
        self.is_training = True
        self.best_loss = np.infty

        self.v_scores = []
        self.v_columns = ["Epoch", "Validation"]

        if activations is None:
            activations = {
                "conv1": tf.nn.tanh,
                "avg_pool": tf.nn.tanh,
                "conv2": tf.nn.tanh,
                "avg_pool2": tf.nn.tanh,
                "conv3": tf.nn.tanh,
                "dense": tf.nn.tanh,
            }
        self.activations = activations

        self.verbose = verbose

        self._params = [
            "optimizer_class",
            "learning_rate",
            "batch_size",
            "activations",
            "learning_rate",
            "dropout_rate",
        ]

        self.session = None

        self.init_sess()
    
    def _build_graph(self):

        self.X = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
        self.y = tf.placeholder(tf.int32, (None))

        conv1 = tf.layers.conv2d(self.X, filters=6, kernel_size=5,  
                strides=1, padding="valid", activation=self.activations["conv1"])
        conv1 = tf.layers.dropout(conv1, rate=self.dropout_rate)

        avg_pool = tf.layers.average_pooling2d(conv1, pool_size=2, strides=2)
        avg_pool = self.activations["avg_pool"](avg_pool)
        avg_pool = tf.layers.dropout(avg_pool, rate=self.dropout_rate)

        conv2 = tf.layers.conv2d(avg_pool, filters=16, kernel_size=5, strides=1,
                               padding="valid", activation=self.activations["conv2"])
        conv2 = tf.layers.dropout(conv2, rate=self.dropout_rate)


        avg_pool2 = tf.layers.average_pooling2d(conv2, pool_size = 2, strides=2)
        avg_pool2 = self.activations["avg_pool2"](avg_pool2)
        avg_pool2 = tf.layers.dropout(avg_pool2, rate=self.dropout_rate)


        conv3 = tf.layers.conv2d(avg_pool2, filters=120, kernel_size=5, strides=1,
                              padding="valid", activation=self.activations["conv3"])
        conv3 = tf.layers.dropout(conv3, rate=self.dropout_rate)

        # flatten for dense layers
        flat = tf.reshape(conv3, [-1, 120])
        flat = tf.layers.dropout(flat, rate=self.dropout_rate)

        dense = tf.layers.dense(inputs=flat, units=84, activation=self.activations["dense"])
        logits = tf.layers.dense(dense, units=10)
        softmax = tf.nn.softmax(logits)

        self._predict = tf.argmax(softmax, axis=1)

        y_labels = tf.one_hot(self.y, 10)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_labels)
        self.loss = tf.reduce_mean(cross_entropy)
        opt = self.optimizer_class(learning_rate=self.learning_rate)
        self.train_op = opt.minimize(self.loss)

        self.init = tf.global_variables_initializer()

    def _build_graph_tb(self):
        """
        Textbook solution to chapter 11 problem 7
        """
        self.X = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        self.y = tf.placeholder(tf.int32, shape=(None))

        conv1_fmaps = 32
        conv1_ksize = 3
        conv1_stride = 1
        conv1_pad = "SAME"

        conv2_fmaps = 64
        conv2_ksize = 3
        conv2_stride = 1
        conv2_pad = "SAME"
        conv2_dropout_rate = 0.25

        pool3_fmaps = conv2_fmaps

        n_fc1 = 128
        fc1_dropout_rate = 0.5

        n_outputs = 10

        conv1 = tf.layers.conv2d(self.X, filters=conv1_fmaps, kernel_size=conv1_ksize,
                            strides=conv1_stride, padding=conv1_pad,
                            activation=self.activations["conv1"], name="conv1")
        conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                            strides=conv2_stride, padding=conv2_pad,
                            activation=self.activations["conv2"], name="conv2")
       	pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], 
                                strides=[1, 2, 2, 1], padding="VALID")
       	pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 14 * 14])
       	pool3_flat_drop = tf.layers.dropout(pool3_flat, self.dropout_rate)
       
        fc1 = tf.layers.dense(pool3_flat_drop, n_fc1, 
                                activation=self.activations["dense"], name="fc1")
        fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate)

        logits = tf.layers.dense(fc1, n_outputs, name="output")
        Y_proba = tf.nn.softmax(logits, name="Y_proba")
        
        self._predict = tf.argmax(Y_proba, axis=1)

        
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.y)
        self.loss = tf.reduce_mean(xentropy)
        opt = self.optimizer_class(learning_rate=self.learning_rate)
        self.train_op = opt.minimize(loss)

        self.init = tf.global_variables_initializer()

    def get_params(self, deep=None):
        return {p: self.__getattribute__(p) for p in self._params}

    def set_params(self, **params):
        self.__dict__.update(params)
        return self

    def log(self, *msg, level=0):
        if level < self.verbose:
            print(*msg)

    def tf_log(self):
        # log for tensorboard
        if self.logdir:
            if not self.writer:
                self.writer = tf.summary.FileWriter(self.logdir, self.graph)

    def init_sess(self):
        if self.session:
            self.session.close()

        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_graph()

        self.tf_log()

        # print params
        self.log(self.get_params(), level=1)

        self.session = tf.Session(graph=self.graph)
        with self.session.as_default() as sess:
            self.init.run()

    def fit(self, X, y, n_epochs=5, X_valid=None, y_valid=None):
        with self.session.as_default() as sess:
            self.init.run()

            checks_since_prog = 0
        
            for i in range(n_epochs):
              if i % (n_epochs / 10) == 0:
                  self.log("Epoch:", i, level=2)
              X_train, y_train = shuffle(X, y)
              for off in range(0, len(X_train), self.batch_size):
                  end = off + self.batch_size
                  batch_x, batch_y = X_train[off:end], y_train[off:end]
                  sess.run(self.train_op, feed_dict={self.X: batch_x, self.y: batch_y})
                  loss = sess.run(self.loss, feed_dict={self.X: batch_x, self.y: batch_y})

                  checks_since_prog += 1
                  if loss < self.best_loss:
                      self.best_loss = loss
                      checks_since_prog = 0
                      best_params_ = self.get_params()
                  if checks_since_prog > 50:
                      print("Early stopping!")
                      # self.log_filters(X, y)
                      return self        
            
                  valid_acc = self.score(X_valid, y_valid) 
                  self.v_scores.append({"Epoch": i * self.batch_size + off, "Validation": valid_acc})
                  msg = "Loss: %.4f; Validation accuracy: %s" % (loss, valid_acc)
                  self.log(msg)

        # self.log_filters(X, y)
        return self

    def save_model(self):
        with self.session.as_default() as sess:
            saver = tf.train.Saver()
            dt = datetime.now().strftime("%H-%M-%S")
            save_path = saver.save(self.session, "./models/%s.ckpt" % dt)

    def restore_model(self, model_name):
        with self.session.as_default() as sess:

            reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
            print(reuse_vars_dict)
            restore_saver = tf.train.Saver(reuse_vars_dict)
            saver.restore(sess, "./models/" + model_name)

    def log_filters(self, X, y):
        # log conv output
        conv1 = self.graph.get_tensor_by_name("conv2d/Conv2D:0")
        conv1 = tf.reduce_mean(conv1, axis=3)
        sh = conv1.shape
        conv1 = tf.reshape(conv1, shape=(-1, sh[1], sh[2], 1))

        summary_op = tf.summary.image("conv2d/Conv2D:0", conv1)
        summary = self.session.run(summary_op, feed_dict={self.X:X, self.y:y})

        self.writer.add_summary(summary)

    def predict(self, X):
        with self.session.as_default() as sess:
            prediction = sess.run(self._predict, feed_dict={self.X: X})
            return prediction

    def score(self, X_eval, y_eval):
        with self.session.as_default() as sess:
            out = sess.run(self._predict, feed_dict={self.X: X_eval, self.y: y_eval})
            return np.mean(out == y_eval)

    def save_val_plot(self):
        df = pd.DataFrame(data=self.v_scores, columns=self.v_columns)
        fn = "./val_scores/" + datetime.now().strftime("%H-%M-%S") + ".csv"
        df.to_csv(fn, index=False)

print("Num samples:", len(X_train))

default_activations = {
    "conv1": tf.nn.tanh,
    "avg_pool": tf.nn.tanh,
    "conv2": tf.nn.tanh,
    "avg_pool2": tf.nn.tanh,
    "conv3": tf.nn.tanh,
    "dense": tf.nn.tanh,
}

from copy import copy

clf = CNNClassifier()

a1 = copy(default_activations)
a1["avg_pool2"] = tf.nn.elu
a1["conv3"] = tf.nn.elu
a1["dense"] = tf.nn.elu

elu = {x: tf.nn.elu for x in default_activations}
selu = {x: tf.nn.selu for x in default_activations}
relu = {x: tf.nn.relu for x in default_activations}
tanh= {x: tf.nn.tanh for x in default_activations}

param_grid = {
    # "optimizer_class": [tf.train.AdamOptimizer, tf.train.AdagradOptimizer, tf.train.MomentumOptimizer],
    "verbose":  [1],
    "activations": [relu, elu, selu],
    # "learning_rate": [0.1, 0.01, 0.001],
    "dropout_rate": [0.25],
}

from functools import partial

momentum = partial(tf.train.MomentumOptimizer, momentum=0.99)

param_grid = {
    "verbose":  2,
    "activations": selu,
    "learning_rate": 0.001,
    "dropout_rate": 0.55,
    "batch_size": len(X_train) // 20,
}

clf.set_params(**param_grid)
clf.logdir = "./tflogs"
clf.fit(X_train, y_train, n_epochs=1000, X_valid=X_validation, y_valid=y_validation)
print("Test size:", len(X_test))
test_msg = "Test accuracy: %s" % clf.score(X_test, y_test)
print(test_msg)
clf.save_model()
clf.save_val_plot()

log_name = "./logs/cnn_log_" + datetime.now().strftime("%H-%M-%S") + ".log"
with open(log_name, "w") as log:
    log.write("Param grid:\n")
    param_grid = clf.get_params()
    for p in param_grid:
        log.write("%s: %s\n" % (p, param_grid[p]))
    log.write(test_msg)
 
#     gs = GridSearchCV(clf, param_grid=param_grid, n_jobs=-1)
#     gs.fit(X_train, y_train, n_epochs=25, X_valid=X_validation, y_valid=y_validation)
#     best_msg = "best score: %s, params: %s" % (gs.best_score_, gs.best_estimator_.get_params())
#     print(best_msg)
#     log.write(best_msg)
# 
#     test_msg = "Test accuracy: %s" % gs.best_estimator_.score(X_test, y_test)
#     print(test_msg)
#     log.write(test_msg)
# 
#     gs.best_estimator_.save_val_plot()
