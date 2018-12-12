import tensorflow as tf
import numpy as np
from datetime import datetime
from ext_test import kaggle_test
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class DataAugment(TransformerMixin):
    def __init__(self, n_iter=10, rotation=None, shear=None, rot_and_shear=False,
            x_shifts=None, y_shifts=None):
        self.n_iter = n_iter
        self.rotation = rotation or []
        self.shear = shear or []
        self.rot_and_shear = rot_and_shear

        self.x_shifts = x_shifts or []
        self.y_shifts = y_shifts or []

        self._params = {
            "n_iter",
            "rotation",
            "shear",
            "rot_and_shear",
            "x_shifts",
            "y_shifts",
        }

    def get_params(self, deep=None):
        return {p: self.__getattribute__(p) for p in self._params}

    def set_params(self, **params):
        self.__dict__.update(params)
        return self

    def fit(self, X, Y):
        pass

    def transform(self, X):
        return X

    def fit_transform(self, X, y):
        #print("before TF, shapes = %s, %s" % (X.shape, y.shape))
        X_aug = []
        y_aug = []

        for x, label in zip(X, y):
            # include originals
            X_aug.append(x)
            y_aug.append(label)

            for _ in range(self.n_iter):
                # random rotations up to -60/60 degrees
                for degree in self.rotation:
                    rotated = tf.contrib.keras.preprocessing.image.random_rotation(
                                x, degree, row_axis=0, col_axis=1, channel_axis=2)
                    X_aug.append(rotated)
                    y_aug.append(label)


                    if self.rot_and_shear:
                        # random shears up to 40% intensity        
                        for sh in self.shear:
                            sh /= 10
                            sheared = tf.contrib.keras.preprocessing.image.random_shear(
                                    rotated, sh, row_axis=0, col_axis=1, channel_axis=2)
                            X_aug.append(sheared)
                            y_aug.append(label)

                # random shifts 20% left, right, up or down
                for xsh in self.x_shifts:
                    xsh /= 10
                    for ysh in self.y_shifts:
                        ysh /= 10
                        shifted = tf.contrib.keras.preprocessing.image.random_shift(
                                x, xsh, ysh, row_axis=0, col_axis=1, channel_axis=2)
                        X_aug.append(shifted)
                        y_aug.append(label)

        X_aug = np.array(X_aug)
        y_aug = np.array(y_aug)

        # print("after TF, shapes = %s, %s" % (X_aug.shape, y_aug.shape))
        return (X_aug, y_aug)

X = np.load("x_mnist1000.npy")
X = X.reshape((-1, 28, 28, 1))
y = np.load("y_mnist1000.npy")

np.random.seed(1)

indices = np.random.permutation(len(X))

test_ind = indices[:100]
# valid_ind = indices[100:200]
train_ind = indices[100:]

X_train = X[train_ind]
y_train = y[train_ind]

# X_validation = X[valid_ind]
# y_validation = y[valid_ind]

X_test = X[test_ind]
y_test = y[test_ind]
# X_test, y_test = kaggle_test()

def class_prevalence(v):
    return [sum(v == c) / len(v) for c in range(0, 10)]
print("Train cp:", class_prevalence(y_train))
# print("Validation cp:", class_prevalence(y_validation))
print("Test cp:", class_prevalence(y_test))

# pad for lenet
pad_dims = ((0, 0), (2, 2), (2, 2), (0, 0))
X_train = np.pad(X_train, pad_dims, "constant")
X_test = np.pad(X_test, pad_dims, "constant")

N = len(X_train)

class CNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, verbose=2, optimizer_class=tf.train.AdamOptimizer,
            learning_rate=0.0005, batch_size=500, max_iter=200000,
            activations=None, dropout_rate=0.35, logdir=None, using_da=True):
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.max_iter = max_iter

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

        # hack to use DataAugment in pipeline 
        # (transformers only return x)
        self.using_da = using_da

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
        self.train_op = opt.minimize(self.loss)

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
        self.log(self.get_params(), level=2)

        self.session = tf.Session(graph=self.graph)
        with self.session.as_default() as sess:
            self.init.run()

    def fit(self, X, y, n_epochs=10, X_valid=None, y_valid=None):

        # transformers in pipeline can only return X 
        # so X could be a tuple (X, y)
        if self.using_da:
            X, y = X

        print("Calling fit with shapes: %s, %s" % (X.shape, y.shape))

        bi = 0

        with self.session.as_default() as sess:
            self.init.run()

            checks_since_prog = 0
        
            # for i in range(n_epochs):
            #   if i % (n_epochs / 10) == 0:
            #       self.log("Epoch:", i, level=2)
            #   X_train, y_train = shuffle(X, y)
            while True:
              for off in range(0, len(X_train), self.batch_size):
                  bi += self.batch_size
                  if bi > self.max_iter:
                      print("Early stopping!", self.max_iter, "samples.")
                      self.save_val_plot()
                      return self

                  end = off + self.batch_size
                  batch_x, batch_y = X_train[off:end], y_train[off:end]
                  sess.run(self.train_op, feed_dict={self.X: batch_x, self.y: batch_y})
                  loss = sess.run(self.loss, feed_dict={self.X: batch_x, self.y: batch_y})

                  checks_since_prog += 1
                  if loss < self.best_loss:
                      self.best_loss = loss
                      checks_since_prog = 0
                      best_params_ = self.get_params()
                  # if checks_since_prog > 25:
                  #     print("Early stopping!")
                  #     return self        

                  if X_valid is not None and y_valid is not None:
                      valid_acc = self.score(X_valid, y_valid) 

                      self.v_scores.append({"Epoch": bi, "Validation": valid_acc})
                      msg = "Loss: %.4f; Validation accuracy: %s" % (loss, valid_acc)
                      self.log(msg)

        return self

    def save_model(self):
        with self.session.as_default() as sess:
            saver = tf.train.Saver()
            dt = datetime.now().strftime("%H-%M-%S")
            save_path = saver.save(sess, "./models/%s.ckpt" % dt)

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
        """
        Save validation plot as csv and 
        create log file to show params
        """
        df = pd.DataFrame(data=self.v_scores, columns=self.v_columns)
        now = datetime.now().strftime("%H-%M-%S")
        name = now + "no_aug_200k"
        fn = "./val_scores/" + name + ".csv"
        df.to_csv(fn, index=False)

default_activations = [
    "conv1",
    "avg_pool",
    "conv2",
    "avg_pool2",
    "conv3",
    "dense",
]

elu = {x: tf.nn.elu for x in default_activations}
selu = {x: tf.nn.selu for x in default_activations}
relu = {x: tf.nn.relu for x in default_activations}
tanh= {x: tf.nn.tanh for x in default_activations}

param_grid = {
    "aug__rotation": [[30], [60], [90]],
    "aug__shear": [[10], [20], [40]],
    "aug__rot_and_shear": [True, False],
    "aug__n_iter": [10],
    "clf__activations": [selu],
    "clf__learning_rate": [0.01],
    "clf__dropout_rate": [0.1],
    "clf__batch_size": [200],
}

da = DataAugment()
clf = CNNClassifier()

pipe = Pipeline(
    steps=[
        ("aug", da),
        ("clf", clf),
    ]
)

now = datetime.now().strftime("%H-%M-%S")
log_name = "./logs/cnn_log_" + now + ".log"
with open(log_name, "w") as log:
    log.write("Param grid:\n")
    for p in param_grid:
        log.write("%s: %s\n" % (p, param_grid[p]))
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=5)
    gs.fit(X_train, y_train)
    best_msg = "best score: %s, params: %s" % (gs.best_score_, gs.best_params_)
    print(best_msg)
    log.write(best_msg)
    
    test_msg = "Test accuracy: %s" % gs.best_estimator_.score(X_test, y_test)
    test_msg = "Test accuracy: %s" % pipe.score(X_test, y_test)
    print(test_msg)
    log.write(test_msg)
    
df = pd.DataFrame(gs.cv_results_)
df.to_csv("./gs_results/" + now + ".csv")

