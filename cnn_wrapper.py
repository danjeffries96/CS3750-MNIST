import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


X = np.load("x_mnist1000.npy")
X = X.reshape((-1, 28, 28, 1))
y = np.load("y_mnist1000.npy")

np.random.seed(1)

indices = np.random.permutation(len(X))
train_indices = indices[:800]
valid_ind = indices[800:900]
test_ind = indices[900:]

X_train = X[train_indices]
y_train = y[train_indices]

X_validation = X[valid_ind]
y_validation = y[valid_ind]

X_test = X[test_ind]
y_test = y[test_ind]

# pad for lenet
pad_dims = ((0, 0), (2, 2), (2, 2), (0, 0))
X_train = np.pad(X_train, pad_dims, "constant")
X_validation = np.pad(X_validation, pad_dims, "constant")
X_test = np.pad(X_test, pad_dims, "constant")


N = len(X_train)
BATCH_SIZE = N // 10 

class CNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, verbose=1, optimizer_class=tf.train.AdamOptimizer,
            learning_rate=0.001, batch_size=BATCH_SIZE, activation=tf.nn.tanh):
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation=activation

        self.verbose = verbose

        self._params = [
            "optimizer_class",
            "learning_rate",
            "batch_size",
            "activation",
        ]

        self.session = None
    
    def _build_graph(self):
        self.X = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
        self.y = tf.placeholder(tf.int32, (None))

        conv1 = tf.layers.conv2d(self.X, filters=6, kernel_size=5,  
                strides=1, padding="valid", activation=self.activation)

        avg_pool = tf.layers.average_pooling2d(conv1, pool_size=2, strides=2)
        avg_pool = self.activation(avg_pool)

        conv2 = tf.layers.conv2d(avg_pool, filters=16, kernel_size=5, strides=1,
                               padding="valid", activation=self.activation)

        avg_pool2 = tf.layers.average_pooling2d(conv2, pool_size = 2, strides=2)
        avg_pool2 = self.activation(avg_pool2)

        conv3 = tf.layers.conv2d(avg_pool2, filters=120, kernel_size=5, strides=1,
                              padding="valid", activation=self.activation)

        # flatten for dense layers
        flat = tf.reshape(conv3, [-1, 120])

        dense = tf.layers.dense(inputs=flat, units=84, activation=self.activation)

        logits = tf.layers.dense(dense, units=10)

        softmax = tf.nn.softmax(logits)
        self._predict = tf.argmax(softmax, axis=1)

        y_labels = tf.one_hot(self.y, 10)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_labels)
        loss = tf.reduce_mean(cross_entropy)
        opt = tf.train.AdamOptimizer(learning_rate=0.001)
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

    def fit(self, X, y, n_epochs=5, X_valid=None, y_valid=None):
        if self.session:
            self.session.close()

        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_graph()

        # print params
        self.log(self.get_params(), level=1)

        self.session = tf.Session(graph=self.graph)
        with self.session.as_default() as sess:
            self.init.run()
        
            for i in range(n_epochs):
              if i % (n_epochs / 10) == 0:
                  self.log("Epoch:", i, level=2)
              X_train, y_train = shuffle(X, y)
              for off in range(0, N, BATCH_SIZE):
                  end = off + BATCH_SIZE
                  batch_x, batch_y = X_train[off:end], y_train[off:end]
                  sess.run(self.train_op, feed_dict={self.X: batch_x, self.y: batch_y})
              valid_acc = self.score(X_valid, y_valid) 
              self.log("Validation accuracy: " + str(valid_acc))
        
        return self
    
    def predict(self, X):
        with self.session.as_default() as sess:
            prediction = sess.run(self._predict, feed_dict={self.X: X})
            return prediction

    def score(self, X_eval, y_eval):
        with self.session.as_default() as sess:
            out = sess.run(self._predict, feed_dict={self.X: X_eval, self.y: y_eval})
            return np.mean(out == y_eval)


clf = CNNClassifier()

param_grid = {
    # "optimizer_class": [tf.train.AdagradOptimizer, tf.train.MomentumOptimizer],
    "verbose":  [2],
    "activation": [tf.nn.tanh, tf.nn.relu, tf.nn.elu],
}

gs = GridSearchCV(clf, param_grid=param_grid, n_jobs=-1)
gs.fit(X_train, y_train, n_epochs=10, X_valid=X_validation, y_valid=y_validation)
print("best score, params:", gs.best_score_, ",", gs.best_params_)
