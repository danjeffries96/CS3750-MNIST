import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class CNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, verbose=1, optimizer_class=tf.train.AdamOptimizer,
            learning_rate=0.001, batch_size=128, activation=tf.nn.tanh):
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
              for off in range(0, len(X_train), self.batch_size):
                  end = off + self.batch_size
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

