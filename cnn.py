import tensorflow as tf
from sklearn.utils import shuffle
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


X_inp = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
y_inp = tf.placeholder(tf.int32, (None))

conv1 = tf.layers.conv2d(X_inp, filters=6, kernel_size=5,  strides=1, padding="valid", activation=tf.nn.tanh) 

avg_pool = tf.layers.average_pooling2d(conv1, pool_size=2, strides=2)
avg_pool = tf.nn.tanh(avg_pool)

conv2 = tf.layers.conv2d(avg_pool, filters=16, kernel_size = 5, strides=1,
                       padding="valid", activation = tf.nn.tanh)

avg_pool2 = tf.layers.average_pooling2d(conv2, pool_size = 2, strides=2)
avg_pool2 = tf.nn.tanh(avg_pool2)

conv3 = tf.layers.conv2d(avg_pool2, filters=120, kernel_size = 5, strides=1,
                      padding="valid", activation = tf.nn.tanh)

# flatten for dense layers
flat = tf.reshape(conv3, [-1, 120])

dense = tf.layers.dense(inputs=flat, units=84, activation=tf.nn.tanh)

logits = tf.layers.dense(dense, units=10)

softmax = tf.nn.softmax(logits)
predict = tf.argmax(softmax, axis=1)

y_labels = tf.one_hot(y_inp, 10)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_labels)
loss = tf.reduce_mean(cross_entropy)
opt = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = opt.minimize(loss)

epochs = 10
N = len(X_train)
BATCH_SIZE = N // 10 

def evaluate(X_eval, y_eval):
    sess = tf.get_default_session()
    out = sess.run(predict, feed_dict ={X_inp: X_eval, y_inp: y_eval})
    return np.mean(out == y_eval)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for i in range(epochs):
      X_train, y_train = shuffle(X_train, y_train)
      for off in range(0, N, BATCH_SIZE):
          end = off + BATCH_SIZE
          batch_x, batch_y = X_train[off:end], y_train[off:end]
          sess.run(train_op, feed_dict={X_inp: batch_x, y_inp : batch_y})
      valid_acc = evaluate(X_validation, y_validation) 
      print("Validation accuracy: " + str(valid_acc))
  test_acc = evaluate(X_test, y_test) 
  print("Test accuracy: " + str(test_acc))


