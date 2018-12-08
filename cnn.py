import tensorflow as tf
from tensorflow.contrib.layers import flatten
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

# pad for lenet
pad_dims = ((0, 0), (2, 2), (2, 2), (0, 0))
X_train = np.pad(X_train, pad_dims, "constant")

# Conv2d
# input: A Tensor. Must be one of the following types: half, bfloat16, float32, float64. A 4-D tensor. The dimension order is interpreted according to the value of data_format, see below for details.
# filter: A Tensor. Must have the same type as input. 
#         A 4-D tensor of shape 
#         [filter_height, filter_width, in_channels, out_channels]
# strides: A list of ints. 1-D tensor of length 4. The stride of the sliding window for each dimension of input. The dimension order is determined by the value of data_format, see below for details.
# padding: A string from: "SAME", "VALID". The type of padding algorithm to use.

X_inp = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
conv1_fmaps = np.zeros(shape=(5, 5, 1, 6))
conv1 = tf.nn.conv2d(X_inp, conv1_fmaps, strides=[1, 1, 1, 1], padding="VALID")
conv1 = tf.nn.tanh(conv1)

avg_pool = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding="VALID")
avg_pool = tf.nn.tanh(avg_pool)

conv2_fmaps = np.zeros(shape=(10, 10, 6, 16))
conv2 = tf.nn.conv2d(avg_pool, conv2_fmaps, strides=[1, 1, 1, 1],
                       padding="VALID")
conv2 = tf.nn.tanh(conv2)

avg_pool2 = tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                            padding="VALID")
avg_pool2 = tf.nn.tanh(avg_pool2)

conv3_fmaps = np.zeros(shape=(1, 1, 16, 120))
conv3 = tf.nn.conv2d(avg_pool2, conv3_fmaps, strides=[1, 1, 1, 1],
                      padding="VALID")
conv3 = tf.nn.tanh(conv3)

# flatten for dense layers
fl = flatten(conv3)

dense = tf.layers.dense(inputs=fl, units=84, activation=tf.nn.tanh)

logits = tf.layers.dense(dense, units=10)

softmax = tf.nn.softmax(logits)
final = tf.argmax(softmax, axis=1)
  
y_labels = tf.one_hot(y_train, 10)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_labels)
loss = tf.reduce_mean(cross_entropy)
opt = tf.train.AdamOptimizer(learning_rate=0.1)
train_op = opt.minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(train_op, feed_dict={X_inp: X_train})

  output = sess.run(final, feed_dict={X_inp: X_train})
  print(output)

  print(np.mean(output == y_train))
   

