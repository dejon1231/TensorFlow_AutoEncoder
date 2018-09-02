import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)

x_test = mnist.test.images
with tf.Session() as sess:
	saver = tf.train.import_meta_graph('./model.ckpt.meta')
	saver.restore(sess, './model.ckpt')
	input = tf.get_default_graph().get_tensor_by_name('input:0')
	prediction = tf.get_default_graph().get_tensor_by_name('output:0')
	fig = plt.figure()
	for i in range(10):
		index = np.random.random_integers(0,5000)
		pred = sess.run(prediction, feed_dict={input:x_test[index,:].reshape(1,784)})

		
		plt.subplot('211')
		plt.imshow(x_test[index,:].reshape(28,28), cmap=plt.cm.gray)
		plt.subplot('212')
		plt.imshow(pred.reshape(28,28), cmap=plt.cm.gray)
		plt.pause(0.5)