from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

def Weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

def biases(shape):
	return tf.Variable(tf.constant(0.1,shape=shape))

xs = tf.placeholder(tf.float32, [None,784], name='input')
learning_rate = tf.placeholder(tf.float32)


# layer1
W1 = Weights([784,300])
b1 = biases([300])
h_layer1 = tf.matmul(xs,W1)+b1
h_layer1_relu = tf.nn.relu(h_layer1)


# layer2
W2 = Weights([300,100])
b2 = biases([100])
h_layer2 = tf.matmul(h_layer1_relu,W2)+b2
h_layer2_relu = tf.nn.relu(h_layer2)

# layer3
W3 = Weights([100,50])
b3 = biases([50])
h_layer3 = tf.matmul(h_layer2_relu,W3)+b3
h_layer3_relu = tf.nn.relu(h_layer3)

# encoder
W4 = Weights([50,10])
b4 = biases([10])
h_layer4 = tf.matmul(h_layer3_relu,W4)+b4
h_layer4_relu = tf.nn.relu(h_layer4)

# layer3
W5 = Weights([10,50])
b5 = biases([50])
h_layer5 = tf.matmul(h_layer4_relu,W5)+b5
h_layer5_relu = tf.nn.relu(h_layer5)

# layer2
W6 = Weights([50,100])
b6 = biases([100])
h_layer6 = tf.matmul(h_layer5_relu,W6)+b6
h_layer6_relu = tf.nn.relu(h_layer6)

# layer1
W7 = Weights([100,300])
b7 = biases([300])
h_layer7 = tf.matmul(h_layer6_relu,W7)+b7
h_layer7_relu = tf.nn.relu(h_layer7)

# output layer
W8 = Weights([300,784])
b8 = biases([784])
h_layer8 = tf.matmul(h_layer7_relu,W8)+b8
h_layer8_relu = tf.nn.relu(h_layer8, name='output')

loss = tf.reduce_mean(tf.square(xs-h_layer8_relu))
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

l = 0.01
total_step = 10000
for i in range(total_step):
	xs_train, xs_labels = mnist.train.next_batch(100)
	sess.run(train_step,feed_dict={xs:xs_train,learning_rate:l})
	if i%50 == 0 :
		print(sess.run(loss, feed_dict={xs:xs_train,learning_rate:l}))
	
	if i+total_step/2 == total_step:
		print('lower down learning_rate')
		l = 0.001
		
saver.save(sess,'./model.ckpt')
'''
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

fig = plt.figure()

count = 1
for i in range(1,10):
	plt.subplot(2,10,count)
	plt.imshow(mnist.test.images[i].reshape([28,28]), cmap=plt.cm.gray)
	decoder = sess.run(h_layer8_relu,feed_dict = {xs:mnist.test.images[i].reshape([1,784])})
	plt.subplot(2,10,count+10)
	plt.imshow(decoder.reshape([28,28]), cmap=plt.cm.gray)
	count+=1
#plt.show()

fig2 = plt.figure()
encoder = sess.run(h_layer4_relu,feed_dict = {xs:mnist.test.images})
#pca = PCA(n_components=2)
#X = pca.fit_transform(encoder)
tsne = TSNE(n_components=2)
X = tsne.fit_transform(encoder)
Y = np.argmax(mnist.test.labels, axis=1)

plt.scatter(X[:,0] , X[:,1],c=Y)
plt.show()
'''