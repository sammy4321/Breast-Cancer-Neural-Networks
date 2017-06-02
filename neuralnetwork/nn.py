import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing,cross_validation

n_classes = 2

attributes=pd.read_csv('cancer_attributes.txt')
classes=pd.read_csv('cancer_classes.txt')

attributes=preprocessing.scale(attributes)

x_train,x_test,y_train,y_test=cross_validation.train_test_split(attributes,classes,test_size=0.2)

x = tf.placeholder('float', [None, 9],name='x')
y = tf.placeholder('float',name='y')

hidden1w=tf.Variable(tf.random_normal([9,500]))
hidden1b=tf.Variable(tf.random_normal([500]))

hidden2w=tf.Variable(tf.random_normal([500,500]))
hidden2b=tf.Variable(tf.random_normal([500]))

hidden3w=tf.Variable(tf.random_normal([500,500]))
hidden3b=tf.Variable(tf.random_normal([500]))

outputw=tf.Variable(tf.random_normal([500,2]))
outputb=tf.Variable(tf.random_normal([2]))

l1=tf.add(tf.matmul(x,hidden1w),hidden1b)
l1=tf.nn.relu(l1)

l2=tf.add(tf.matmul(l1,hidden2w),hidden2b)
l2=tf.nn.relu(l2)

l3=tf.add(tf.matmul(l2,hidden3w),hidden3b)
l3=tf.nn.relu(l3)

output=tf.add(tf.matmul(l3,outputw),outputb)

prediction=output

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

hm_epochs=100
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver=tf.train.Saver()
	for epoch in range(hm_epochs):
		epoch_loss=0
		epoch_x=x_train
		epoch_y=y_train
		_,c=sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y})
		epoch_loss+=c

		print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

	correct=tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

	accuracy=tf.reduce_mean(tf.cast(correct, 'float'),name="accuracy")
	saver.save(sess, 'my_test_model',global_step=1000)
	