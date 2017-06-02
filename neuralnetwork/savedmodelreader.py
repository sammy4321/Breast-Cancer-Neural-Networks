import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing,cross_validation

attributes=pd.read_csv('cancer_attributes.txt')
classes=pd.read_csv('cancer_classes.txt')

attributes=preprocessing.scale(attributes)
x_train,x_test,y_train,y_test=cross_validation.train_test_split(attributes,classes,test_size=0.2)

with tf.Session() as sess:
	
	
	saver = tf.train.import_meta_graph('my_test_model-1000.meta')
	saver.restore(sess,tf.train.latest_checkpoint('./'))
	
	graph = tf.get_default_graph()
	x=graph.get_tensor_by_name("x:0")
	y=graph.get_tensor_by_name("y:0")
	accuracy=graph.get_tensor_by_name("accuracy:0")
	print('Accuracy:',accuracy.eval({x:x_test, y:y_test}))
