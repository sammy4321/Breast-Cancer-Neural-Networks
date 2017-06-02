import tensorflow as tf

sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('my_test_model-1000.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))
sess.run(tf.global_variables_initializer())

# Now, let's access and create placeholders variables and
# create feed-dict to feed new data

graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name("w1:0")
w2 = graph.get_tensor_by_name("w2:0")
w3 = graph.get_tensor_by_name("w3:0")
feed_dict ={w1:11,w2:4}

#Now, access the op that you want to run. 
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

print sess.run(w3,feed_dict)
print sess.run(op_to_restore,feed_dict)
#This will print 60 which is calculated 
#using new values of w1 and w2 and saved value of b1.