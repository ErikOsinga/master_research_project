from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

'''
Following the introduction from
https://www.tensorflow.org/guide/low_level_intro

'''


# A tensors rank is the number of dimensions, and its shape is its shape.

# A tensorflow program consists of two discrete sections:
#   1. Building the computational graph (tf.Graph)
#   2. Running the computational graph (tf.Session)

# A graph is a series of TensorFlow operations arranged into a graph
# The graph is composed of two types of objects:
#   1. tf.Operation ("ops"): The nodes of the graph. Operations describe
#   calculations that consume and produce tensors

# tf.Tensor: The edges in the graph. These represent values that will flow
# through the graph. Most tensorflow functions return tf.Tensors

# ! tf.Tensors do not have values, they are just handles to elements in the
# computations graph

tf.reset_default_graph() # start fresh

def part1():
	# The most basic computational graph:
	a = tf.constant(3.0, dtype=tf.float32)
	b = tf.constant(4.0)
	total = a+b
	print(a)
	print(b)
	print(total)
	# Each computation in the graph is given a unique name.
	# Tensors are named after the operation that produces them followed by an index

	# Tensorboard for visualizing a computation graph:
	writer = tf.summary.FileWriter('.')
	writer.add_graph(tf.get_default_graph())


	# To evaluate tensors, instantiate a tf.Session object, informally known
	# as a session. A session is like the python executable of the .py file
	sess = tf.Session()
	print(sess.run(total)) # evaluates the 'total' object we have created above

	# can also return stuff in a dictionary
	print(sess.run({'ab':(a, b), 'total':total}))

	# some tensorflow functions return tf.Operations instead of tf.Tensors
	# the result of calling run on an Operation is None. You run an operation
	# to cause a side-effect, not to retrieve a value. Examples include 
	# initialization and training ops (tf.Operations) demonstrated later

	# ===========================================
	# a graph can be parameterized to accept external inputs, known as 
	# placeholders. A placeholder is a promise to provide a value later
	# like a function argument.

	x = tf.placeholder(tf.float32)
	y = tf.placeholder(tf.float32)
	z = x + y

	# Can evaluate that graph with multiple inputs by using the feed_dict
	# argument of tf.Session.run
	print(sess.run(z, feed_dict={x: 3, y: 4.5}))
	print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))

	# ===========================================
	# Placeholders work for simple experiments, but tf.data are the preferred
	# method of streaming data into a model
	# To get a runnable tf.Tensor from a Dataset you must first convert it to
	# a tf.data.Iterator and then call the iterators tf.data.Iterator.get_next
	# method.

	# The simplest way to create an Iterator is with the
	# tf.data.Dataset.make_one_shot_iterator method.

	my_data = [
	    [0, 1,],
	    [2, 3,],
	    [4, 5,],
	    [6, 7,],
	]
	slices = tf.data.Dataset.from_tensor_slices(my_data)
	next_item = slices.make_one_shot_iterator().get_next()
	while True:
	  try:
	    print(sess.run(next_item))
	  except tf.errors.OutOfRangeError:
	    break

	# If the dataset depends on stateful operations (?)
	# you may need to initialize the iterator before using it, as shown below
	r = tf.random_normal([10,3])
	dataset = tf.data.Dataset.from_tensor_slices(r)
	iterator = dataset.make_initializable_iterator()
	next_row = iterator.get_next()

	sess.run(iterator.initializer)
	while True:
	  try:
	    print(sess.run(next_row))
	  except tf.errors.OutOfRangeError:
	    break

# ===============================================
# Layers
# a trainable model must modify the valeus in the graph to get new outputs
# with the same input. tf.layers are the preferred way to add trainable params
# to a graph.

'''
Layers package together both the variables and the operations that act on them. 
For example a densely-connected layer performs a weighted sum across all inputs 
for each output and applies an optional activation function. The connection 
weights and biases are managed by the layer object.
'''
def layers():
	sess = tf.Session()
	# Create a dense layer that takes a batch of input vectors, and produces
	# a single output value for each. To apply a layer to an input, call the layer
	# as if it were a function. E.g.,
	x = tf.placeholder(tf.float32, shape=[None, 3])
	linear_model = tf.layers.Dense(units=1)
	y = linear_model(x)
	# The layer contains values that must be initialized
	init = tf.global_variables_initializer() # this initializes all variables in
											# a tensorflow graph
	sess.run(init)										
	print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6]]})) # print y give x

	# for each layer class (like Dense) tf also provides a shortcut function (dense)
	# The only difference is that the shortcut function versions create and run the
	# layer in a single call. For example, the following code is equivalent to the
	# earlier version
	x = tf.placeholder(tf.float32, shape=[None,3])
	y = tf.layers.dense(x,units=1)
	init = tf.global_variables_initializer()
	sess.run(init)
	print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6]]})) # print y give x
	# While convenient, this approach allows no access to the tf.layers.Layer object. 
	# This makes introspection and debugging more difficult, and layer reuse impossible

# ===============================================
# Feature columns
# The easiest way to experiment with feature columns is using the 
# tf.feature_column.input_layer function. This function only accepts
# dense columns as inputs. So to view the result of a categorical column
# you must wrap it in an tf.feature_column.indicator_column.

def feature_columns():
	features = {
	    'sales' : [[5], [10], [8], [9]],
	    'department': ['sports', 'sports', 'gardening', 'gardening']}
	department_column = tf.feature_column.categorical_column_with_vocabulary_list(
	        'department', ['sports', 'gardening'])
	department_column = tf.feature_column.indicator_column(department_column)
	columns = [
	    tf.feature_column.numeric_column('sales'),
	    department_column
	]
	inputs = tf.feature_column.input_layer(features, columns)
	# Running the inputs tensor will parse the features into a batch of vectors
	# Feature columns can have internal state, like layers, so they often need
	# to be initialzed. Categorical columns use tf.contrib.lookup internally
	# and these require a seperate initialization op, tf.tables_initializer

	var_init = tf.global_variables_initializer()
	table_init = tf.tables_initializer()
	sess = tf.Session()
	sess.run((var_init, table_init))
	print(sess.run(inputs))

# ==================================================
# Training
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# print (sess.run(y_pred))
# Is very wrong, thus we need to train it. Need to define the loss
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
print (sess.run(loss))

# writer = tf.summary.FileWriter('.')
# writer.add_graph(tf.get_default_graph())

# Then we train it. With an optimizer --> gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(100):
  _, loss_value = sess.run((train, loss))
  print(loss_value)
# train is an op, not a tensor, it doesnt return a value when run.
# to see the progression, we run the loss tensor at the same time,
# producing the output

print (sess.run(y_pred))