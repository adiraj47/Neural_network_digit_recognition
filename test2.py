import tensorflow as tf
number = tf.Variable(10, tf.int16)
string = tf.Variable("This is a string", tf.string)
float_no = tf.Variable(6.273, tf.float32)
print("{0} \n {1}\n {2}".format(number, string, float_no))