import tensorflow as tf
sess = tf.InteractiveSession()
labels = [[1.0,2.0,3.0],[4.0,5.0,6.0]]
# x = tf.expand_dims(labels, 0)
# print(sess.run(x))
# x = tf.expand_dims(labels, 1)
# print(sess.run(x))
# x = tf.expand_dims(labels, 2)
# print(sess.run(x))
z = tf.expand_dims(labels, 2)  # 把二维的张量转为三维的张量，在数据上相当于对矩阵做了一次转置
corr = tf.reduce_mean(tf.matmul(z, tf.transpose(z, perm=[0, 2, 1])), 0)
print('z=',sess.run(z))
print('zT=',sess.run(tf.transpose(z, perm=[0, 2, 1])))
print('matmul=',sess.run(tf.matmul(z, tf.transpose(z, perm=[0, 2, 1]))))
print('corr=')
print(sess.run(corr))
