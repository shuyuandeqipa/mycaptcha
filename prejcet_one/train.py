"""
训练代码
"""
import tensorflow as tf
from project_data_one.cfg import MAX_CAPTCHA, CHAR_SET_LEN, tb_log_path, save_model
from project_data_one.cnn_sys import crack_captcha_cnn, Y, keep_prob, X
from project_data_one.get_data import get_next_batch_image_and_name
def train_crack_captcha_cnn():
    """
    训练模型
    :return:
    """
    output = crack_captcha_cnn() # 需要改
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])  # 36行，4列
    label = tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN])

    max_idx_p = tf.argmax(predict, 2)  # shape:batch_size,4,nb_cls
    max_idx_l = tf.argmax(label, 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)

    with tf.name_scope('my_monitor'):
       loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    tf.summary.scalar('my_loss', loss)

    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰减
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    with tf.name_scope('my_monitor'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('my_accuracy', accuracy)

    saver = tf.train.Saver()  # 将训练过程进行保存

    sess = tf.InteractiveSession(
        config=tf.ConfigProto(
            log_device_placement=False
            # 或许这里需要设置可以动态分配gpu内存
        )
    )

    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tb_log_path, sess.graph)

    accuracy_history = 0.0  # 保留前一次的训练精度
    accuracy_history_times = 0  # 差不多训练精度的次数
    step = 0
    while True:
        batch_x, batch_y = get_next_batch_image_and_name(batch_size=64,flag=0)  # 64
        _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.95})#减弱过拟合
        print(step, 'loss:\t', loss_)
        step += 1
        # 每500步保存一次实验结果
        if step % 500 == 0:
            saver.save(sess, save_model, global_step=step)
        # 在测试集上计算精度
        if step % 50 != 0:
            continue
        # 每50 step计算一次准确率，使用测试数据集中的图片
        batch_x_test, batch_y_test = get_next_batch_image_and_name(batch_size=128,flag=1)  # 加载测试数据
        acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
        print(step, 'acc---------------------------------\t', acc)

        # 终止条件
        if acc > 0.995 and step>2000:
           break
        # if step>10100 and acc<0.650:
        #     break
        #    # 终止条件
        #    if acc > 0.800 and step > 1500:
        #        break
        #    if step > 10100 and acc < 0.650:
        #        break
        #    if acc > 0.85:  # 用来检查是否真的是过拟合
        #        break
        #    if step > 3000:
        #        break
           # 用来做为早停止的代码
           accuracy_history = acc
           if abs(acc - accuracy_history) < 0.005:
               accuracy_history_times += 1
           else:
               accuracy_history_times = 0
           if accuracy_history_times > 100:
               break

        # 启用监控 tensor board
        summary = sess.run(merged, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
        writer.add_summary(summary, step)

if __name__ == '__main__':
    train_crack_captcha_cnn()
    print('end')
    pass
