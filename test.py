import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# array2freqs = np.arange(1, 10)
# repeat_array = np.repeat(array2freqs, 2)
# array11 = np.concatenate(([1, 2], repeat_array), 0)
# print(array2freqs)
# print(repeat_array)
# print(array11)

# base_array = np.repeat(np.arange(1, 10), 2)
# base_freqs = np.concatenate(([1, 2], base_array), 0)
# freq = np.concatenate((base_freqs, 2*base_freqs, 4*base_freqs, 8*base_freqs), axis=0)
# print(freq)

# base_freqs = np.arange(1, 11)
# base_freqs = np.concatenate(([1, 2, 3, 4, 5], base_freqs), 0)
# high_freqs = np.arange(91, 100)
# freq = np.concatenate(([1], base_freqs, 2*base_freqs, 4*base_freqs, 6*base_freqs, 8*base_freqs, 10*base_freqs, high_freqs), axis=0)
# print(freq)
# print(len(freq))


def smrelu(x):
    # out = 0.4*tf.nn.relu(1 - x) * tf.nn.relu(1 + x) * tf.cos(np.pi * x)
    # out = 0.21*tf.nn.relu(1-x)*tf.nn.relu(1+x)*tf.sin(2*np.pi*x)
    # out = tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.sin(2*np.pi*x)
    # out = 1.5*tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.cos(np.pi*x)
    # out = tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.sin(2*np.pi*tf.abs(x))
    out = tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.sin(2*np.pi*x)
    # out = tf.nn.relu(1 - x) * tf.nn.relu(x+0.5) * tf.sin(2 * np.pi * x)
    # out = tf.nn.relu(1 - tf.abs(x)) * tf.nn.relu(tf.abs(x)) * tf.sin(2 * np.pi * tf.abs(x))
    # out = tf.nn.relu(1 - tf.abs(x)) * tf.nn.relu(tf.abs(x)) * tf.sin(np.pi * tf.abs(x))
    return out


def csrelu(x):
    out = 1.5*tf.nn.relu(1 - tf.abs(x)) * tf.nn.relu(tf.abs(x)) * tf.cos(np.pi * x)
    # out = 1.5*tf.nn.relu(1 - x) * tf.nn.relu(x) * tf.cos(np.pi * x)
    return out


def test():
    with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
        X_it = tf.placeholder(tf.float32, name='X_it', shape=[None, 1])
        Ysin = smrelu(X_it)
        Ycos = csrelu(X_it)

    test_batch_size = 1000
    test_x_bach = np.reshape(np.linspace(-1.5, 1.5, num=test_batch_size), [-1, 1])
    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        ysin,ycos = sess.run([Ysin, Ycos], feed_dict={X_it: test_x_bach})

        figsin = plt.figure()
        plt.plot(test_x_bach, ysin, 'b-.', label='ysin')
        plt.show()

        figcos = plt.figure()
        plt.plot(test_x_bach, ycos, 'r-.', label='ycos')
        plt.show()


if __name__ == "__main__":
    test()