# ！/usr/bin/python3
# -*- coding: utf-8 -*-
# @TIme 2023/5/3 17:28
# @Author: SAM
# @File: helloworld.py
# @Email: SAM-Turentu@outlook.com
# @Desc:


import tensorflow as tf

from utils import join_path


def main():
    mnist = tf.keras.datasets.mnist

    # 每个图转换为 28 * 28 的二维数组，记录每个像素点的(只有黑色) black rgb值

    file_path = join_path.keras_data_path.helloworld_keras_mnist_data_path  # 已经下载的数据集

    # 加载 mnist 数据集
    (x_train, y_train), (x_test, y_test) = mnist.load_data(file_path)  # 没有数据集时，自动下载
    x_train, x_test = x_train / 255, x_test / 255  # 每个单元格 除以 255， 0-1范围内

    # 通过堆叠层构建模型
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    predictions = model(x_train[:1]).numpy()  # numpy.ndarray 类型
    predictions = tf.nn.softmax(predictions).numpy()  # softmax函数将这些 logits 转换为每个类的概率
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # 损失函数，为每个样本返回一个标量损失
    loss_fn(y_train[:1], predictions).numpy()

    # 配置和编译模型
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])  # accuracy 模型评估的指标
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test, verbose=2)  # 检查模型性能
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    probability_model(x_test[:5])


if __name__ == '__main__':
    main()
