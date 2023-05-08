# ！/usr/bin/python3
# -*- coding: utf-8 -*-
# @TIme 2023/5/7 14:26
# @Author: SAM
# @File: tfds.py
# @Email: SAM-Turentu@outlook.com
# @Desc: 电影评论文本


import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt

from utils.join_path import hub_model_path, tf_datasets


def main():
    path = tf_datasets.tf_datasets
    train_data, validation_data, test_data = tfds.load(
        name='imdb_reviews',
        split=('train[:60%]', 'train[60%:]', 'test'),
        data_dir=path,
        as_supervised=True
    )
    train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

    # region 构建模型
    #  # embedding = 'https://tfhub.dev/google/nnlm-en-dim50/2'  # tf官方教程中的下载地址错误，下面是正确的
    # embedding = 'https://hub.tensorflow.google.cn/google/nnlm-en-dim50/2' # 正确地址

    # 嵌入 embedding 输出的形状都是：生成的维度：(num_examples, embedding_dimension)
    # embedding = 'nnlm-en-dim50_2'  # val_accuracy ≈ 0.87
    # embedding = 'nnlm-en-dim128_2'  # val_accuracy ≈ 0.87
    # embedding = 'nnlm-en-dim128-with-normalization_2'  # 训练的准确率会提升，但需要更多时间训练 val_accuracy ≈ 0.88
    # embedding = 'universal-sentence-encoder-4'  # val_accuracy ≈ 0.88  0.89

    embedding = hub_model_path.model_path('nnlm-en-dim50_2')

    hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
    hub_layer(train_examples_batch[:3])

    model = tf.keras.Sequential()
    model.add(hub_layer)  #
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.summary()  # 查看 model 摘要
    # endregion

    # 编译
    # 这是一个二元分类问题，并且模型输出logit（具有线性激活的单一单元层），因此选择 BinaryCrossentropy 损失函数，更适合处理概率问题
    # 这个损失函数也可以 MeanSquaredError
    model.compile(optimizer='adam',  # 优化器
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),  # 损失函数
                  metrics=['accuracy'])

    # 训练
    history = model.fit(train_data.shuffle(10000).batch(512),
                        epochs=10,
                        validation_data=validation_data.batch(512),
                        verbose=1)

    history_dict = history.history
    loss = history_dict['loss']  # 损失值
    accuracy = history_dict['accuracy']  # 准确率
    val_loss = history_dict['val_loss']
    val_accuracy = history_dict['val_accuracy']

    epochs = range(1, len(accuracy) + 1)

    # 损失值图
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()  # 清楚数字

    # 准确率图
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')  # bo 蓝色圆点
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')  # b 蓝色实线
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    results = model.evaluate(test_data.batch(512), verbose=2)
    for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))


if __name__ == '__main__':
    main()
