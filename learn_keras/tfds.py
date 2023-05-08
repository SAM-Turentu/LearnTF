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

from utils.utils import tools


def main():
    path = tools.tf_datasets
    train_data, validation_data, test_data = tfds.load(
        name='imdb_reviews',
        split=('train[:60%]', 'train[60%:]', 'test'),
        data_dir=path,
        as_supervised=True
    )
    train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

    # 构建模型
    embedding = 'https://tfhub.dev/google/nnlm-en-dim50/2'
    hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
    hub_layer(train_examples_batch[:3])

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.summary()  # 查看 model 摘要


if __name__ == '__main__':
    main()
