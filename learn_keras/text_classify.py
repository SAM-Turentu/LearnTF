# ！/usr/bin/python3
# -*- coding: utf-8 -*-
# @TIme 2023/5/7 09:24
# @Author: SAM
# @File: text_classify.py
# @Email: SAM-Turentu@outlook.com
# @Desc: 电影评论文本分类（IMDB：Internet movie database）


import tensorflow as tf
from matplotlib import pyplot as plt

from utils.utils import tools


def main():
    imdb = tf.keras.datasets.imdb
    # cifar10 = tf.keras.datasets.cifar10
    # boston_housing = tf.keras.datasets.boston_housing
    # cifar100 = tf.keras.datasets.cifar100
    # reuters = tf.keras.datasets.reuters

    imdb_path = tools.imdb_data_path
    (train_data, train_labels), (test_data, test_tabels) = imdb.load_data(imdb_path)
    # train_data[0]  单词已被转换为数字
    # train_labels 1：积极评论，0：消极评论

    # region 将数字转换为单词
    reuters_word_index_path = tools.imdb_word_index_path
    word_index = imdb.get_word_index(reuters_word_index_path)  # 加载 单词- 数字 对应的json文件

    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1  # 评论开始
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    reverse_dict = {v: k for k, v in word_index.items()}  # 反转字典键值对
    train_data_0_to_word = " ".join([reverse_dict.get(i, "?") for i in train_data[0]])  # 将数字转换文本
    # endregion

    print(train_data_0_to_word)

    # 长度标准化
    train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                               maxlen=256,  # 截断相同长度，默认去掉前面的 truncating="pre"
                                                               padding="post",  # post 在后面补充value， pre 在前面补充 value
                                                               value=word_index['<PAD>'])
    test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data,
                                                              maxlen=256,
                                                              padding="post",
                                                              value=word_index['<PAD>'])

    # region 构建模型
    # 输入形状是用于电影评论的词汇数目（1w词）
    vocab_size = 10000

    model = tf.keras.Sequential()

    # 第一层 嵌入 Embedding层，该层采用整数编码的词汇表，并查找每个词索引的嵌入向量 embedding vector
    # 这些向量是通过模型训练学习得到的，向量向输出数组增加了一个维度，维度：batch，sequence，embedding
    model.add(tf.keras.layers.Embedding(vocab_size, 16))

    # 通过对序列维度求平均值来为每个样本返回一个定长输出向量
    model.add(tf.keras.layers.GlobalAveragePooling1D())

    # 该定长输出向量通过一个有16个隐层但愿的全连接 Dense 层传输
    model.add(tf.keras.layers.Dense(16, activation='relu'))

    # 最后一层与单个输出节点密集连接。Sigmoid 激活函数，函数值0～1之间浮点数，表示概率或置信度
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    # endregion

    # 编译模型
    model.compile(optimizer='adam',  # 优化器
                  # 损失函数  binary_crossentropy 更适合处理概率（能够度量概率分布之间的"距离"）度量 ground-truth 分布与预测值之间的"距离"
                  loss='binary_crossentropy',
                  metrics=['accuracy'])  # 指标

    # 创建一个验证集，用训练数据的一半训练，然后还有一半用于开发调整模型，最后用测试数据评估准确率
    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]

    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    # 馈送数据，以512个样本迭代40个epoch来训练模型
    model.fit(partial_x_train, partial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
    # model.fit(partial_x_train, partial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

    history_dict = model.history.history  # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

    # region 准确率，损失值画图
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
    # endregion

    # 20个 epoch 之后，loss，acc 达到峰值，这是过拟合的一个实例：模型在训练数据上的表现比在以前从未见过的数据上的表现都要好
    # 模型过度优化并学习特定与训练数据的表示，而不能够泛化到测试数据
    # 需要停止训练，避免过拟合

    # 评估模型准确率
    results = model.evaluate(test_data, test_tabels, verbose=2)


if __name__ == '__main__':
    main()
