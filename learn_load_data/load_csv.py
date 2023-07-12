# ！/usr/bin/python3
# -*- coding: utf-8 -*-
# @TIme 2023/6/7 16:17
# @Author: SAM
# @File: load_csv.py
# @Email: SAM-Turentu@outlook.com
# @Desc: 用 tf.data 加载 CSV 数据


import functools
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from utils import join_path

np.set_printoptions(precision=3, suppress=True)

# 泰坦尼克号乘客的数据
# TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
# TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"
# train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
# test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

train_path = join_path.tf_datasets.titanic_file_path('train')
test_path = join_path.tf_datasets.titanic_file_path('eval')

# region 加载数据

CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']

LABEL_COLUMN = 'survived'
LABELS = [0, 1]


# 从文件中读取 CSV 数据并且创建 dataset
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name=LABEL_COLUMN,
        na_value='?',
        num_epochs=1,
        ignore_errors=True
    )
    return dataset


raw_train_data = get_dataset(train_path)
raw_test_data = get_dataset(test_path)

# 第一个批次数据
examples, labels = next(iter(raw_train_data))
print('EXAMPLE: \n', examples, '\n')
print('LABELS: \n', labels, '\n')

# endregion


# region 数据预处理

# region 分类数据

CATEGORIES = {
    'sex': ['male', 'female'],
    'class': ['First', 'Second', 'Third'],
    'deck': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town': ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone': ['y', 'n']
}

# 使用 tf.feature_column API 创建一个 tf.feature_column.indicator_column 集合
#   每个 tf.feature_column.indicator_column 对应一个分类的列
categorical_columns = []
for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))


# endregion


# region 连续数据
def process_continuous_data(mean, data):
    # 标准化数据
    data = tf.cast(data, tf.float32) * 1 / (2 * mean)
    return tf.reshape(data, [-1, 1])


MEANS = {
    'age': 29.631308,
    'n_siblings_spouses': 0.545455,
    'parch': 0.379585,
    'fare': 34.385399
}

numerical_columns = []
for feature in MEANS.keys():
    # functools.partial 由使用每个列的均值进行标准化的函数构成
    num_col = tf.feature_column.numeric_column(feature,
                                               normalizer_fn=functools.partial(process_continuous_data, MEANS[feature]))
    numerical_columns.append(num_col)

# endregion


# region 创建预处理层

# 将这两个特征列的集合相加，并且传给 tf.keras.layers.DenseFeatures 从而创建一个进行预处理的输入层
preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numerical_columns)

# endregion


# endregion


# region 构建模型
# 从 preprocessing_layer 开始构建模型

model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# endregion


# region 训练、评估和预测
train_data = raw_train_data.shuffle(500)
test_data = raw_test_data
model.fit(train_data, epochs=20)

# 使用测试集 test_data 检查准确性
test_loss, test_accuracy = model.evaluate(test_data)
print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

# 使用 tf.keras.Model.predict 推断一个批次或多个批次的标签
predictions = model.predict(test_data)
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
    print("Predicted survival: {:.2%}".format(prediction[0]),
          " | Actual outcome: ",
          ("SURVIVED" if bool(survived) else "DIED"))

# endregion
