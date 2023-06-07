# ！/usr/bin/python3
# -*- coding: utf-8 -*-
# @TIme 2023/6/6 18:00
# @Author: SAM
# @File: tuner.py
# @Email: SAM-Turentu@outlook.com
# @Desc: Keras Tuner


import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

from utils.utils import tools

# 超参数集
# 为机器学习应用选择正确的超参数集，这个过程称为 超参数调节 或 超调
# 超参数有两种：模型超参数，算法超参数

# 加载数据
(img_train, label_train), (img_test, label_test) = tools.load_fashion_data()
# keras.datasets.fashion_mnist.load_data()

img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0


# 定义超模型，有下面两种方法
#  1.使用模型构建工具
#  2.将 Keras Tuner api 的 HyperModel 类子类化
def model_builder(hp):
    """
     使用方法1
    :param hp:
    :return: 模型共建工具函数返回已编译的模型
    """
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))

    # 调整 第一个密集层的单元数 32-512 随机
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(10))

    # 优化器的学习率， 0.01,0.001,0.0001 随机一个
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),  # Adam 传统优化器，在m1 m2 芯片上运行慢
    #               loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])

    return model


# 实例化调节器并执行超调
# Keras Tuner 有四种调节器：RandomSearch  Hyperband  BayesianOptimization  Sklearn
#  实例化 Hyperband 调节器，必须制定超模型、要优化的 objective 和 要训练的最大周期数 max_epochs
tuner = kt.Hyperband(
    model_builder,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='my_dir',
    project_name='intro_to_kt'  # 目录中包含了超参数搜索期间每次实验运行的详细日志和检查点
)
# 如果重新运行超参数搜索，Keras Tuner 将使用日志中记录的现有状态继续搜索。若要停用，实例化调节器传递一个参数 overwrite=True


# Hyperband 调节算法使用自适应资源分配和早停法来快速手链到最高性能模型
#  该算法会将大量模型训练多个周期，并将性能最高的一半模型 送入 下一轮 训练

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# 运行超参数搜索， 除了回调，搜索方法的参数 和 tf.keras.model.fit 参数相同
tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# 获得最佳 超参数
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# 训练模型，从搜索中过的的超参数找到训练模型的最佳周期数
model = tuner.hypermodel.build(best_hps)
history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)

val_acc_pre_epoch = history.history['val_accuracy']
best_epoch = val_acc_pre_epoch.index(max(val_acc_pre_epoch)) + 1  # 最佳周期数
print(f'Best epoch: {best_epoch}')

# 重新实例化超模型 并使用最佳周期数进行训练
hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)

# 用测试数据评估超模型
eval_result = hypermodel.evaluate(img_test, label_test)
print('[test loss, test accuracy]: ', eval_result)
