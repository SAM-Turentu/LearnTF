# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: custom_callback
# CreateTime: 2023/6/30 8:45
# Summary: 编写回调函数


import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils import join_path


# 可以将回调函数的列表传递给以下模型方法
# keras.Model.fit()  # keras.Model.evaluate()  # keras.Model.predict()


# region 回调函数方法概述

# on_(train|test|predict)_begin(self, logs=None)  在 fit/evaluate/predict 开始时调用
# on_(train|test|predict)_end(self, logs=None)  在 fit/evaluate/predict 结束时调用

# on_(train|test|predict)_batch_begin(self, batch, logs=None)   on_train_batch_end(self, batch, log=None)
# 正好在训练/测试/预测期间处理批次 之前(结束时) 调用

# on_epoch_begin(self, epoch, logs=None)   on_epoch_end(self, epoch, logs=None)
# 在训练期间周期开始(后)时调用


# endregion


# region 基本示例：记录回调顺序

def get_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(1, input_dim=784))
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    return model


path = join_path.keras_data_path.helloworld_keras_mnist_data_path
path = path.replace('BaseLearnTF\\', '')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path)
x_train = x_train.reshape(-1, 784).astype('float32') / 255.
x_test = x_test.reshape(-1, 784).astype('float32') / 255.

x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:1000]
y_test = y_test[:1000]


class CustomCallback(keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        """
        1: fit
        """
        keys = list(logs.keys())
        print('Start training; got log keys: ', keys)

    def on_train_end(self, logs=None):
        """
        10: fit
        """
        keys = list(logs.keys())
        print('Stop training; got log keys: ', keys)

    def on_epoch_begin(self, epoch, logs=None):
        """
        2: fit
        """
        keys = list(logs.keys())
        print(f'Start epoch {epoch} of training; got log keys: {keys}')

    def on_epoch_end(self, epoch, logs=None):
        """
        9: fit
        """
        keys = list(logs.keys())
        print(f'End epoch {epoch} of training; got log keys: {keys}')

    def on_test_begin(self, logs=None):
        """
        5: fit
        1: evaluate
        """
        keys = list(logs.keys())
        print(f'Start testing; got log keys: {keys}')

    def on_test_end(self, logs=None):
        """
        8: fit
        4: evaluate
        """
        keys = list(logs.keys())
        print(f'Stop testing; got log keys: {keys}')

    def on_predict_begin(self, logs=None):
        """
        1: predict
        """
        keys = list(logs.keys())
        print(f'Start predicting; got log keys: {keys}')

    def on_predict_end(self, logs=None):
        """
        4: predict
        """
        keys = list(logs.keys())
        print(f'Stop predicting; got log keys: {keys}')

    def on_train_batch_begin(self, batch, logs=None):
        """
        3: fit
        """
        keys = list(logs.keys())
        print(f'...Training: start of batch{batch}; got log keys: {keys}')

    def on_train_batch_end(self, batch, logs=None):
        """
        4: fit
        """
        keys = list(logs.keys())
        print(f'...Training: end of batch{batch}; got log keys: {keys}')

    def on_test_batch_begin(self, batch, logs=None):
        """
        6: fit
        2: evaluate
        """
        keys = list(logs.keys())
        print(f'...Evaluating: start of batch{batch}; got log keys: {keys}')

    def on_test_batch_end(self, batch, logs=None):
        """
        7: fit
        3: evaluate
        """
        keys = list(logs.keys())
        print(f'...Evaluating: end of batch{batch}; got log keys: {keys}')

    def on_predict_batch_begin(self, batch, logs=None):
        """
        2: predict
        """
        keys = list(logs.keys())
        print(f'...Predicting: start of batch{batch}; got log keys: {keys}')

    def on_predict_batch_end(self, batch, logs=None):
        """
        3: predict
        """
        keys = list(logs.keys())
        print(f'...Predicting: end of batch{batch}; got log keys: {keys}')


model = get_model()
model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=1,
    verbose=0,
    validation_split=0.5,
    callbacks=[CustomCallback()]
)

res = model.evaluate(
    x_test, y_test, batch_size=128, verbose=0, callbacks=[CustomCallback()]
)
res = model.predict(x_test, batch_size=128, callbacks=[CustomCallback()])


# region logs 字典的用法
# logs 字典包含损失值，以及批次或周期结束时的所有指标

class LossAndErrorPrintingCallback(keras.callbacks.Callback):

    def on_train_batch_end(self, batch, logs=None):
        print(f'Up to batch {batch}, the average loss is {logs["loss"]:7.2f}.')

    def on_test_batch_end(self, batch, logs=None):
        print(f'test;Up to batch {batch}, the average loss is {logs["loss"]:7.2f}.')

    def on_epoch_end(self, epoch, logs=None):
        print(f'The average loss for epoch {epoch} is {logs["loss"]:7.2f} '
              f'and mean absolute error is {logs["mean_absolute_error"]:7.2f}')


model = get_model()
model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=2,
    verbose=0,
    validation_split=0.5,
    callbacks=[LossAndErrorPrintingCallback()]
)

res = model.evaluate(
    x_test,
    y_test,
    batch_size=128,
    verbose=0,
    callbacks=[LossAndErrorPrintingCallback()]
)


# endregion


# endregion


# region self.model 属性的用法

# 设置 self.model.stop_training = True 以立即中断训练
# 转变优化器（可作为 self.model.optimizer）的超参数，例如 self.model.optimizer.learning_rate
# 定期保存模型
# 在每个周期结束时，在少量测试样本上记录 model.predict() 的输出，以用作训练期间的健全性检查
# 在每个周期结束时提取中间特征的可视化，随时间推移监视模型当前的学习内容

# endregion


# region Keras 回调函数应用示例

# region 在达到最小损失时尽早停止
# 设置 self.model.stop_training（布尔）属性来创建能够在达到最小损失时停止训练的 Callback
#   还可以提供参数 patience 来指定在达到局部最小值后应该等待多少个周期然后停止

class EarlyStoppingAtMinLoss(keras.callbacks.Callback):

    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights 存储损失最小的权重
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.wait = 0  # 当损失不再是最小值时等待的时间
        self.stopped_epoch = 0  # 训练停止的时间
        self.best = np.Inf  # 将最大值初始化为无穷大

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('loss')
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # 如果当前结果更好(更少)，则记录最佳权重
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print('Restoring model weights from the end of the best eopch.')
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f'Epoch {(self.stopped_epoch + 1):5d}: early stopping')


model = get_model()
model.fit(
    x_train,
    y_train,
    batch_size=64,
    steps_per_epoch=5,
    epochs=30,
    verbose=0,
    callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()]
)


# endregion


# region 学习率规划

class CustomLearningRateScheduler(keras.callbacks.Callback):

    def __init__(self, scheduler):
        super(CustomLearningRateScheduler, self).__init__()
        self.scheduler = scheduler

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        scheduled_lr = self.scheduler(epoch, lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print(f'\nEpoch {epoch:5d}: Learning rate is {scheduled_lr:6.4f}.')


LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (3, 0.05),
    (6, 0.01),
    (9, 0.005),
    (12, 0.001),
]


def lr_schedule(epoch, lr):
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr


model = get_model()
model.fit(
    x_train,
    y_train,
    batch_size=64,
    steps_per_epoch=5,
    epochs=15,
    verbose=0,
    callbacks=[LossAndErrorPrintingCallback(), CustomLearningRateScheduler(lr_schedule)]
)

# endregion


# endregion
