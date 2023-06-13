# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: basic_training_loops
# CreateTime: 2023/6/13 14:10
# Summary: 基本训练循环


import tensorflow as tf
from matplotlib import pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# f(x) = x * W + b
#  给定x， y 找到直线的斜率 和偏移量

# region 数据, x 输入数据，y输出数据

TRUE_W = 3.
TRUE_B = 2.

NUM_EXAMPLES = 201
x = tf.linspace(-2, 2, NUM_EXAMPLES)
x = tf.cast(x, tf.float32)


# 监督学习使用 输入x 和输出y
def f(x):
    return x * TRUE_W + TRUE_B


noise = tf.random.normal(shape=[NUM_EXAMPLES])

y = f(x) + noise  # 将高斯（正态）噪声添加到直线上的点而合成的一些数据

plt.plot(x, y, '.')
plt.show()


# 张量以 batches 的形式聚集在一起，或者是成组的 输入和输出 堆叠在一起；
#  给定次数据集的大小，可以将整个数据集视为一个批次

# endregion

# region 定义模型

class MyModel(tf.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.w = tf.Variable(5., name='w')
        self.b = tf.Variable(0., name='b')

    def __call__(self, x):
        return self.w * x + self.b


model = MyModel()

print('Variables: ', model.variables)

assert model(3.).numpy() == 15.


# endregion

# region 定义损失函数

# 损失函数 衡量 给定输入的模型 输出与目标输出的匹配程度
#  目的是 减少训练中这种差异
def loss(target_y, predicted_y):
    return tf.reduce_mean(tf.square(target_y - predicted_y))


plt.plot(x, y, '.', label='Data')
plt.plot(x, f(x), label='Ground truth')  # 真实
plt.plot(x, model(x), label='Predictions')  # 预测
plt.legend()
plt.show()

print('Current loss: %1.6f' % loss(y, model(x)).numpy())


# endregion

# region 定义循环训练

def train(model, x, y, learning_rate):
    with tf.GradientTape() as tape:
        # 自动跟踪可训练变量
        current_loss = loss(y, model(x))

    # 使用GradientTape计算关于W和b的梯度
    dw, db = tape.gradient(current_loss, [model.w, model.b])

    # 减去由学习率缩放的梯度
    model.w.assign_sub(learning_rate * dw)  # w = w - (learning_rate * dw) # 相减后更新赋值
    model.b.assign_sub(learning_rate * db)


model = MyModel()

weights = []
biases = []
epochs = range(10)


def report(model, loss):
    # 观察 w b 的变化
    return f'W = {model.w.numpy():1.2f}, b = {model.b.numpy():1.2f}, loss = {loss:2.5f}'


def training_loop(model, x, y):
    for epoch in epochs:
        train(model, x, y, learning_rate=0.1)

        weights.append(model.w.numpy())
        biases.append(model.b.numpy())
        current_loss = loss(y, model(x))

        print(f'Epoch {epoch:2d}: ')
        print('   ', report(model, current_loss))


# 进行训练
current_loss = loss(y, model(x))
print(f'Starting: ')
print('   ', report(model, current_loss))
training_loop(model, x, y)

# 使用视图 展示  权重随时间的演变
plt.plot(epochs, weights, label='Weights', color=colors[0])
plt.plot(epochs, [TRUE_W] * len(epochs), '--', label='True weight', color=colors[0])

plt.plot(epochs, biases, label='bias', color=colors[1])
plt.plot(epochs, [TRUE_B] * len(epochs), '--', label='True bias', color=colors[1])

plt.legend()
plt.show()

# 呈现训练的模型的性能
plt.plot(x, y, '.', label='Data')
plt.plot(x, f(x), label='Ground truth')
plt.plot(x, model(x), label='Predictions')
plt.legend()
plt.show()

print('Current loss: %1.6f' % loss(model(x), y).numpy())


# endregion

# region 使用Keras完成相同的解决方案

class MyModelKeras(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.w = tf.Variable(5., name='w')
        self.b = tf.Variable(2., name='b')

    def call(self, x):
        return self.w * x + self.b


keras_model = MyModelKeras()

training_loop(keras_model, x, y)  # 可以使用keras 内置功能，不必每次都写新的 训练循环

keras_model.save_weights('basic_training_loops/my_checkpoint')  # 保存检查点

# 使用 model.compile() 去设置参数, 使用model.fit() 进行训练
keras_model = MyModelKeras()
keras_model.compile(
    run_eagerly=False,  # 默认情况 fit 使用 tf.function
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    loss=tf.keras.losses.mean_squared_error,  # 也可以 用 上面自定义的损失函数 loss
    # loss=loss,
)

keras_model.fit(x, y, epochs=10, batch_size=1000)

# endregion
