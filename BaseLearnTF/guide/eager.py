# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: eager
# CreateTime: 2023/6/8 16:30
# Summary: TensorFlow 基础知识

# https://tensorflow.google.cn/guide/eager?hl=zh-cn

# Eager Execution 用于研究和试验的灵活机器学习平台
#  直观的界面
#  更方便的调试功能
#  自然的控制流
import os
import time

import numpy as np
import tensorflow as tf
import cProfile

from matplotlib import pyplot as plt

from utils import join_path

tf.executing_eagerly()  # 开启 eager


# x = [[2.]]
# m = tf.matmul(x, x)  # 两数相乘
# print(f'hello, {m}')
#
# a = tf.constant([[1, 2],
#                  [3, 4]])
#
# b = tf.add(a, 1)  # 所有元素值+1  # array([[2, 3], [4, 5]])>
#
# print(a * b)  # a * b == c
#
# c = np.multiply(a, b)  # a * b 后转换了类型


# # region 动态控制流
#
# def fizzbuzz(max_num):
#     counter = tf.constant(0)
#     max_num = tf.convert_to_tensor(max_num)
#     for num in range(1, max_num.numpy() + 1):
#         num = tf.constant(num)
#         if int(num % 3) == 0 and int(num % 5) == 0:
#             print('FizzBuzz')
#         elif int(num % 3) == 0:
#             print('Fizz')
#         elif int(num % 5) == 0:
#             print('Buzz')
#         else:
#             print(num.numpy())
#         counter += 1
#
#
# fizzbuzz(15)
#
# # endregion
#
#
# # region Eager 训练
#
# # 计算梯度
# w = tf.Variable([[1.0]])
# with tf.GradientTape() as tape:  # GradientTape 跟踪运算， 稍后计算梯度
#     loss = w * w
#
# grad = tape.gradient(loss, w)
# print(grad)  # tf.Tensor([[2.]], shape=(1, 1), dtype=float32)
#
# # endregion
#
#
# # region 训练模型
# file_path = join_path.keras_data_path.helloworld_keras_mnist_data_path  # 已经下载的数据集地址
#
# # 加载 mnist 数据集
# mnist = tf.keras.datasets.mnist
# (mnist_images, mnist_labels), _ = mnist.load_data(file_path)
#
# dataset = tf.data.Dataset.from_tensor_slices(
#     (tf.cast(mnist_images[..., tf.newaxis] / 255, tf.float32),
#      tf.cast(mnist_labels, tf.int64))
# )
# dataset = dataset.shuffle(1000).batch(32)
#
# # 构建model
# mnist_model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(16, [3, 3], activation='relu', input_shape=[None, None, 1]),
#     tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
#     tf.keras.layers.GlobalAvgPool2D(),
#     tf.keras.layers.Dense(10),
# ])
#
# # 沒有训练，可在 Eager Execution 中调用模型并检查输出
# for images, labels in dataset.take(1):
#     print('Logits: ', mnist_model(images[0: 1]).numpy())  # 打印一个
#
# # Keras 内置了循环训练（fit方法）
# # 下面使用Eager Execution实现循环训练
# optimizer = tf.keras.optimizers.Adam()
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#
# loss_history = []
#
#
# def train_step(images, labels):
#     with tf.GradientTape() as tape:
#         logits = mnist_model(images, training=True)
#
#         tf.debugging.assert_equal(logits.shape, (32, 10))
#
#         loss_value = loss_object(labels, logits)
#
#     loss_history.append(loss_value.numpy().mean())
#     grads = tape.gradient(loss_value, mnist_model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))
#
#
# def train(epochs):
#     for epoch in range(epochs):
#         for (batch, (images, labels)) in enumerate(dataset):
#             train_step(images, labels)
#         print(f'Epoch {epoch} finished.')
#
#
# # train(epochs=3)
# #
# # plt.plot(loss_history)
# # plt.xlabel('Batch')
# # plt.ylabel('Loss [entropy]')
# # plt.show()
#
#
# # endregion
#
#
# # region 变量和优化器
#
#
# class Linear(tf.keras.Model):
#     """简单的自动微分"""
#
#     def __init__(self):
#         super(Linear, self).__init__()
#         self.W = tf.Variable(5., name='weight')
#         self.B = tf.Variable(10., name='bias')
#
#     def call(self, inputs):
#         return inputs * self.W + self.B
#
#
# NUM_EXAMPLES = 2000
# training_inputs = tf.random.normal([NUM_EXAMPLES])
# noise = tf.random.normal([NUM_EXAMPLES])
# training_outputs = training_inputs * 3 + 2 + noise
#
#
# def loss(model, inputs, targets):
#     error = model(inputs) - targets
#     return tf.reduce_mean(tf.square(error))
#
#
# def grad(model, inputs, targets):
#     with tf.GradientTape() as tape:
#         loss_value = loss(model, inputs, targets)
#
#     return tape.gradient(loss_value, [model.W, model.B])
#
#
# model = Linear()
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
#
# print(f'Initial loss: {loss(model, training_inputs, training_outputs):.3f}')
#
# steps = 300
# for i in range(steps):
#     grads = grad(model, training_inputs, training_outputs)
#     optimizer.apply_gradients(zip(grads, [model.W, model.B]))
#     if i % 20 == 0:
#         print(f'Loss at step {i:03d}: {loss(model, training_inputs, training_outputs):.3f}')
#
# print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
# print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))
#
# # endregion
#
# # region 对象的保存
#
# # save_weights 创建检查点
# model.save_weights('guide/eager/weights')
# status = model.load_weights('guide/eager/weights')
#
# x = tf.Variable(10.)
# checkpoint = tf.train.Checkpoint(x=x)
#
# x.assign(2.)  # 设置一个新值
# checkpoint_path = './guide/eager/ckpt/'
# checkpoint.save('./guide/eager/ckpt/')
#
# x.assign(11.)  # 设置一个新值
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))  # 从检查点恢复值  x => 2
# print(x)
#
# # 记录 model、optimizer 和全局步骤的状态
# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dense(10)
# ])
#
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# checkpoint_dir = 'guide/eager/path/to/model_dir'
# if not os.path.exists(checkpoint_dir):
#     os.makedirs(checkpoint_dir)
#
# checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
# root = tf.train.Checkpoint(optimizer=optimizer, model=model)  # 存储对象的内部状态
#
# root.save(checkpoint_prefix)
# root.restore(tf.train.latest_checkpoint(checkpoint_dir))  # 变量创建后立即恢复，并且可以使用断言来确保检查点已完全加载
#
# # endregion
#
# # region 面向对象的指标
#
# m = tf.keras.metrics.Mean("loss")
# m(0)
# m(5)
# m.result()  # => 2.5
# m([8, 9])
# m.result()  # => 5.5
#
# # endregion
#
#
# logdir = './guide/eager/tb/'
# writer = tf.summary.create_file_writer(logdir)
#
# steps = 1000
# with writer.as_default():
#     for i in range(steps):
#         step = i + 1
#         loss = 1 - 0.001 * steps
#         if step % 100 == 0:
#             tf.summary.scalar('loss', loss, step=step)
#
#
# # region 自动微分
#
# def line_search_step(fn, init_x, rate=1.0):
#     """ 回溯线搜索算法 """
#     with tf.GradientTape() as tape:  # GradientTape 可用于动态模型，
#         tape.watch(init_x)
#         value = fn(init_x)
#
#     grad = tape.gradient(value, init_x)
#     grad_norm = tf.reduce_sum(grad * grad)
#     init_value = value
#
#     # 简单的理解为：搜索不通，退回一步继续搜索
#     while value > init_value - rate * grad_norm:
#         x = init_x - rate * grad
#         value = fn(x)
#         rate /= 2.0
#     return x, value
#
#
# # endregion
#
# # region 自定义梯度（重写梯度）
#
# @tf.custom_gradient
# def clip_gradient_by_norm(x, norm):
#     """ 向后传递中剪裁梯度范数 """
#     y = tf.identity(x)
#
#     def grad_fn(dresult):
#         return [tf.clip_by_norm(dresult, norm), None]
#
#     return y, grad_fn
#
#
# def log1pexp(x):
#     return tf.math.log(1 + tf.exp(x))
#
#
# # 自定义梯度通常用来为运算序列提供数值稳定的梯度
# def grad_log1pexp(x):
#     with tf.GradientTape() as tape:
#         tape.watch(x)
#         value = log1pexp(x)
#     return tape.gradient(value, x)
#
#
# grad_log1pexp(tf.constant(0.)).numpy()  # 0.5
#
# grad_log1pexp(tf.constant(100.)).numpy()  # nan
#
#
# # endregion
#
# # region 优化上面的自定义梯度
#
# @tf.custom_gradient
# def log1pexp_v2(x):
#     e = tf.exp(x)
#
#     def grad(dy):
#         return dy * (1 - 1 / (1 + e))
#
#     return tf.math.log(1 + e), grad
#
#
# def grad_log1pexp_v2(x):
#     with tf.GradientTape() as tape:
#         tape.watch(x)
#         value = log1pexp_v2(x)
#
#     return tape.gradient(value, x)
#
#
# grad_log1pexp_v2(tf.constant(0.)).numpy()  # 0.5
# grad_log1pexp_v2(tf.constant(100.)).numpy()  # 1.0
#
#
# # endregion

# region 性能

def measure(x, steps):
    tf.matmul(x, x)
    start = time.time()
    for i in range(steps):
        x = tf.matmul(x, x)

    _ = x.numpy()
    return time.time() - start


shape = (1000, 1000)
steps = 200
print(f'Time to multiply a {shape} matrix by itself {steps} times: ')

with tf.device('/cpu:0'):
    print(f'CPU: {measure(tf.random.normal(shape), steps)} secs')

if tf.config.experimental.list_logical_devices('GPU'):
    with tf.device('/gpu:0'):
        print(f'GPU: {measure(tf.random.normal(shape), steps)} secs')
else:
    print('GPU: not found')

x = tf.random.normal([10, 10])

x_gpu0 = x.gpu()
x_cpu = x.cpu()

_ = tf.matmul(x_cpu, x_cpu)  # Runs on CPU
_ = tf.matmul(x_gpu0, x_gpu0)  # Runs on GPU:0
# endregion
