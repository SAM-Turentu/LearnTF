# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: intro_to_modules
# CreateTime: 2023/6/12 10:58
# Summary: 模块、层、模型


from datetime import datetime

import tensorflow as tf


class SimpleModule(tf.Module):

    def __init__(self, name=None):
        super().__init__(name=name)
        self.a_variable = tf.Variable(5., name='train_me')
        self.non_trainable_variable = tf.Variable(5., trainable=False, name='do_not_train_me')

    def __call__(self, x):
        return self.a_variable * x + self.non_trainable_variable


simple_module = SimpleModule(name='simple')
simple_module(tf.constant(5.))


# simple_module.trainable_variables # 可训练变量
# simple_module.variables # 所有变量

# 密集（线性）层
class Dense(tf.Module):

    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(
            tf.random.uniform([in_features, out_features]), name='w'
        )
        self.b = tf.Variable(tf.zeros([out_features]), name='b')

    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


class SequentialModule(tf.Module):
    """创建了两个层实例"""

    def __init__(self, name=None):
        super(SequentialModule, self).__init__(name=name)

        self.dense_1 = Dense(in_features=3, out_features=3)
        self.dense_2 = Dense(in_features=3, out_features=2)

    def __call__(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)


my_model = SequentialModule(name='the_model')
my_model(tf.constant([[2., 2., 2.]]))


# my_model.submodules  # 使用单个模型管理 tf.Module 集合


# region 等待创建变量

# class FlexibleDenseModule(tf.Module):
#
#     def __init__(self, out_features, name=None):
#         super().__init__(name=name)
#         self.is_built = False
#         self.out_features = out_features
#
#     def __call__(self, x):
#         if not self.is_built:
#             self.w = tf.Variable(
#                 tf.random.uniform([x.shape[-1], self.out_features]), name='w'
#             )
#             self.b = tf.Variable(tf.zeros([self.out_features]), name='b')
#             self.is_built = True
#         y = tf.matmul(x, self.w) + self.b
#         return tf.nn.relu(y)
#
#
# class MySequentialModule(tf.Module):
#
#     def __init__(self, name=None):
#         super().__init__(name=name)
#
#         self.dense_1 = FlexibleDenseModule(out_features=3)
#         self.dense_2 = FlexibleDenseModule(out_features=2)
#
#     def __call__(self, x):
#         x = self.dense_1(x)
#         return self.dense_2(x)
#
#
# my_model = MySequentialModule(name='the_model')
# my_model(tf.constant([[2., 2., 2.]]))


# endregion


# region 保存函数

class MySequentialModule(tf.Module):

    def __init__(self, name=None):
        super().__init__(name=name)

        self.dense_1 = Dense(in_features=3, out_features=3)
        self.dense_2 = Dense(in_features=3, out_features=2)

    # 计算图
    @tf.function
    def __call__(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)


my_model = MySequentialModule(name='the_model')
my_model(tf.constant([[2., 2., 2.]]))
my_model(tf.constant([[2., 2., 2.], [2., 2., 2.]]))

stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
logdir = 'intro_to_modules/logs/func/%s' % stamp
writer = tf.summary.create_file_writer(logdir)

new_model = MySequentialModule()

tf.summary.trace_on(graph=True)
tf.profiler.experimental.start(logdir)

print(new_model(tf.constant([[2., 2., 2.]])))
with writer.as_default():
    tf.summary.trace_export(
        name='my_func_trace',
        step=0,
        profiler_outdir=logdir
    )

# %tensorboard --logdir intro_to_modules/logs/func  --port 6007

tf.saved_model.save(my_model, "intro_to_modules/the_saved_model")

# new_model = tf.saved_model.load("the_saved_model")  # 加载

# isinstance(new_model, SequentialModule)
#  new_model 是tf内部的用户对象，不是 SequentialModule 类型的对象
print(new_model([[2.0, 2.0, 2.0]]))


# endregion


class MyLayersDense(tf.keras.layers.Layer):

    def __init__(self, in_feature, out_feature, **kwargs):
        super().__init__(**kwargs)

        self.w = tf.Variable(
            tf.random.normal([in_feature, out_feature]), name='w'
        )
        self.b = tf.Variable(tf.zeros([out_feature]), name='b')

    # keras层 有自己的 __call__
    def call(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


simple_layer = MyLayersDense(name='simple', in_feature=3, out_feature=3)
print(simple_layer([[2., 2., 2.]]))


class FlexibleDense(tf.keras.layers.Layer):

    def __init__(self, out_feature, **kwargs):
        super().__init__(**kwargs)
        self.out_feature = out_feature

    # build 仅被调用一次，而且是使用输入的形状调用的。通常用于创建变量（权重）
    def build(self, input_shape):
        self.w = tf.Variable(
            tf.random.normal([input_shape[-1], self.out_feature]), name='w'
        )
        self.b = tf.Variable(tf.zeros([self.out_feature]), name='b')

    def call(self, inpusts):
        return tf.matmul(inpusts, self.w) + self.b


flexible_dense = FlexibleDense(out_feature=3)
print(flexible_dense(tf.constant([[2., 2., 2.], [3., 3., 3.]])))


# build 仅被调用一次，所以输入形状与层的变量不兼容时，输入被拒绝
# print(flexible_dense(tf.constant([[2., 2., 2., 2.]])))

# region Keras 模型

class MySequentialModel(tf.keras.Model):

    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)

        self.dense_1 = FlexibleDense(out_feature=3)
        self.dense_2 = FlexibleDense(out_feature=2)

    # 将 __call__ 转换为 call() 并更改父项
    def call(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)


my_sequential_model = MySequentialModel(name='the_model')
print(my_sequential_model(tf.constant([[2., 2., 2.]])))

# 重写 tf.keras.Model 是构建tf模型的 极python化方式，迁移模型，非常简单

# 下面是使用函数式 API 构造相同的模型
inputs = tf.keras.Input(shape=[3, ])
x = FlexibleDense(3)(inputs)
x = FlexibleDense(2)(x)
my_functional_model = tf.keras.Model(inputs=inputs, outputs=x)
my_functional_model.summary()

# endregion
