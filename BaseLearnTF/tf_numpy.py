# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: tf_numpy
# CreateTime: 2023/6/18 13:17
# Summary: tf 的 numpy


import timeit
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

# 启用 Numpy 行为

tnp.experimental_enable_numpy_behavior()
# 此调用可在 TensorFlow 中启用类型提升，并在将文字转换为张量时更改类型推断，以更严格地遵循 NumPy 标准
# 此调用更改整个TensorFlow 的行为


# region TensorFlow NumPy ND 数组

ones = tnp.ones([5, 3], dtype=tnp.float32)
print('Create ND array with shape = %s, rank = %s, dtype = %s, on device = %s\n'
      % (ones.shape, ones.ndim, ones.dtype, ones.device))

print('Is `ones` an instance of tf.Tensor: %s\n' % isinstance(ones, tf.Tensor))
print('ndarray.T has shape %s' % str(ones.T.shape))
print('narray.reshape(-1) has shape %s' % ones.reshape(-1).shape)

# region 类型提升

values = [tnp.asarray(1, dtype=d) for d in (tnp.int32, tnp.int64, tnp.float32, tnp.float64)]
for i, v1 in enumerate(values):
    for v2 in values[i + 1:]:
        print('%s + %s => %s' % (v1.dtype.name, v2.dtype.name, (v1 + v2).dtype.name))

print('tnp.asarray(1).dtype == tnp.%s' % tnp.asarray(1).dtype.name)  # tnp.int32
print('tnp.asarray(1.).dtype == tnp.%s' % tnp.asarray(1.).dtype.name)  # tnp.float64

# NumPy 倾向于使用 tnp.int64 tnp.float64

tnp.experimental_enable_numpy_behavior(prefer_float32=True)
print('tnp.asarray(1).dtype == tnp.%s' % tnp.asarray(1).dtype.name)  # tnp.int32
print('tnp.asarray(1.).dtype == tnp.%s' % tnp.asarray(1.).dtype.name)  # tnp.float32

tnp.experimental_enable_numpy_behavior(prefer_float32=False)  # 默认 False
print('tnp.asarray(1).dtype == tnp.%s' % tnp.asarray(1).dtype.name)  # tnp.int32
print('tnp.asarray(1.).dtype == tnp.%s' % tnp.asarray(1.).dtype.name)  # tnp.float64

# endregion


# region 广播

x = tnp.ones([2, 3])
y = tnp.ones([3])
z = tnp.ones([1, 2, 1])
print('Broadcasting shapes %s. %s and %s gives shape %s' % (x.shape, y.shape, z.shape, (x + y + z).shape))
# (x + y + z).shape  #  (1, 2, 3)
# x + y + z == [[[3., 3., 3.], [3., 3., 3.]]]

# endregion


# region 索引

x = tnp.arange(24).reshape(2, 3, 4)
# array([[[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]],
#        [[12, 13, 14, 15],
#         [16, 17, 18, 19],
#         [20, 21, 22, 23]]])>

print(x[1, tnp.newaxis, 1:3, ...], '\n')
# tf.Tensor(
# [[[16 17 18 19]
#   [20 21 22 23]]], shape=(1, 2, 4), dtype=int32)

print(x[:, [True, False, True]], '\n')
# tf.Tensor(
# [[[ 0  1  2  3]
#   [ 8  9 10 11]]
#  [[12 13 14 15]
#   [20 21 22 23]]], shape=(2, 2, 4), dtype=int32)

print(x[1, (0, 0, 1), tnp.asarray([0, 1, 1])])
# tf.Tensor([12 13 17], shape=(3,), dtype=int32)

try:
    tnp.arange(6)[1] = -1  # 无法赋值
except TypeError:
    print('Currently, TensorFlow NumPy does not support mutation')


# endregion


# region 示例模型

class Model(object):

    def __init__(self):
        self.weights = None

    def predict(self, inputs):
        if self.weights is None:
            size = inputs.shape[1]
            stddev = tnp.sqrt(size).astype(tnp.float32)
            w1 = tnp.random.randn(size, 64).astype(tnp.float32) / stddev
            bias = tnp.random.randn(64).astype(tnp.float32)
            w2 = tnp.random.randn(64, 2).astype(tnp.float32) / 8
            self.weights = (w1, bias, w2)
        else:
            w1, bias, w2 = self.weights
        y = tnp.matmul(inputs, w1) + bias
        y = tnp.maximum(y, 0)  # ReLU 层
        return tnp.matmul(y, w2)


model = Model()
print(model.predict(tnp.ones([2, 32], dtype=tnp.float32)))
# tf.Tensor(
# [[ 1.2506579  -0.59781814]
#  [ 1.2506579  -0.59781814]], shape=(2, 2), dtype=float32)

# endregion


# endregion


# region TensorFlow NumPy 和 NumPy

# region NumPy 互操作性
#  ND 数组与 np.ndarray 之间的转换可能会触发实际数据副本
np_sum = np.sum(tnp.ones([2, 3]))
print('sum = %s. Class: %s' % (float(np_sum), np_sum.__class__))
# sum = 6.0. Class: <class 'numpy.float64'>

tnp_sum = tnp.sum(np.ones([2, 3]))
print('sum = %s. Class: %s' % (float(tnp_sum), tnp_sum.__class__))
# sum = 6.0. Class: <class 'tensorflow.python.framework.ops.EagerTensor'>

labels = 15 + 2 * tnp.random.randn(1, 1000)
plt.hist(labels)
plt.show()
# endregion

# 缓冲区副本
#  tf 对齐要求比 numpy 严格，在需要时触发副本，需要注意复制数据的开销
#  ND 数组可以引用放置在本地 CPU 内存以外设备上的缓冲区。在这种情况下，调用 NumPy 函数将根据需要触发网络或设备上的副本


# region 算子优先级

# 同时涉及 ND 数组和 np.ndarray 的算子，前者将获得优先权
x = tnp.ones([2]) + np.ones([2])
print('x = %s\nclass = %s' % (x, x.__class__))
# x = tf.Tensor([2. 2.], shape=(2,), dtype=float64)
# class = <class 'tensorflow.python.framework.ops.EagerTensor'>

# endregion


# endregion


# region TF NumPy 和 TensorFlow

# tf.Tensor 和 ND 数组
# ND 数组是 tf.Tensor 的别名，因此显然可以在不触发实际数据副本的情况下将它们混合到一起
x = tf.constant([1, 2])  # tf.Tensor([1 2], shape=(2,), dtype=int32)

tnp_x = tnp.asarray(x)  # tf.Tensor([1 2], shape=(2,), dtype=int32)
print(tf.convert_to_tensor(tnp_x))  # tf.Tensor([1 2], shape=(2,), dtype=int32)

print(x.numpy(), x.numpy().__class__)  # [1 2] <class 'numpy.ndarray'>

# region TensorFlow 互操作性
#  ND 数组可以传递给 TensorFlow API,如上文所述，因为 ND 数组只是 tf.Tensor 的别名。
#  即使是放置在加速器或远程设备上的数据，这种互操作也不会创建数据副本


# 将 tf.Tensor 对象传递给 tf.experimental.numpy API，而无需执行数据副本
tf_sum = tf.reduce_sum(tnp.ones([2, 3], tnp.float32))
print('Output = %s' % tf_sum)  # Output = tf.Tensor(6.0, shape=(), dtype=float32)

tnp_sum = tnp.sum(tf.ones([2, 3]))
print('Output = %s' % tnp_sum)  # Output = tf.Tensor(6.0, shape=(), dtype=float32)


# endregion


# region 梯度和雅可比矩阵：tf.GradientTape

def create_batch(batch_size=32):
    return (tnp.random.randn(batch_size, 32).astype(tnp.float32),
            tnp.random.randn(batch_size, 2).astype(tnp.float32))


def compute_gradients(model, inputs, labels):
    """ 计算模型预测和标签之间的平方损失梯度 """
    with tf.GradientTape() as tape:
        assert model.weights is not None
        tape.watch(model.weights)
        prediction = model.predict(inputs)
        loss = tnp.sum(tnp.sqrt(prediction - labels))
    return tape.gradient(loss, model.weights)


inputs, labels = create_batch()
gradients = compute_gradients(model, inputs, labels)

# 检查返回梯度的形状，以验证它们与参数形状匹配
print('Parameter shapes: ', [w.shape for w in model.weights])  # [TensorShape([32, 64]), TensorShape([64]), TensorShape([64, 2])]
print('Gradient shapes: ', [g.shape for g in gradients])  # [TensorShape([32, 64]), TensorShape([64]), TensorShape([64, 2])]
assert isinstance(gradients[0], tnp.ndarray)


def prediction_batch_jacobian(inputs):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        prediction = model.predict(inputs)
    return prediction, tape.batch_jacobian(prediction, inputs)


inp_batch = tnp.ones([16, 32], tnp.float32)
output, batch_jacobian = prediction_batch_jacobian(inp_batch)
print('Output shape: %s, input shape: %s' % (output.shape, inp_batch.shape))
print('Batch jacobian shape: ', batch_jacobian.shape)

# endregion


# region 跟踪编译：tf.function

inputs, labels = create_batch(512)
compute_gradients(model, inputs, labels)
print(timeit.timeit(lambda: compute_gradients(model, inputs, labels),
                    number=10) * 100, 'ms')  # 2.156000000104541 ms

complied_compute_gradients = tf.function(compute_gradients)
complied_compute_gradients(model, inputs, labels)
print(timeit.timeit(lambda: compute_gradients(model, inputs, labels),
                    number=10) * 100, 'ms')  # 0.7104699998308206 ms


# endregion


# region 向量化：tf.vectorized_map
# tf 内置对向量化并行循环的支持，可以将速度提高一到两个数量级

@tf.function
def vectorized_per_example_gradients(inputs, labels):
    def single_example_gradient(arg):
        inp, label = arg
        return compute_gradients(model,
                                 tnp.expand_dims(inp, 0),
                                 tnp.expand_dims(label, 0))

    return tf.vectorized_map(single_example_gradient, (inputs, labels))


batch_size = 128
inputs, labels = create_batch(batch_size)
per_example_gradients = vectorized_per_example_gradients(inputs, labels)
for w, p in zip(model.weights, per_example_gradients):
    print('Weight shape: %s, batch size: %s, per example gradient shape: %s' % (
        w.shape, batch_size, p.shape
    ))


@tf.function
def unvectorized_per_example_gradients(inputs, labels):
    def single_example_gradient(arg):
        inp, label = arg
        return compute_gradients(model,
                                 tnp.expand_dims(inp, 0),
                                 tnp.expand_dims(label, 0))

    return tf.map_fn(single_example_gradient, (inputs, labels),
                     fn_output_signature=(tf.float32, tf.float32, tf.float32))


print(timeit.timeit(lambda: vectorized_per_example_gradients(inputs, labels),
                    number=10) * 100, 'ms')  # 0.6389400001353351 ms

per_example_gradients = unvectorized_per_example_gradients(inputs, labels)
print(timeit.timeit(lambda: unvectorized_per_example_gradients(inputs, labels),
                    number=10) * 100, 'ms')  # 99.0780399999494 ms

# endregion


# region 设备放置(TensorFlow NumPy 可以将运算置于 CPU、GPU、TPU 和远程设备上)

# 先列出设备
print(tf.config.list_logical_devices())
print(tf.config.list_physical_devices())
try:
    device = tf.config.list_logical_devices(device_type='GPU')[0]
except:
    device = '/device:CPU:0'

# 放置运算
with tf.device(device):
    prediction = model.predict(create_batch(5)[0])

print('prediction si placed on %s' % prediction.device)

# 跨设备复制ND数组：tnp.copy
with tf.device('/device:CPU:0'):
    prediction_cpu = tnp.copy(prediction)

print(prediction.device)
print(prediction_cpu.device)


# endregion


# endregion


# region 性能比较
# TensorFlow NumPy 使用高度优化的 TensorFlow 内核

def benchmark(f, inputs, number=30, force_gpu_sync=False):
    times = []
    for inp in inputs:
        def _g():
            if force_gpu_sync:
                one = tnp.asarray(1)
            f(inp)
            if force_gpu_sync:
                with tf.device("CPU:0"):
                    tnp.copy(one)  # Force a sync for GPU case

        _g()  # warmup
        t = timeit.timeit(_g, number=number)
        times.append(t * 1000. / number)
    return times


def plot(np_times, tnp_times, compiled_tnp_times, has_gpu, tnp_times_gpu):
    plt.xlabel("size")
    plt.ylabel("time (ms)")
    plt.title("Sigmoid benchmark: TF NumPy vs NumPy")
    plt.plot(sizes, np_times, label="NumPy")
    plt.plot(sizes, tnp_times, label="TF NumPy (CPU)")
    plt.plot(sizes, compiled_tnp_times, label="Compiled TF NumPy (CPU)")
    if has_gpu:
        plt.plot(sizes, tnp_times_gpu, label="TF NumPy (GPU)")
    plt.legend()
    plt.show()


def np_sigmoid(y):
    return 1. / (1. + np.exp(-y))


def tnp_sigmoid(y):
    return 1. / (1. + tnp.exp(-y))


@tf.function
def compiled_tnp_sigmoid(y):
    return tnp_sigmoid(y)


sizes = (2 ** 0, 2 ** 5, 2 ** 10, 2 ** 15, 2 ** 20)
np_inputs = [np.random.randn(size).astype(np.float32) for size in sizes]
np_times = benchmark(np_sigmoid, np_inputs)

with tf.device('/device:CPU:0'):
    tnp_inputs = [tnp.random.randn(size).astype(np.float32) for size in sizes]
    tnp_times = benchmark(tnp_sigmoid, tnp_inputs)
    compiled_tnp_times = benchmark(compiled_tnp_sigmoid, tnp_inputs)

has_gpu = len(tf.config.list_logical_devices('GPU'))
if has_gpu:
    with tf.device('/device:GPU:0'):
        tnp_inputs = [tnp.random.randn(size).astype(np.float32) for size in sizes]
        tnp_times_gpu = benchmark(compiled_tnp_sigmoid, tnp_inputs, 100, True)
else:
    tnp_times_gpu = None

plot(np_times, tnp_times, compiled_tnp_times, has_gpu, tnp_times_gpu)

# endregion
