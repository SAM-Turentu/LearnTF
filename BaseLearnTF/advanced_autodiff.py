# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: advanced_autodiff
# CreateTime: 2023/6/13 21:30
# Summary: 高级自动微分


import tensorflow as tf
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['figure.figsize'] = (8, 6)

# region 控制梯度记录

x = tf.Variable(2., name='x')
y = tf.Variable(3., name='y')

# 停止记录
with tf.GradientTape() as t:
    x_sq = x * x
    with t.stop_recording():  # 停止记录梯度，暂时挂起
        y_sq = y * y
    z = x_sq + y_sq

grad = t.gradient(z, {'x': x, 'y': y})
print('dz/dx: ', grad['x'])
print('dz/dy: ', grad['y'])

reset = True
with tf.GradientTape() as t:
    y_sq = y * y
    if reset:
        t.reset()
    z = x * x + y_sq

grad = t.gradient(z, {'x': x, 'y': y})  # dz_dy = 2y; dz_dx = 2x
print('dz/dx: ', grad['x'])
print('dz/dy: ', grad['y'])  # reset = False 时，dz/dy = 2*3=6

# endregion


# region 精确停止梯度流

with tf.GradientTape() as t:
    y_sq = y * y
    z = x * x + tf.stop_gradient(y_sq)  # 可以用来阻止梯度沿着特定路径流动，而不需要访问条带本身

grad = t.gradient(z, {'x': x, 'y': y})
print('dz/dx: ', grad['x'])
print('dz/dy: ', grad['y'])


# endregion


# region 自定义梯度
# 1. 正在编写的新运算没有定义的梯度;  可以使用 tf.RegisterGradient 设置
# 后面三种可以使用 tf.custom_gradient
# 2. 默认计算在数值上不稳定
# 3. 从前向传递缓存开销大的计算
# 4. 修改一个值，而不修改梯度（例：tf.clip_by_value  tf.math.round）


@tf.custom_gradient
def clip_gradients(y):
    def backward(dy):
        # tf.clip_by_norm 计算公式： t * clip_norm / l2norm(t)
        # l2norm: l2 范数 指 向量各元素的平方和然后平方根
        # todo ## 此处传入的 y 不是向量；所以 l2norm = 1 (暂时这么理解吧)
        return tf.clip_by_norm(dy, 0.5)  # 应用于中间梯度

    return y, backward


# tf.Variable([1.]) # 这个才是向量

v = tf.Variable(2., name='v')
with tf.GradientTape() as t:
    output = clip_gradients(v * v)
print(t.gradient(output, v))  # result = (v **2) * 0.5 / （(v **2)**2）的平方根

print('SavedModel 中的自定义梯度')


class MyModule(tf.Module):

    @tf.function(input_signature=[tf.TensorSpec(None)])
    def call_custom_grad(self, x):
        return clip_gradients(x)


model = MyModule()

tf.saved_model.save(
    model,
    'advanced_autodiff/saved_model',
    # experimental_custom_gradients=True 可以将自定义梯度保存到 savemodel
    options=tf.saved_model.SaveOptions(experimental_custom_gradients=True)
)

# experimental_custom_gradients=False，重新运行时产生相同的结果，因为梯度注册表依然包含 call_custom_op
#  如果没有自定义梯度 保存后 重新运行，GradientTape 会报错

v = tf.Variable(2., name='v')

loaded = tf.saved_model.load('advanced_autodiff/saved_model')
with tf.GradientTape() as t:
    output = loaded.call_custom_grad(v * v)
print(t.gradient(output, v))

# endregion


# region 多个条带
x0 = tf.Variable(0., name='x0')
x1 = tf.Variable(0., name='x1')

with tf.GradientTape() as tape0, tf.GradientTape() as tape1:
    tape0.watch(x0)
    tape1.watch(x1)

    y0 = tf.math.sin(x0)
    y1 = tf.nn.sigmoid(x1)  # 1 / (1 + e**(-x))          ； n=x

    y = y0 + y1

    ys = tf.reduce_sum(y)

tape0.gradient(ys, x0).numpy()  # dys_dx0 = cos(x) == cos(0) = 1
tape1.gradient(ys, x1).numpy()

# 高阶梯度
x = tf.Variable(1.)
with tf.GradientTape() as t2:
    with tf.GradientTape() as t1:
        y = x * x * x

    dy_dx = t1.gradient(y, x)  # 3x**2
d2y_d2x = t2.gradient(dy_dx, x)  # 3*2*x => 6x；二次导数
# 这种模式不能通用于生成黑塞矩阵（参考雅可比矩阵）

print('dy_dx: ', dy_dx.numpy())
print('d2y_d2x: ', d2y_d2x.numpy())

# 输入梯度正则化（增强模型收到对抗样本影响）
#  会尝试将输入梯度的幅度最小化，对应的输出变化应该也较小
#  1. 使用内条带计算输出相对于输入的梯度
#  3. 计算输入梯度的幅度
#  2. 计算幅度相对于模型的梯度

x = tf.random.normal([7, 5])
layer = tf.keras.layers.Dense(10, activation=tf.nn.relu)

with tf.GradientTape() as t2:
    # 内条带只接受关于 输入的梯度，而不是变量
    with tf.GradientTape(watch_accessed_variables=False) as t1:
        t1.watch(x)
        y = layer(x)
        out = tf.reduce_sum(layer(x) ** 2)
    g1 = t1.gradient(out, x)
    # 计算输入梯度
    g1_mag = tf.norm(g1)  # 默认 l2 范数

# 计算该幅度相对于模型的梯度
dg1_mag = t2.gradient(g1_mag, layer.trainable_variables)

print([var.shape for var in dg1_mag])

# endregion


# region 雅可比矩阵 (向量值函数的梯度)
#  每行都包含其中一个向量元素的梯度

# region 标量源
x = tf.linspace(-10., 10., 200 + 1)
delta = tf.Variable(0.)

with tf.GradientTape() as tape:
    y = tf.nn.sigmoid(x + delta)

dy_dx = tape.jacobian(y, delta)  # 计算雅可比矩阵

print(y.shape)
print(dy_dx.shape)

plt.plot(x.numpy(), y, label='y')
plt.plot(x.numpy(), dy_dx, label='dy/dx')
plt.legend()
_ = plt.xlabel('x')
plt.show()

# endregion


# region 张量源

x = tf.random.normal([7, 5])
layer = tf.keras.layers.Dense(10, activation=tf.nn.relu)

with tf.GradientTape(persistent=True) as tape:
    y = layer(x)

print('y.shape: ', y.shape)
print('layer.kernel.shape: ', layer.kernel.shape)

# 将 y 和 layer 两个形状连在一起就是 输出相对于内核的 雅可比矩阵的形状
j = tape.jacobian(y, layer.kernel)
print('j.shape: ', j.shape)

g = tape.gradient(y, layer.kernel)
print('g.shape: ', g.shape)

j_sum = tf.reduce_sum(j, axis=[0, 1])  # 总和的梯度
delta = tf.reduce_max(abs(g - j_sum)).numpy()
assert delta < 1e-3
print('delta: ', delta)

# 黑塞矩阵
#  可以使用 tf.GradientTape.jacobian 方法进行构建

#  黑塞矩阵包含N**2 个参数。
x = tf.random.normal([7, 5])
layer1 = tf.keras.layers.Dense(8, activation=tf.nn.relu)
layer2 = tf.keras.layers.Dense(6, activation=tf.nn.relu)

with tf.GradientTape() as t2:
    with tf.GradientTape() as t1:
        x = layer1(x)
        x = layer2(x)
        loss = tf.reduce_mean(x ** 2)

    g = t1.gradient(loss, layer1.kernel)

h = t2.jacobian(g, layer1.kernel)  # 黑塞矩阵

print(f'layer.kernel.shape: {layer1.kernel.shape}')
print(f'h.shape: {h.shape}')

# 将次黑塞矩阵用于 牛顿方法，先将其轴展平为矩阵，然后将梯度展平为向量
n_params = tf.reduce_prod(layer1.kernel.shape)
g_vec = tf.reshape(g, [n_params, 1])
h_mat = tf.reshape(h, [n_params, n_params])


# 黑塞矩阵应当对称
def imshow_zero_center(image, **kwargs):
    lim = tf.reduce_max(abs(image))
    plt.imshow(image, vmin=-lim, vmax=lim, cmap='seismic', **kwargs)
    plt.colorbar()


imshow_zero_center(h_mat)
plt.show()

eps = 1e-3
eye_eps = tf.eye(h_mat.shape[0]) * eps

# X(k+1) = X(k) - (∇²f(X(k)))^-1 @ ∇f(X(k))
# h_mat = ∇²f(X(k))
# g_vec = ∇f(X(k))
update = tf.linalg.solve(h_mat + eye_eps, g_vec)

_ = layer1.kernel.assign_sub(tf.reshape(update, layer1.kernel.shape))

# endregion

# region 批量雅可比矩阵

x = tf.random.normal([7, 5])
layer1 = tf.keras.layers.Dense(8, activation=tf.nn.relu)
layer2 = tf.keras.layers.Dense(6, activation=tf.nn.relu)

with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
    tape.watch(x)
    y = layer1(x)
    y = layer2(y)

print('y.shape: ', y.shape)

j = tape.jacobian(y, x)
print('j.shape: ', j.shape)
# 如果堆栈中各项的梯度相互独立，那么此张量的每一个 (batch, batch) 切片都是对角矩阵
imshow_zero_center(j[:, 0, :, 0])
plt.title('A (batch, batch) slice')
plt.show()


def plot_as_patches(j):
    j = tf.transpose(j, [1, 0, 3, 2])
    lim = tf.reduce_max(abs(j))
    j = tf.pad(j, [[0, 0], [1, 1], [0, 0], [1, 1]], constant_values=-lim)
    s = j.shape
    j = tf.reshape(j, [s[0] * s[1], s[2] * s[3]])
    imshow_zero_center(j, extent=[-0.5, s[2] - 0.5, s[0] - 0.5, -0.5])


plot_as_patches(j)
plt.title('All (batch, batch) slices are diagonal')
plt.show()

# endregion

# endregion
