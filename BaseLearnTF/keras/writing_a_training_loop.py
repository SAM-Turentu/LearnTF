# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: writing_a_training_loop
# CreateTime: 2023/6/28 16:52
# Summary: 从头编写训练循环


import os.path
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from utils import join_path

# 对训练和评估进行低级别控制，则应当从头开始编写自己的训练和评估循环

# region 使用 GradientTape：第一个端到端示例

# 在 GradientTape 作用域内调用模型使您可以检索层的可训练权重相对于损失值的梯度
#  利用优化器实例，您可以使用上述梯度来更新这些变量（可以使用 model.trainable_weights 检索这些变量）

inputs = keras.Input(shape=(784,), name='digits')
x1 = layers.Dense(64, activation='relu')(inputs)
x2 = layers.Dense(64, activation='relu')(x1)
outputs = layers.Dense(10, name='predictions')(x2)
model = keras.Model(inputs=inputs, outputs=outputs)
# 实例化 优化器
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# 实例化 损失函数
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

batch_size = 64
path = join_path.keras_data_path.helloworld_keras_mnist_data_path
path = path.replace('BaseLearnTF\\', '')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path)
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# 准备训练数据
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# 准备验证数据
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)

# 1. 我们打开一个遍历各周期的 for 循环
# 2. 对于每个周期，我们打开一个分批遍历数据集的 for 循环
# 3. 对于每个批次，我们打开一个 GradientTape() 作用域
# 4. 在此作用域内，我们调用模型（前向传递）并计算损失
# 5. 在作用域之外，我们检索模型权重相对于损失的梯度
# 6. 最后，我们根据梯度使用优化器来更新模型的权重

epochs = 2
for epoch in range(epochs):
    print('\nStart of epoch ', epoch)
    # 遍历数据集的批次
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # 对于每个批次，我们打开一个 GradientTape() 作用域
        with tf.GradientTape() as tape:
            # 调用模型（前向传递）并计算损失
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)

        # 使用梯度带自动检索可训练变量相对于损失的梯度
        grads = tape.gradient(loss_value, model.trainable_weights)

        # 根据梯度使用优化器来更新模型的权重，以最小化损失
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if step % 200 == 0:
            print('Training loss (for one batch) at step %d: %.4f' % (step, float(loss_value)))
            print(f'Seen so far: {(step + 1) * batch_size} samples')

# endregion


# region 指标的低级处理

# 1. 在循环开始时实例化指标
# 2. 在每个批次后调用 metric.update_state()
# 3. 当需要显示指标的当前值时，调用 metric.result()
# 4. 当需要清除指标的状态（通常在周期结束）时，调用 metric.reset_states()

inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, name='predictions')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

optimizer = keras.optimizers.SGD(learning_rate=1e-3)  # 实例化 优化器
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # 实例化 损失函数

# 准备指标(metric)
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

epochs = 2
for epoch in range(epochs):
    print('\nStart of epoch ', epoch)
    start_time = time.time()

    # 遍历数据集的批次
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # 更新 训练 metric
        train_acc_metric.update_state(y_batch_train, logits)

        if step % 200 == 0:
            print('Training loss (for one batch) at step %d: %.4f' % (step, float(loss_value)))
            print(f'Seen so far: {(step + 1) * batch_size} samples')

    # 在每个epoch结束时显示 metric
    train_acc = train_acc_metric.result()
    print('Train acc over epoch: %.4f' % float(train_acc))
    # 在每个epoch结束时重置训练 metric
    train_acc_metric.reset_state()

    # 在每个循环结束时运行一个验证循环
    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val, training=True)
        # 更新 val metric
        val_acc_metric.update_state(y_batch_val, val_logits)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_state()
    print('Validation acc: %.4f' % float(val_acc))
    print('Time taken: %.2fs' % (time.time() - start_time))

print(111)


# endregion


# region 使用 tf.function 加快训练步骤速度

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value


@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)


epochs = 2
for epoch in range(epochs):
    print('\nStart of epoch ', epoch)
    start_time = time.time()

    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(x_batch_train, y_batch_train)

        if step % 200 == 0:
            print('Training loss (for one batch) at step %d: %.4f' % (step, float(loss_value)))
            print(f'Seen so far: {(step + 1) * batch_size} samples')

    train_acc = train_acc_metric.result()
    print('Training acc over epoch: %.4f' % float(train_acc))

    train_acc_metric.reset_state()

    for x_batch_val, y_batch_val in val_dataset:
        test_step(x_batch_val, y_batch_val)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_state()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))

print(222)


# 比上一种方法，速度快了很多

# endregion


# region 对模型跟踪的损失进行低级处理
# 层和模型以递归方式跟踪调用 self.add_loss(value) 的层在前向传递过程中创建的任何损失
# 可在前向传递结束时通过属性 model.losses 获得标量损失值的结果列表

class ActivityRegularizationLayer(layers.Layer):

    def call(self, inputs):
        self.add_loss(1e-2 * tf.reduce_sum(inputs))
        return inputs


inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu')(inputs)
x = ActivityRegularizationLayer()(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, name='predictions')(x)

model = keras.Model(inputs, outputs)


@tf.function
def train_step_v2(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
        loss_value += sum(model.losses)  # 通过属性 model.losses 获得标量损失值的结果列表

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value


# endregion


# region 总结

# GAN 训练循环如下所示：
# 1.训练判别器
#   a.在隐空间中对一批随机点采样
#   b.通过“生成器”模型将这些点转换为虚假图像
#   c.获取一批真实图像，并将它们与生成的图像组合
#   d.训练“判别器”模型以对生成的图像与真实图像进行分类
# 2.训练生成器
#   a.在隐空间中对随机点采样
#   b.通过“生成器”网络将这些点转换为虚假图像
#   c.获取一批真实图像，并将它们与生成的图像组合
#   d.训练“生成器”模型以“欺骗”判别器，并将虚假图像分类为真实图像

# 创建用于区分虚假数字和真实数字的 判别器
discriminator = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name="discriminator",
)
discriminator.summary()

# 创建一个生成器网络（它可以将隐向量转换成形状为 (28, 28, 1)）
latent_dim = 128
generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        # We want to generate 128 coefficients to reshape into a 7x7x128 map
        layers.Dense(7 * 7 * 128),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",
)

d_optimizer = keras.optimizers.Adam(learning_rate=0.0003)  # 判别器实例化一个优化器
g_optimizer = keras.optimizers.Adam(learning_rate=0.0004)  # 生成器实例化

loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)  # 损失函数


@tf.function
def train_step_v3(real_images):
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    generated_images = generator(random_latent_vectors)
    combined_images = tf.concat([generated_images, real_images], axis=0)

    labels = tf.concat(
        [tf.ones((batch_size, 1)), tf.zeros((real_images.shape[0], 1))], axis=0
    )
    labels += 0.05 * tf.random.uniform(labels.shape)

    with tf.GradientTape() as tape:
        predictions = discriminator(combined_images)
        d_loss = loss_fn(labels, predictions)

    grads = tape.gradient(d_loss, discriminator.trainable_weights)
    d_optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))

    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    misleading_labels = tf.zeros((batch_size, 1))

    with tf.GradientTape() as tape:
        predictions = discriminator(generator(random_latent_vectors))
        g_loss = loss_fn(misleading_labels, predictions)

    grads = tape.gradient(g_loss, generator.trainable_weights)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_weights))
    return d_loss, g_loss, generated_images


# 通过在各个图像批次上重复调用 train_step 来训练 GAN

batch_size = 64
path = join_path.keras_data_path.helloworld_keras_mnist_data_path
path = path.replace('BaseLearnTF\\', '')
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data(path)
all_digits = np.concatenate([x_train, x_test])
all_digits = all_digits.astype('float32') / 255.
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices(all_digits)
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

epochs = 1  # 至少需要20次迭代才能生成漂亮的数字
save_dir = './writing_a_training_loop'

for epoch in range(epochs):

    for step, real_images in enumerate(dataset):
        # 在一批真实图像上训练鉴别器和生成器
        d_loss, g_loss, generated_images = train_step_v3(real_images)

        if step % 200 == 0:
            print('discriminator loss at step %d: %.4f' % (step, float(d_loss)))
            print('adversarial loss at step %d: %.4f' % (step, float(g_loss)))

            img = keras.preprocessing.image.array_to_img(
                generated_images[0] * 255., scale=False
            )
            # 保存一个生成的图像, 虚假 MNIST 数字
            img.save(os.path.join(save_dir, 'generated_img' + str(step) + '.png'))

        if step > 10:
            break

print(333)

# endregion
