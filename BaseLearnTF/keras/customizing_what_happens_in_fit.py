# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: customizing_what_happens_in_fit
# CreateTime: 2023/6/28 9:39
# Summary: 自定义 Model.fit 的内容


import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils import join_path


# region 第一个简单的示例

# 在 train_step 方法的主体中实现了定期的训练更新
#  通过 self.compiled_loss 计算损失，它会封装传递给 compile() 的损失函数

#  调用 self.compiled_metrics.update_state(y, y_pred) 来更新在 compile() 中传递的指标的状态，
#  并在最后从 self.metrics 中查询结果以检索其当前值


class CustomModel(keras.Model):

    def train_step(self, data):
        """ """
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # 计算损失值(损失函数在`compile()`中配置)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # 更新权重
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # 更新在 compile() 中传递的指标的状态(包括跟踪损失的指标)
        self.compiled_metrics.update_state(y, y_pred)
        # 将度量名称映射到当前值的dict
        return {m.name: m.result() for m in self.metrics}


# 构造一个 CustomModel
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=3)

history = model.history.history
print(history)

# endregion


# region 在更低级别上操作

# 使用 compile() 配置优化器

loss_tracker = keras.metrics.Mean(name='loss')
mae_metric = keras.metrics.MeanAbsoluteError(name='mae')


class CustomModel_V2(keras.Model):

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            loss = keras.losses.mean_squared_error(y, y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        loss_tracker.update_state(loss)
        mae_metric.update_state(y, y_pred)
        return {'loss': loss_tracker.result(), 'mae': mae_metric.result()}

    @property
    def metrics(self):
        # 列出`Metric`对象，以便在每个epoch开始时或在`evaluate()`开始时自动调用`reset_states()`
        # 如果没有实现这个属性，必须调用`reset_states()`
        # 否则，调用 result() 会返回自训练开始以来的平均值
        return [loss_tracker, mae_metric]


inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel_V2(inputs, outputs)

# 这里不传递损失或指标
model.compile(optimizer='adam')

x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=5)

history = model.history.history
print(history)


# endregion


# region 支持 sample_weight 和 class_weight
# 从 data 参数中解包 sample_weight
# 将其传递给 compiled_loss 和 compiled_metrics
#   （当然，如果您不依赖 compile() 来获取损失和指标，也可以手动应用）

class CustomModel_V3(keras.Model):

    def train_step(self, data):
        # 解包数据
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # 计算损失
            # 损失函数在`compile()`中配置
            loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # 更新 weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # 更新 metrics
        # Metrics 在`compile()`中配置
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}


inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel_V3(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
sw = np.random.random((1000, 1))
model.fit(x, y, sample_weight=sw, epochs=3)

history = model.history.history
print(history)


# endregion


# region 提供您自己的评估步骤

class CustomModel_V4(keras.Model):

    def train_step(self, data):
        x, y = data
        y_pred = self(x, training=True)

        self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred)
        # 返回一个将度量名称映射到当前值的dict
        # 包括损失值(在self.metrics中跟踪)
        return {m.name: m.result() for m in self.metrics}


inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel_V4(inputs, outputs)
model.compile(loss='mse', metrics=['mae'])

# 自定义的 train_step 进行评估
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.evaluate(x, y)

history = model.history.history
print(history)

# endregion


# region 端到端 GAN 示例
#  1. 生成 28x28x1 图像的生成器网络
#  2. 将 28x28x1 图像分为两类（“fake”和“real”）的鉴别器网络
#  3. 分别用于两个网络的优化器
#  4. 训练鉴别器的损失函数


# 创建一个鉴别器
discriminator = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.GlobalMaxPooling2D(),
        keras.layers.Dense(1)
    ],
    name='discriminator'
)

# 创建生成器
latent_dim = 128
generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        keras.layers.Dense(7 * 7 * 128),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Reshape((7, 7, 128)),
        keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')
    ],
    name='generator'
)


class GAN(keras.Model):

    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # 在潜空间中对随机点进行采样
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        # 将它们解码为假图像
        generated_images = self.generator(random_latent_vectors)
        # 将它们与真实图像相结合
        combined_images = tf.concat([generated_images, real_images], axis=0)
        # generated_images：设为1, real_images：设为0
        # 组合标签，区分真假图像
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # 为标签添加随机噪声
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # 训练判别器
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # 在潜空间中对随机点进行采样
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # 组合标有“所有真实图像”的标签
        misleading_labels = tf.zeros((batch_size, 1))

        # 训练生成器(注意我们应该* *不* *更新判别器的权重)
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {'d_loss': d_loss, 'g_loss': g_loss}


batch_size = 64
path = join_path.keras_data_path.helloworld_keras_mnist_data_path
path = path.replace('BaseLearnTF\\', '')
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data(path)
all_digits = np.concatenate([x_train, x_test])
all_digits = all_digits.astype('float32') / 255.
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices(all_digits)
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    g_optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
)
gan.fit(dataset.take(100), epochs=1)

history = gan.history.history
print(history)

# endregion
