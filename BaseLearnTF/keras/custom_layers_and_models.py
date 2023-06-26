# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: custom_layers_and_models
# CreateTime: 2023/6/25 19:43
# Summary: 通过子类化创建新的层和模型


import numpy as np
import tensorflow as tf
from tensorflow import keras

# region Layer 类：状态（权重）和部分计算的组合
from utils import join_path


class Linear(keras.layers.Layer):

    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype='float32'),
            trainable=True
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype='float32', ),
            trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)

assert linear_layer.weights == [linear_layer.w, linear_layer.b]


# 为层添加权重
class Linear_Add_Weight(keras.layers.Layer):

    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer='random_normal', dtype='float32', trainable=True
        )
        self.b = self.add_weight(
            shape=(units,), initializer='zeros', dtype='float32', trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# endregion


# region 层可以具有不可训练权重

class ComputeSum(keras.layers.Layer):

    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=True)

    def call(self, inputs):
        self.total.assign_add((tf.reduce_sum(inputs, axis=0)))
        return self.total


x = tf.ones((2, 2))
my_sum = ComputeSum(2)
y = my_sum(x)
print(y.numpy())
y = my_sum(x)
print(y.numpy())

print("weights:", len(my_sum.weights))  # 可训练权重
print("non-trainable weights:", len(my_sum.non_trainable_weights))  # 不可训练权重
print("trainable_weights:", my_sum.trainable_weights)


# endregion


# region 将权重创建推迟到得知输入的形状之后

class Linear_V2(keras.layers.Layer):

    def __init__(self, units=32, input_dim=32):
        super(Linear_V2, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer='random_normal', trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer='zeros', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class Linear_V3(keras.layers.Layer):

    def __init__(self, units=32):
        super(Linear_V3, self).__init__()
        self.units = units

    def build(self, input_shape):
        """在层的 build 方法中创建层权重"""
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer='random_normal', trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# endregion


# region 层可递归组合

class MLPBlock(keras.layers.Layer):

    def __init__(self):
        super(MLPBlock, self).__init__()
        self.liner_1 = Linear_V3(32)
        self.liner_2 = Linear_V3(32)
        self.liner_3 = Linear_V3(1)

    def call(self, inputs):
        x = self.liner_1(inputs)
        x = tf.nn.relu(x)
        x = self.liner_2(x)
        x = tf.nn.relu(x)
        return self.liner_3(inputs)


mlp = MLPBlock()
y = mlp(tf.ones(shape=(3, 64)))


# endregion


# region add_loss() 方法

class ActivityRegularizationLayer(keras.layers.Layer):

    def __init__(self, rate=1e-3):
        super(ActivityRegularizationLayer, self).__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs


class OuterLayer(keras.layers.Layer):

    def __init__(self):
        super(OuterLayer, self).__init__()
        self.activity_reg = ActivityRegularizationLayer(1e-2)  # 这些损失（包括由任何内部层创建的损失）可通过 layer.losses 取回
        # 此属性会在每个 __call__() 开始时重置到顶层，因此 layer.losses 始终包含在上一次前向传递过程中创建的损失值

    def call(self, inputs):
        return self.activity_reg(inputs)


layer = OuterLayer()
assert len(layer.losses) == 0

_ = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1

_ = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1


class OuterLayerWithKernelRegularizer(keras.layers.Layer):

    def __init__(self):
        super(OuterLayerWithKernelRegularizer, self).__init__()
        self.dense = keras.layers.Dense(
            32, kernel_regularizer=tf.keras.regularizers.l2(1e-3)
        )

    def call(self, inputs):
        return self.dense(inputs)


layer = OuterLayerWithKernelRegularizer()
layer(tf.zeros((1, 1)))
# loss 属性还包含为任何内部层的权重创建的正则化损失
print(layer.losses)


# endregion


# region add_metric() 方法
# 在训练过程中跟踪数量的移动平均值

class LogisticEndpoint(keras.layers.Layer):

    def __init__(self, name=None):
        super(LogisticEndpoint, self).__init__(name=name)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.accuracy_fn = keras.metrics.BinaryAccuracy()

    def call(self, targets, logits, sample_weights=None):
        # 计算训练时间损失值并使用`self.add_loss()`将其添加到层中
        loss = self.loss_fn(targets, logits, sample_weights)
        self.add_loss(loss)

        # 将精度记录为一个指标，并使用`self.add_metric()`将其添加到层中
        acc = self.accuracy_fn(targets, logits, sample_weights)
        self.add_metric(acc, name='accuracy')

        return tf.nn.softmax(logits)


layer = LogisticEndpoint()
targets = tf.ones((2, 2))
logits = tf.ones((2, 2))
y = layer(targets, logits)
print('layer.metrics: ', layer.metrics)
print('current accuracy value: ', float(layer.metrics[0].result()))

# add_metric 和 add_loss() 一样，这些指标也是通过 fit() 跟踪的
inputs = keras.Input(shape=(3,), name="inputs")
targets = keras.Input(shape=(10,), name="targets")
logits = keras.layers.Dense(10)(inputs)
predictions = LogisticEndpoint(name="predictions")(logits, targets)

model = keras.Model(inputs=[inputs, targets], outputs=predictions)
model.compile(optimizer="adam")

data = {
    "inputs": np.random.random((3, 3)),
    "targets": np.random.random((3, 10)),
}

model.fit(data)


# endregion


# region 可选择在层上启用序列化
# 如果需要将自定义层作为函数式模型的一部分进行序列化，您可以选择实现 get_config() 方法

class Linear_V4(keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear_V4, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        return {"units": self.units}


layer = Linear_V4(64)
config = layer.get_config()
print(config)
new_layer = Linear_V4.from_config(config)


class Linear_V5(keras.layers.Layer):

    def __init__(self, units=32, **kwargs):
        """ Layer 类的 __init__() 方法会接受一些关键字参数 """
        super(Linear_V5, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear_V5, self).get_config()
        config.update({"units": self.units})
        return config


layer = Linear_V5(64)
config = layer.get_config()
print(config)
new_layer = Linear_V5.from_config(config)


# 根据层的配置对层进行反序列化时需要更大的灵活性，还可以重写 from_config() 类方法
# def from_config(cls, config):
#     return cls(**config)

# endregion


# region call()

# call() 方法中的特权 training 参数
#  某些层，尤其是 BatchNormalization 层和 Dropout 层，在训练和推断期间具有不同的行为。对于此类层，标准做法是在 call() 方法中公开 training（布尔）参数
#  公开此参数，可以启用内置的训练和评估循环（例如 fit()）以在训练和推断中正确使用层
class CustomDropout(keras.layers.Layer):

    def __init__(self, rate, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs


# call() 方法中的特权 mask 参数
#  出现在所有 Keras RNN 层中。掩码是布尔张量（在输入中每个时间步骤对应一个布尔值），用于在处理时间序列数据时跳过某些输入时间步骤
#  当先前的层生成掩码时，Keras 会自动将正确的 mask 参数传递给 __call__()（针对支持它的层）。掩码生成层是配置了 mask_zero=True 的 Embedding 层和 Masking 层

# endregion


# region 汇总：端到端示例
# 实现一个变分自动编码器 (VAE)，并用 MNIST 数字对其进行训练

class Sampling(keras.layers.Layer):
    """使用(z_mean, z_log_var)对z进行采样，z是一个编码数字的向量"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(keras.layers.Layer):
    """将MNIST数字映射为三元组(z_mean, z_log_var, z)。"""

    def __init__(self, latent_dim=32, intermediate_dim=64, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = keras.layers.Dense(intermediate_dim, activation='relu')
        self.dense_mean = keras.layers.Dense(latent_dim)
        self.dense_log_var = keras.layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(keras.layers.Layer):
    """将编码的数字向量z转换为可读的数字"""

    def __init__(self, original_dim, intermediate_dim=64, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = keras.layers.Dense(intermediate_dim, activation='relu')
        self.dense_output = keras.layers.Dense(original_dim, activation='sigmoid')

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class VariationalAutoEncoder(keras.Model):
    """将编码器和解码器组合成一个端到端的模型进行训练"""

    def __init__(
            self,
            original_dim,
            intermediate_dim=64,
            latent_dim=32,
            name='autoencoder',
            **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed


# 在 MNIST 上编写一个简单的训练循环

original_dim = 784

#  VAE 是 Model 的子类
vae = VariationalAutoEncoder(original_dim, 64, 32)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()

loss_metric = tf.keras.metrics.Mean()

path = join_path.keras_data_path.helloworld_keras_mnist_data_path
path = path.replace('BaseLearnTF\\', '')
(x_train, _), _ = tf.keras.datasets.mnist.load_data(path)
x_train = x_train.reshape(60000, 784).astype('float32') / 255

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

epochs = 2

for epoch in range(epochs):
    print('Start of epoch', epoch)

    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            reconstructed = vae(x_batch_train)

            loss = mse_loss_fn(x_batch_train, reconstructed)
            loss += sum(vae.losses)

        grads = tape.gradient(loss, vae.trainable_weights)
        optimizer.apply_gradients(zip(grads, vae.trainable_weights))

        loss_metric(loss)

        if step % 100 == 0:
            print("step %d: mean loss = %.4f" % (step, loss_metric.result()))

# VAE 是 Model 的子类，它具有内置的训练循环。也可以用以下方式训练它
vae = VariationalAutoEncoder(784, 64, 32)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=2, batch_size=64)

# endregion


# region 超越面向对象的开发：函数式 API
original_dim = 784
intermediate_dim = 64
latent_dim = 32

#  定义 encoder model
original_inputs = keras.Input(shape=(original_dim,), name='encoder_input')
x = keras.layers.Dense(intermediate_dim, activation='relu')(original_inputs)
z_mean = keras.layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = keras.layers.Dense(latent_dim, name='z_log_var')(x)
z = Sampling()((z_mean, z_log_var))
encoder = keras.Model(inputs=original_inputs, outputs=z, name='encoder')

# 定义 decoder model
latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
x = keras.layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = keras.layers.Dense(original_dim, activation='sigmoid')(x)
decoder = keras.Model(inputs=latent_inputs, outputs=outputs, name='decoder')

# 定义 vae model
outputs = decoder(z)
vae = keras.Model(inputs=original_inputs, outputs=outputs, name='vae')

# 添加KL散度正则化损失
kl_loss = -0.5 * tf.reduce_mean(
    z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
)
vae.add_loss(kl_loss)

# 训练
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
vae.compile(optimizer, loss=keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=3, batch_size=64)

# endregion
