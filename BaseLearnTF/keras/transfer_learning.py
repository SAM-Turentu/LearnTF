# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: transfer_learning
# CreateTime: 2023/6/30 15:11
# Summary: 迁移学习和微调


import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow_datasets as tfds
from utils import join_path

# 迁移学习最常见的形式是以下工作流：
#  1. 从之前训练的模型中获取层。
#  2. 冻结这些层，以避免在后续训练轮次中破坏它们包含的任何信息。
#  3. 在已冻结层的顶部添加一些新的可训练层。这些层会学习将旧特征转换为对新数据集的预测。
#  4. 在您的数据集上训练新层。

# 最后一个可选步骤是微调
#  包括解冻上面获得的整个模型（或模型的一部分），然后在新数据上以极低的学习率对该模型进行重新训练

#  Keras trainable API


# region 冻结层：了解 trainable 特性

# Dense 层具有 2 个可训练权重


layer = keras.layers.Dense(3)
layer.build((None, 4))  # 创建一个 weights
print("weights:", len(layer.weights))
print("trainable_weights:", len(layer.trainable_weights))
print("non_trainable_weights:", len(layer.non_trainable_weights))

# BatchNormalization 层具有 2 个可训练权重和 2 个不可训练权重
layer = keras.layers.BatchNormalization()
layer.build((None, 4))  # 创建一个 weights
print("weights:", len(layer.weights))
print("trainable_weights:", len(layer.trainable_weights))
print("non_trainable_weights:", len(layer.non_trainable_weights))

# 将 trainable 设置为 False
layer = keras.layers.Dense(3)
layer.build((None, 4))  # 创建一个 weights
layer.trainable = False  # 冻结 layer
print("weights:", len(layer.weights))
print("trainable_weights:", len(layer.trainable_weights))
print("non_trainable_weights:", len(layer.non_trainable_weights))

# 当可训练权重变为不可训练时，它的值在训练期间不再更新
layer1 = keras.layers.Dense(3, activation="relu")
layer2 = keras.layers.Dense(3, activation="sigmoid")
model = keras.Sequential([keras.Input(shape=(3,)), layer1, layer2])

# 冻结 layer1
layer1.trainable = False
initial_layer1_weights_values = layer1.get_weights()  # 保留layer1的权重副本

model.compile(optimizer="adam", loss="mse")
model.fit(np.random.random((2, 3)), np.random.random((2, 3)))

# 检查layer1的权重在训练期间没有改变
final_layer1_weights_values = layer1.get_weights()
np.testing.assert_allclose(
    initial_layer1_weights_values[0], final_layer1_weights_values[0]
)
np.testing.assert_allclose(
    initial_layer1_weights_values[1], final_layer1_weights_values[1]
)

# endregion


# region trainable 特性的递归设置
# 在模型或具有子层的任何层上设置 trainable = False，则所有子层也将变为不可训练
inner_model = keras.Sequential([
    keras.Input(shape=(3,)),
    keras.layers.Dense(3, activation="relu"),
    keras.layers.Dense(3, activation="relu"),
])

model = keras.Sequential([
    keras.Input(shape=(3,)),
    inner_model,
    keras.layers.Dense(3, activation="sigmoid")
])

model.trainable = False

#  model 所有层都被冻结
assert inner_model.trainable == False
assert inner_model.layers[0].trainable == False  # `trainable`是递归传播的

# endregion


# region 典型的迁移学习工作流

# 第一种工作流（在 Keras 中实现典型的迁移学习工作流）
#  1. 实例化一个基础模型并加载预训练权重
#  2. 通过设置 trainable = False 冻结基础模型中的所有层
#  3. 根据基础模型中一个（或多个）层的输出创建一个新模型
#  4. 在您的新数据集上训练新模型

# 第二种工作流（更轻量的工作流）：
#  1. 实例化一个基础模型并加载预训练权重
#  2. 通过该模型运行新的数据集，并记录基础模型中一个（或多个）层的输出。这一过程称为特征提取
#  3. 使用该输出作为新的较小模型的输入数据

# 第二种工作流优势：只需在自己的数据上运行一次基础模型，而不是每个训练周期都运行一次，速度更快，开销也更低
# 第二种工作流存在的问题：不允许在训练期间动态修改新模型的输入数据（在进行数据扩充时，这种修改必不可少）


# 第一种工作流
# 1. 实例化一个基础模型
base_model = keras.applications.Xception(
    weights='imagenet',  # 在 ImageNet 上预训练的权重
    input_shape=(150, 150, 3),
    include_top=False  # 不要在顶部包含 ImageNet 分类器
)
# 2. 冻结该基础模型
base_model.trainable = False

# 3. 根据基础模型创建一个新模型
inputs = keras.Input(shape=(150, 150, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

# 4. 在新数据上训练该模型
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()]
)
# model.fit(new_dataset, epochs=20, callbacks=..., validation_data=...)

# endregion


# region 微调

# 解冻
base_model.trainable = True

# 对任何内层的`trainable`属性进行任何更改后，需要重新编译模型，这样更改就会被考虑在内
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()]
)
# model.fit(new_dataset, epochs=10, callbacks=..., validation_data=...)

# endregion


# region 使用自定义训练循环进行迁移学习和微调
# 如果使用自己的低级训练循环而不是 fit()，则工作流基本保持不变
#   在应用梯度更新时，您应当注意只考虑清单 model.trainable_weights

# 创建基础model，并下载数据 download   xception_weights_tf_dim_ordering_tf_kernels_notop.h5
base_model = keras.applications.Xception(
    weights='imagenet',
    input_shape=(150, 150, 3),
    include_top=False
)
# 冻结
base_model.trainable = False

# 在顶部创建新模型
inputs = keras.Input(shape=(150, 150, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam()

new_dataset = []  # 数据集
for inputs, targets in new_dataset:
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss_value = loss_fn(targets, predictions)

    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

# endregion


# region 端到端示例：基于 Dogs vs. Cats 数据集微调图像分类模型


# region 获取数据

tfds.disable_progress_bar()  # 禁用 Tqdm 进度条

path = join_path.tf_datasets.tf_datasets

train_ds, validation_ds, test_ds = tfds.load(
    "cats_vs_dogs",
    # 40% 用于训练, 保留10%用于验证，10%用于测试
    split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
    data_dir=path,
    download=False,
    as_supervised=True,  # Include labels
)

print('Number of training samples: ', tf.data.experimental.cardinality(train_ds))
print('Number of validation samples: ', tf.data.experimental.cardinality(validation_ds))
print('Number of test samples: ', tf.data.experimental.cardinality(test_ds))

# region 显示训练集中前9张图，具有不同的大小
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_ds.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(int(label))  # 1 是 狗， 0 是 猫
    plt.axis('off')
plt.show()
# endregion

# endregion

# region 标准化数据

size = (150, 150)
train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))

# 对数据进行批处理并使用缓存和预提取来优化加载速度
batch_size = 32
train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)

# 使用随机数据扩充
#   没有较大的图像数据集时，通过将随机但现实的转换（例如随机水平翻转或小幅随机旋转）应用于训练图像
#   人为引入样本多样性
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip('horizontal'),
    keras.layers.RandomRotation(0.1)
])

# region 查看图像随机翻转
for images, labels in train_ds.take(1):
    plt.figure(figsize=(10, 10))
    first_image = images[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(
            tf.expand_dims(first_image, 0), training=True
        )
        plt.imshow(augmented_image[0].numpy().astype('int32'))
        plt.title(int(labels[0]))
        plt.axis('off')
plt.show()
# endregion

# endregion

# region 构建模型

#   1. 添加 Rescaling 层以将输入值（最初在 [0, 255] 范围内）缩放到 [-1, 1] 范围
#   2. 在分类层之前添加一个 Dropout 层，以进行正则化
#   3. 在调用基础模型时传递 training=False，使其在推断模式下运行（这样，即使在我们解冻基础模型以进行微调后，batchnorm 统计信息也不会更新）

base_model = keras.applications.Xception(
    weights='imagenet',
    input_shape=(150, 150, 3),
    include_top=False
)
base_model.trainable = False  # 冻结
# 在顶部创建新模型
inputs = keras.Input(shape=(150, 150, 3))
x = data_augmentation(inputs)  # 应用随机数据增强

scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(x)

x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()

# endregion

# region 训练顶层
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 20
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

# endregion

# region 对整个模型进行一轮微调
# 解冻基础模型，并以较低的学习率端到端地训练整个模型

base_model.trainable = True  # 解冻
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 10
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

# endregion

# endregion
