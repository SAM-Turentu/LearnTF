# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: data
# CreateTime: 2023/7/2 17:19
# Summary: tf.data：构建 TensorFlow 输入流水线


import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import ndimage

from utils import join_path
from utils.utils import tools

np.set_printoptions(precision=4)

# 通过两种不同的方式创建数据集：
#   1. 数据源从存储在内存中或存储在一个或多个文件中的数据构造 Dataset
#   2. 数据转换从一个或多个 tf.data.Dataset 对象构造数据集

# region 基本机制

# Dataset 是一个 python 的可迭代对象，可以用for,next遍历
dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
for elem in dataset:
    print(elem.numpy())

# a = [8, 3, 0, 8, 2, 1]
# print(sum(a))

# 计算整数数据集的和
#   利用 reduce 转换来使用数据集元素，从而减少所有元素以生成单个结果
dataset.reduce(0, lambda state, value: state + value).numpy()

# endregion


# region 数据集结构

# Dataset.element_spec 属性允许检查每个元素组件的类
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
print(dataset1.element_spec)

dataset2 = tf.data.Dataset.from_tensor_slices(
    (tf.random.uniform([4, 10]),
     tf.random.uniform([4, 100], maxval=100, dtype=tf.int32))
)
print(dataset2.element_spec)

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.element_spec)

dataset4 = tf.data.Dataset.from_tensors(tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]))
print(dataset4.element_spec)
print(dataset4.element_spec.value_type)  # 使用value_type查看元素规范所表示的值的类型

dataset1 = tf.data.Dataset.from_tensor_slices(
    tf.random.uniform([4, 10], minval=1, maxval=10, dtype=tf.int32)  # 4 * 10 的矩阵，最小值1，最大值9
)

for data in dataset1:
    print(data.numpy())

for a, (b, c) in dataset3:
    print('shapes: {a.shape}, {b.shape}, {c.shape}'.format(a=a, b=b, c=c))
print(a)
print(b)
print(c)

# endregion


# region 读取输入数据

# region 使用 NumPy 数组

(images, labels), (test_images, test_labels) = tools.load_fashion_data()
images = images / 255
dataset = tf.data.Dataset.from_tensor_slices((images, labels))  # 将输入数据装入内存


# endregion


# region 使用 Python 生成器

#   方式比较简便，但它的可移植性和可扩缩性有限,且仍受 Python GIL 约束
def count(stop):
    i = 0
    while i < stop:
        yield i
        i += 1


for n in count(5):
    print(n)

# Dataset.from_generator 构造函数会将 Python 生成器转换为具有完整功能的 tf.data.Dataset
ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes=())
for count_batch in ds_counter.repeat().batch(10).take(10):
    print(count_batch.numpy())


# output_shapes 参数虽然不是必需的，但强烈建议添加，因为许多 TensorFlow 运算不支持秩未知的张量

# 下面的示例生成器对两方面进行了演示，它会返回由数组组成的元组，其中第二个数组是长度未知的向量
def gen_series():
    i = 0
    while True:
        size = np.random.randint(0, 10)
        yield i, np.random.normal(size=(size,))
        i += 1


for i, series in gen_series():
    print(i, ': ', str(series))
    if i > 5:
        break

ds_series = tf.data.Dataset.from_generator(
    gen_series,
    output_types=(tf.int32, tf.float32),
    output_shapes=((), (None,))
)
print(ds_series)  # 第一个条目是标量，形状为 ()，第二个条目是长度未知的向量，形状为 (None,)

# 在批处理形状可变的数据集时，需要使用 Dataset.padded_batch
ds_series_batch = ds_series.shuffle(20).padded_batch(10)
ids, sequence_batch = next(iter(ds_series_batch))
print(ids.numpy())
print(sequence_batch.numpy())

# 将 preprocessing.image.ImageDataGenerator 封装为 tf.data.Dataset

# flowers = tf.keras.utils.get_file(
#     'flower_photos',
#     'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
#     untar=True)

# 数据路径（文件夹，非压缩文件）  需要把下载的文件tgz解压
images_dir = join_path.load_data.images_path
images_dir = pathlib.Path(images_dir)

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, rotation_range=20)
images, labels = next(img_gen.flow_from_directory(images_dir))

print(images.dtype, images.shape)
print(labels.dtype, labels.shape)

ds = tf.data.Dataset.from_generator(
    lambda: img_gen.flow_from_directory(images_dir),
    output_types=(tf.float32, tf.float32),
    output_shapes=([32, 256, 256, 3], [32, 5])
)
print(ds.element_spec)

for image, label in ds.take(1):
    print('image.shape: ', image.shape)
    print('label.shape: ', label.shape)

# endregion


# region 处理 TFRecord 数据
# 不适合存储在内存中的大型数据集
#   TFRecord 文件格式是一种简单的、面向记录的二进制格式

# fsns_test_file = tf.keras.utils.get_file(
#     "fsns.tfrec",
#     "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001"
# )

fsns_test_file = 'D:\\Projects\\Python\\Local\\LearnTF\\keras_data\\fsns\\fsns-00000-of-00001'
dataset = tf.data.TFRecordDataset(filenames=[fsns_test_file])

# 很多 TFRecord 文件中使用序列化的 tf.train.Example 记录
#   这些记录需要在检查前进行解码
raw_example = next(iter(dataset))
parsed = tf.train.Example.FromString(raw_example.numpy())
parsed.features.feature['image/text']

# endregion


# region 使用文本数据

# directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
# file_names = ['cowper.txt', 'derby.txt', 'butler.txt']
# file_paths = [
#     tf.keras.utils.get_file(file_name, directory_url + file_name)
#     for file_name in file_names
# ]

file_dir = 'D:\\Projects\\Python\\Local\\LearnTF\\keras_data\\illiad_txt'
file_names = ['cowper.txt', 'derby.txt', 'butler.txt']
file_paths = [os.path.join(file_dir, file_name) for file_name in file_names]
# tf.data.TextLineDataset 提供了一种从一个或多个文本文件中提取行的简便方式
#   如果给定一个或多个文件名，TextLineDataset 会为这些文件的每一行生成一个字符串元素

dataset = tf.data.TextLineDataset(file_paths)
for line in dataset.take(5):  # 读取文件前5行
    print(line.numpy())

# Dataset.interleave：交错读取不同文件中的行
files_ds = tf.data.Dataset.from_tensor_slices(file_paths)
lines_ds = files_ds.interleave(tf.data.TextLineDataset, cycle_length=3)

for i, line in enumerate(lines_ds.take(9)):  # 交错读取文件的前三行
    if i % 3 == 0:
        print()
    print(line.numpy())

# 使用 Dataset.skip() 或 Dataset.filter 转换移除一些不需要的行
titanic_file = 'D:\\Projects\\Python\\Local\\LearnTF\\keras_data\\titanic_csv\\train.csv'
titanic_lines = tf.data.TextLineDataset(titanic_file)
for line in titanic_lines.take(10):
    print(line.numpy())


def survived(line):
    return tf.not_equal(tf.strings.substr(line, 0, 1), '0')


# 去除csv第一行，和第一列包含为0的行
survivors = titanic_lines.skip(1).filter(survived)
for line in survivors.take(10):
    print(line.numpy())

# endregion


# region 使用 CSV 数据
df = pd.read_csv(titanic_file)
df.head()

# 如果数据适合存储在内存中，那么使用 Dataset.from_tensor_slices 方法
titanic_slices = tf.data.Dataset.from_tensor_slices(dict(df))
for feature_batch in titanic_slices.take(1):
    for key, value in feature_batch.items():
        print('  {!r:20s}: {}'.format(key, value))

# experimental.make_csv_dataset 函数是用来读取 CSV 文件集的高级接口
titanic_batchs = tf.data.experimental.make_csv_dataset(
    titanic_file, batch_size=4,
    label_name='survived'
)

for feature_batch, label_batch in titanic_batchs.take(1):
    print(f"'survived: {label_batch}'")
    print('features:')
    for key, value in feature_batch.items():
        print('  {!r:20s}: {}'.format(key, value))

# 只需要列的一个子集, 使用 select_columns
titanic_batchs = tf.data.experimental.make_csv_dataset(
    titanic_file, batch_size=4,
    label_name='survived', select_columns=['class', 'fare', 'survived']
)
for feature_batch, label_batch in titanic_batchs.take(1):
    print(f"'survived: {label_batch}'")
    print('features:')
    for key, value in feature_batch.items():
        print('  {!r:20s}: {}'.format(key, value))

# 级别更低的 experimental.CsvDataset 类，该类可以提供粒度更细的控制
# 但它不支持列类型推断，必须指定每个列的类型
titanic_types = [tf.int32, tf.string, tf.float32, tf.int32, tf.int32, tf.float32, tf.string, tf.string, tf.string, tf.string]
dataset = tf.data.experimental.CsvDataset(titanic_file, titanic_types, header=True)
for line in dataset.take(10):
    print([item.numpy() for item in line])

# endregion


# region 使用文件集
# 每个文件都是一个样本

# images_dir   flower_photos
for item in images_dir.glob('*'):
    print(item.name)

# 每个类目录中的文件是样本
list_ds = tf.data.Dataset.list_files(str(images_dir / '*/*'))
for f in list_ds.take(5):
    print(f.numpy())


# 使用 tf.io.read_file 函数读取数据，并从路径提取标签，返回 (image, label)
def process_path(file_path):
    label = tf.strings.split(file_path, os.sep)[-2]
    return tf.io.read_file(file_path), label


labeled_ds = list_ds.map(process_path)
for image_raw, label_text in labeled_ds.take(1):
    print(repr(image_raw.numpy()[:100]))
    print()
    print(label_text.numpy())

# endregion


# endregion


# region 批处理数据集元素

# region 简单批处理

# 最简单的批处理方式是将数据集的 n 个连续元素堆叠成单个元素
#   Dataset.batch() 转换就负责执行此操作
inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)
for batch in batched_dataset.take(4):
    print([arr.numpy() for arr in batch])

# Dataset.batch 的默认设置会导致未知的批次大小，因为最后一个批次可能不完整。请注意形状中的 None
print(batched_dataset)  # <BatchDataset element_spec=(TensorSpec(shape=(None,), dtype=tf.int64, name=None), TensorSpec(shape=(None,), dtype=tf.int64, name=None))>
# 使用 drop_remainder 参数忽略最后一个批次，以获得完整的形状传播
batched_dataset = dataset.batch(7, drop_remainder=True)
print(batched_dataset)  # <BatchDataset element_spec=(TensorSpec(shape=(7,), dtype=tf.int64, name=None), TensorSpec(shape=(7,), dtype=tf.int64, name=None))>

# endregion


# region 批处理带填充的张量
# 处理不同的大小的输入数据
#   通过 Dataset.padded_batch 转换指定一个或多个可能被填充的维度，从而批处理不同形状的张量
dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(4, padded_shapes=(None,))

for batch in dataset.take(2):
    print(batch.numpy())
    print()

# endregion

# endregion


# region 训练工作流

# region 处理多个周期

# 在多个周期内迭代数据集，最简单的方式是使用 Dataset.repeat() 转换

titanic_lines = tf.data.TextLineDataset(titanic_file)


def plot_batch_sizes(ds):
    batch_sizes = [batch.shape[0] for batch in ds]
    plt.bar(range(len(batch_sizes)), batch_sizes)
    plt.xlabel('Batch number')
    plt.ylabel('Batch size')


# 应用不带参数的 Dataset.repeat() 转换，将无限次地重复输入

# 在 Dataset.repeat 之后应用的 Dataset.batch 将生成跨越周期边界的批次

titanic_batchs = titanic_lines.repeat(3).batch(128)
plot_batch_sizes(titanic_batchs)
plt.show()

# 如果需要明确的周期分隔，请将 Dataset.batch 置于重复前
titanic_batchs = titanic_lines.batch(128).repeat(3)
plot_batch_sizes(titanic_batchs)

# 在每个周期结束时执行自定义计算，最简单的方式是在每个周期上重新启动数据集迭代
epochs = 3
dataset = titanic_lines.batch(128)
for epoch in range(epochs):
    for batch in dataset:
        print(batch.shape)
    print('End of epoch: ', epoch)

# endregion


# region 随机重排输入数据
# Dataset.shuffle() 转换会维持一个固定大小的缓冲区
#   并从该缓冲区均匀地随机选择下一个元素
#   虽然较大的 buffer_size 可以更彻底地重排，但可能会占用大量的内存和时间来填充
lines = tf.data.TextLineDataset(titanic_file)
counter = tf.data.experimental.Counter()

dataset = tf.data.Dataset.zip((counter, lines))
dataset = dataset.shuffle(buffer_size=100)
dataset = dataset.batch(20)

# 由于 buffer_size 是 100，而批次大小是 20，第一个批次不包含索引大于 120 的元素
n, line_batch = next(iter(dataset))
print(n.numpy())

# 在重排缓冲区为空之前，Dataset.shuffle 不会发出周期结束的信号
#   因此，置于重复之前的重排会先显示一个周期内的每个元素，然后移至下一个周期
dataset = tf.data.Dataset.zip((counter, lines))
shuffled = dataset.shuffle(buffer_size=100).batch(10).repeat(2)
# 下面是在epoch边界附近的元素ID
for n, line_batch in shuffled.skip(60).take(5):
    print(n.numpy())

shuffle_repeat = [n.numpy().mean() for n, line_batch in shuffled]
plt.plot(shuffle_repeat, label='shuffle().repeat()')
plt.ylabel('Mean item ID')
plt.legend()
plt.show()

# 在重排之前的重复会将周期边界混合在一起
dataset = tf.data.Dataset.zip((counter, lines))
shuffled = dataset.repeat(2).shuffle(buffer_size=100).batch(10)
for n, line_batch in shuffled.skip(55).take(15):
    print(n.numpy())

repeat_shuffle = [n.numpy().mean() for n, line_batch in shuffled]  # 边界混合
plt.plot(shuffle_repeat, label='shuffle().repeat()')
plt.plot(repeat_shuffle, label='repeat().shuffle()')
plt.ylabel('Mean item ID')
plt.legend()
plt.show()

# endregion


# endregion


# region 预处理数据

# Dataset.map(f) 转换会通过对输入数据集的每个元素应用一个给定函数 f 来生成一个新的数据集
#   函数 f 会获取在输入中表示单个元素的 tf.Tensor 对象，并返回在新数据集中表示单个元素的 tf.Tensor 对象

# region 解码图像数据并调整大小
list_ds = tf.data.Dataset.list_files(str(images_dir / '*/*'))


# 从文件中读取图像，将其解码为密集张量，并将其大小调整为固定形状
def parse_image(filename):
    parts = tf.strings.split(filename, os.sep)
    label = parts[-2]

    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [128, 128])
    return image, label


file_path = next(iter(list_ds))
image, label = parse_image(file_path)


def show(image, label):
    plt.figure()
    plt.imshow(image)
    plt.title(label.numpy().decode('utf-8'))
    plt.axis('off')


show(image, label)
plt.show()

images_ds = list_ds.map(parse_image)
for image, label in images_ds.take(2):
    show(image, label)
plt.show()


# endregion


# region 应用任意 Python 逻辑

def random_rotate_image(image):
    """ 随机旋转 """
    image = ndimage.rotate(image, np.random.uniform(-30, 30), reshape=False)
    return image


image, label = next(iter(images_ds))
image = random_rotate_image(image)
show(image, label)
plt.show()


# random_rotate_image将此函数用于 Dataset.map
#   在应用该函数时需要描述返回的形状和类型
def tf_random_rotate_image(image, label):
    im_shape = image.shape
    [image, ] = tf.py_function(random_rotate_image, [image], [tf.float32])
    image.set_shape(im_shape)
    return image, label


rot_ds = images_ds.map(tf_random_rotate_image)
for image, label in rot_ds.take(2):
    show(image, label)
plt.show()

# endregion


# region 解析 tf.Example 协议缓冲区消息
dataset = tf.data.TFRecordDataset(filenames=[fsns_test_file])

raw_example = next(iter(dataset))
parsed = tf.train.Example.FromString(raw_example.numpy())

feature = parsed.features.feature
raw_img = feature['image/encoded'].bytes_list.value[0]
img = tf.image.decode_png(raw_img)
plt.imshow(img)
plt.axis('off')
plt.title(feature['image/text'].bytes_list.value[0])
plt.show()

raw_example = next(iter(dataset))


def tf_parse(eg):
    example = tf.io.parse_example(
        eg[tf.newaxis], {
            'image/encoded': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'image/text': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        })
    return example['image/encoded'][0], example['image/text'][0]


img, txt = tf_parse(raw_example)
print(txt.numpy())
print(repr(img.numpy())[:20], '...')

decoded = dataset.map(tf_parse)
image_batch, text_batch = next(iter(decoded.batch(10)))
print(image_batch.shape)

# endregion


# region 时间序列窗口化
range_ds = tf.data.Dataset.range(100000)
batches = range_ds.batch(10, drop_remainder=True)
for batch in batches.take(5):
    print(batch.numpy())


def dense_1_step(batch):
    return batch[:-1], batch[1:]


predict_dense_1_step = batches.map(dense_1_step)
for features, label in predict_dense_1_step.take(3):
    print(features.numpy(), '=>', label.numpy())

# 要预测整个窗口而非一个固定偏移量，可以将批次分成两部分
batches = range_ds.batch(15, drop_remainder=True)


def label_next_5_steps(batch):
    return (batch[:-5], batch[-5:])


predict_5_step = batches.map(label_next_5_steps)
for features, label in predict_5_step.take(3):
    print(features.numpy(), '=>', label.numpy())

# 要允许一个批次的特征和另一批次的标签部分重叠，使用 Dataset.zip
feature_length = 10
label_length = 3

features = range_ds.batch(feature_length, drop_remainder=True)
labels = range_ds.batch(feature_length).skip(1).map(lambda labels: labels[:label_length])
predicted_steps = tf.data.Dataset.zip((features, labels))
for features, label in predicted_steps.take(5):
    print(features.numpy(), '=>', label.numpy())

# 使用 window  Dataset.window 方法可以为您提供完全控制（返回的是由 Datasets 组成的 Dataset）
window_size = 5
windows = range_ds.window(window_size, shift=1)
for sub_ds in windows.take(5):
    print(sub_ds)

# Dataset.flat_map 方法可以获取由数据集组成的数据集，并将其合并为一个数据集
for x in windows.flat_map(lambda x: x).take(30):
    print(x.numpy(), end=' ')


def sub_to_batch(sub):
    return sub.batch(window_size, drop_remainder=True)


for example in windows.flat_map(sub_to_batch).take(5):
    print(example.numpy())


# shift 参数控制着每个窗口的移动量


def make_window_dataset(ds, window_size=5, shift=1, stride=1):
    windows = ds.window(window_size, shift=shift, stride=stride)

    def sub_to_batch(sub):
        return sub.batch(window_size, drop_remainder=True)

    windows = windows.flat_map(sub_to_batch)
    return windows


ds = make_window_dataset(range_ds, window_size=10, shift=5, stride=3)
for example in ds.take(5):
    print(example.numpy())

# 然后，可以像之前一样轻松提取标签
dense_labels_ds = ds.map(dense_1_step)
for inputs, labels in dense_labels_ds.take(3):
    print(inputs.numpy(), '=>', labels.numpy())

# endregion


# region 重采样
# 在处理类非常不平衡的数据集时，需要对数据集重新采样

# zip_path = tf.keras.utils.get_file(
#     origin='https://storage.googleapis.com/download.tensorflow.org/data/creditcard.zip',  # 信用卡欺诈数据集
#     fname='creditcard.zip',
#     extract=True)
# csv_path = zip_path.replace('.zip', '.csv')

csv_path = 'D:\\Projects\\Python\\Local\\LearnTF\\keras_data\\creditcard\\creditcard.csv'
creditcard_ds = tf.data.experimental.make_csv_dataset(
    csv_path, batch_size=1024, label_name='Class',
    column_defaults=[float()] * 30 + [int()]  # 设置列的类型:30个浮点数和一个整型
)


def count(counts, batch):
    features, labels = batch
    class_1 = labels == 1
    class_1 = tf.cast(class_1, tf.int32)

    class_0 = labels == 0
    class_0 = tf.cast(class_0, tf.int32)

    counts['class_0'] += tf.reduce_sum(class_0)
    counts['class_1'] += tf.reduce_sum(class_1)

    return counts


counts = creditcard_ds.take(10).reduce(
    initial_state={'class_0': 0, 'class_1': 0},
    reduce_func=count
)
counts = np.array([counts['class_0'].numpy(),
                   counts['class_1'].numpy()]).astype(np.float32)

fractions = counts / counts.sum()
print(fractions)

# 数据集采样
#   用过滤器从信用卡欺诈数据中生成一个重采样数据集
negative_ds = (
    creditcard_ds
        .unbatch()
        .filter(lambda features, label: label == 0)
        .repeat())
positive_ds = (
    creditcard_ds
        .unbatch()
        .filter(lambda features, label: label == 1)
        .repeat())

for features, label in positive_ds.batch(10).take(1):
    print(label.numpy())

# 使用 tf.data.Dataset.sample_from_datasets 传递数据集以及每个数据集的权重
balanced_ds = tf.data.Dataset.sample_from_datasets(
    [negative_ds, positive_ds], [0.5, 0.5]  # 数据集为每个类生成样本的概率是 50/50
).batch(10)
for features, labels in balanced_ds.take(10):
    print(labels.numpy())


# 拒绝重采样
#  Dataset.sample_from_datasets 方式的一个问题是每个类需要一个单独的 Dataset
#    可以使用 Dataset.filter 创建这两个数据集，但这会导致所有数据被加载两次

# 将 Dataset.rejection_resample 方法应用于数据集以使其重新平衡，而且只需加载一次数据，元素将被删除或重复，以实现平衡

# rejection_resample 需要一个 class_func 参数
#   这个 class_func 参数会被应用至每个数据集元素，并且会被用来确定某个样本属于哪一类，以实现平衡的目的

def class_func(features, label):
    return label


resample_ds = (
    creditcard_ds
        .unbatch()
        .rejection_resample(class_func, target_dist=[0.5, 0.5],
                            initial_dist=fractions)
        .batch(10))
# 使用 map 删除多余的标签副本
balanced_ds = resample_ds.map(lambda extra_label, features_and_label: features_and_label)
# 数据集为每个类生成样本的概率是 50/50
for features, labels in balanced_ds.take(10):
    print(labels.numpy())
# endregion


# endregion


# region 迭代器检查点操作
# 迭代器检查点可能会很大，Dataset.shuffle 和 Dataset.prefetch 之类的转换需要在迭代器内缓冲元素

range_ds = tf.data.Dataset.range(20)

iterator = iter(range_ds)
ckpt = tf.train.Checkpoint(step=tf.Variable(0), iterator=iterator)
manager = tf.train.CheckpointManager(ckpt, '/data/tmp/my_ckpt', max_to_keep=3)

print([next(iterator).numpy() for _ in range(5)])

save_path = manager.save()

print([next(iterator).numpy() for _ in range(5)])

ckpt.restore(manager.latest_checkpoint)

print([next(iterator).numpy() for _ in range(5)])

# endregion


# region 结合使用 tf.data 与 tf.keras
(images, labels), (img_test, label_test) = tools.load_fashion_data()
# train, test = tf.keras.datasets.fashion_mnist.load_data()

images = images / 255.0
labels = labels.astype(np.int32)
fmnist_train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
fmnist_train_ds = fmnist_train_ds.shuffle(5000).batch(32)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(fmnist_train_ds, epochs=2)

# 对于评估，可以传递评估步数
loss, accuracy = model.evaluate(fmnist_train_ds)
print("Loss :", loss)
print("Accuracy :", accuracy)

# 调用 Model.predict 时不需要标签
predict_ds = tf.data.Dataset.from_tensor_slices(images).batch(32)
result = model.predict(predict_ds, steps=10)
print(result.shape)

# 如果要传递一个无限大的数据集（比如通过调用 Dataset.repeat）
#   只需要同时传递 steps_per_epoch 参数
model.fit(fmnist_train_ds.repeat(), epochs=2, steps_per_epoch=20)

# 对于长数据集，可以设置要评估的步数
loss, accuracy = model.evaluate(fmnist_train_ds.repeat(), steps=10)
print("Loss :", loss)
print("Accuracy :", accuracy)

# 如果传递了包含标签的数据集，则标签会被忽略
result = model.predict(fmnist_train_ds, steps=10)
print(result.shape)

# endregion
