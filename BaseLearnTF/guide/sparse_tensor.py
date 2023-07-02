# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: sparse_tensor
# CreateTime: 2023/6/17 15:51
# Summary: 稀疏张量


import keras
import tensorflow as tf

# 稀疏张量
#  当使用包含大量零值的张量时，以节省空间和时间的方式存储它们

# region 创建 SparseTensor

st1 = tf.sparse.SparseTensor(indices=[[0, 3], [2, 4]],  # 非零值的索引位置
                             values=[10, 20],  # 非零值的一维张量
                             dense_shape=[3, 10])  # 张量的形状

print(st1)


def pprint_spares_tensor(st):
    """ 漂亮的打印稀疏张量 """
    s = '<SparesTensor shape=%s \n values = {' % (st.dense_shape.numpy().tolist(),)
    for (index, value) in zip(st.indices, st.values):
        s += f"\n  %s: %s" % (index.numpy().tolist(), value.numpy().tolist())
    return s + '}>'


print(pprint_spares_tensor(st1))

st2 = tf.sparse.from_dense([[1, 0, 0, 8], [0, 0, 0, 0], [0, 0, 3, 0]])
print(pprint_spares_tensor(st2))

# endregion


# region 操纵稀疏张量

# tf.math.add可用于 密集张量的算数操作 不能 用于 稀疏张量
st_a = tf.sparse.SparseTensor(indices=[[0, 2], [3, 4]],
                              values=[31, 2],
                              dense_shape=[4, 10])
st_b = tf.sparse.SparseTensor(indices=[[0, 2], [7, 0]],
                              values=[56, 38],
                              dense_shape=[4, 10])
st_sum = tf.sparse.add(st_a, st_b)
print(pprint_spares_tensor(st_sum))
# st_a + st_b  # Exception

# 稀疏张量 与 密集矩阵 相乘
st_c = tf.sparse.SparseTensor(indices=[[0, 1], [1, 0], [1, 1]],
                              values=[13, 15, 17],
                              dense_shape=[2, 2])
#  0, 13
#  15,17
mb = tf.constant([[4], [6]])
product = tf.sparse.sparse_dense_matmul(st_c, mb)
print(product)

# 使用 spares_dense_matmul 将稀疏张量 密集矩阵 分开

sparse_pattern_A = tf.sparse.SparseTensor(indices=[[2, 4], [3, 3], [3, 4], [4, 3], [4, 4], [5, 4]],
                                          values=[1, 1, 1, 1, 1, 1],
                                          dense_shape=[8, 5])
sparse_pattern_B = tf.sparse.SparseTensor(indices=[[0, 2], [1, 1], [1, 3], [2, 0], [2, 4], [2, 5], [3, 5],
                                                   [4, 5], [5, 0], [5, 4], [5, 5], [6, 1], [6, 3], [7, 2]],
                                          values=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                          dense_shape=[8, 6])
sparse_pattern_C = tf.sparse.SparseTensor(indices=[[3, 0], [4, 0]],
                                          values=[1, 1],
                                          dense_shape=[8, 6])
sparse_pattern_list = [sparse_pattern_A, sparse_pattern_B, sparse_pattern_C]
spares_pattern = tf.sparse.concat(axis=1, sp_inputs=sparse_pattern_list)
print(tf.sparse.to_dense(spares_pattern))

spares_slice_A = tf.sparse.slice(sparse_pattern_A, start=[0, 0], size=[8, 5])
spares_slice_B = tf.sparse.slice(sparse_pattern_B, start=[0, 5], size=[8, 6])
spares_slice_C = tf.sparse.slice(sparse_pattern_C, start=[0, 10], size=[8, 6])
print(tf.sparse.to_dense(spares_slice_A))
print(tf.sparse.to_dense(spares_slice_B))
print(tf.sparse.to_dense(spares_slice_C))

# tf2.4 以上版本使用 tf.sparse.map_values
st2_plus_5 = tf.sparse.map_values(tf.add, st2, 5)
print(tf.sparse.to_dense(st2_plus_5))

# 其他tf版本
st2_plus_5 = tf.sparse.SparseTensor(
    st2.indices,
    st2.values + 5,
    st2.dense_shape)
print(tf.sparse.to_dense(st2_plus_5))
# endregion


# region tf.keras

# sparse=True 可以在 Keras层之间传递稀疏张量，也可以让 keras 返回稀疏张量

x = tf.keras.Input(shape=(4,), sparse=True)
y = tf.keras.layers.Dense(4)(x)
model = tf.keras.Model(x, y)
# 仅使用支持稀疏输入的层，则如何将稀疏张量作为输入传递到Keras模型
sparse_data = tf.sparse.SparseTensor(
    indices=[(0, 0), (0, 1), (0, 2),
             (4, 3), (5, 0), (5, 1)],
    values=[1, 1, 1, 1, 1, 1],
    dense_shape=(6, 4)
)
model(sparse_data)

model.predict(sparse_data)
# endregion


# region tf.data


dataset = tf.data.Dataset.from_tensor_slices(sparse_data)
for element in dataset:
    print(pprint_spares_tensor(element))
    print('**********')

# 稀疏张量数据集的 批处理 和 非批处理
batched_dataset = dataset.batch(2)
for element in batched_dataset:
    print(pprint_spares_tensor(element))
    print('**********')

unbatched_dataset = dataset.unbatch()
for element in unbatched_dataset:
    print(pprint_spares_tensor(element))
    print('**********')

# 使用 Dataset.map 稀疏张量的创建和转换
transform_dataset = dataset.map(lambda x: x * 2)
for i in transform_dataset:
    print(pprint_spares_tensor(i))
    print('**********')


# endregion


# region tf.function

@tf.function
def f(x, y):
    return tf.sparse.sparse_dense_matmul(x, y)  # 矩阵的乘积


a = tf.sparse.SparseTensor(indices=[[0, 3], [2, 4]],
                           values=[15, 25],
                           dense_shape=[3, 10])

b = tf.sparse.to_dense(tf.sparse.transpose(a))
c = f(a, b)
print(c)

# endregion
