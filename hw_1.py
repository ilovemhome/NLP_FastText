import numpy as np

# 2. 建立一维数组a
a = np.array([4, 5, 6])
print("2.1 a的类型:", type(a))
print("2.2 a的形状:", a.shape)
print("2.3 a的第一个元素:", a[0])

# 3. 建立二维数组b
b = np.array([[4, 5, 6], [1, 2, 3]])
print("3.1 b的形状:", b.shape)
print("3.2 b(0,0):", b[0, 0], "b(0,1):", b[0, 1], "b(1,1):", b[1, 1])

# 4. 各种特殊矩阵
a_zeros = np.zeros((3, 3), dtype=int)
print("4.1 全0矩阵a:\n", a_zeros)
b_ones = np.ones((4, 5))
print("4.2 全1矩阵b:\n", b_ones)
c_eye = np.eye(4)
print("4.3 单位矩阵c:\n", c_eye)
d_rand = np.random.rand(3, 2)
print("4.4 随机数矩阵d:\n", d_rand)

# 5. 建立数组a
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("5.1 a:\n", a)
print("5.2 a(2,3):", a[2, 3], "a(0,0):", a[0, 0])

# 6. 切片
b = a[0:2, 2:4]
print("6.1 b:\n", b)
print("6.2 b(0,0):", b[0, 0])

# 7. 最后两行
c = a[1:3, :]
print("7.1 c:\n", c)
print("7.2 c第一行最后一个元素:", c[0, -1])

# 8. 花式索引
a = np.array([[1, 2], [3, 4], [5, 6]])
print("8. a[[0,1,2],[0,1,0]]:", a[[0, 1, 2], [0, 1, 0]])

# 9. 花式索引取多行
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
b = np.array([0, 2, 0, 1])
print("9. a[np.arange(4), b]:", a[np.arange(4), b])

# 10. 对第9题的四个元素加10
a[np.arange(4), b] += 10
print("10. 新的a:\n", a)