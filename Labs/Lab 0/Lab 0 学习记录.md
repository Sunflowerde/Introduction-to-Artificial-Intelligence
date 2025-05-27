# 1. Numpy的用法

## 1. 创建数组

```python
np.array(num, dtype)
```

`array`中传入的第一个参数为数据，第二个参数为数据类型

如果想要修改数据形状，可以使用`reshape`方法，

```python
np.array([1, 2, 3, 4]).reshape((2, 2))
```

即可生成一个$2\times 2$的矩阵。

## 2. 创建一个全为0的数组

```python
>>> np.zeros((3, 4))
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]]) 	
```

`zeros`内部传入一个元组`shape`，表示其各维度的大小。比如`zeros((3, 4))`就是一个$3$行$4$列的零矩阵。



注：如果不知道`shape`的具体维数，直接传入该`shape`即可。



## 3. 设置和修改数据类型

```python
t = np.array(range(1, 4), dtype = "float32")
```

在初始化数组时，需要多传入一个参数`dtype = ""`，表示该数组中元素的数据类型

```python
t1 = t.astype("int8")
```

可以使用`astype("")`来修改原数组的数据类型



## 4. 广播机制

广播机制是Numpy对不同形状数组进行计算的方式。

```python
>>> a = np.array([[0, 0, 0],
             [10, 10, 10],
             [20, 20, 20],
             [30, 30, 30]])
>>> b = np.array([0, 1, 2])
>>> a + b
[[0 1 2]
[10 11 12]
[20 21 22]
[30 31 32]]
```

当两数组形状不同时，就会进行复制，使二者达到相同的形状。

复制后：

```python
b = np.array([[0, 1, 2],
             [0, 1, 2],
             [0, 1, 2],
             [0, 1, 2]])
```

然后再直接进行相加。



## 5. np.where

语法：

```python
np.where(condition, x, y)
```

如果`condition`为`True`，则返回`x`，否则返回`y`

`np.where`会返回一个数组，其中数组与`condition`的`size`相同。



## 6. 随机数

```python
np.random.uniform(low = , high = , size = None)
```

`low`表示随机数的下限，`high`表示随机数的上限，`size`表示数组形状。

# 2. 类(Class)

## 1. 定义类

```python
class Dog:
    pass
```



## 2. 类的属性与方法

### 2.1 初始化方法

```python
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age
```

- `self`参数表示实例本身，必须放在第一个参数

### 2.2 定义方法

```python
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age
       
   	def bark(self):
   		print(f"{self.name} barking!")
```

- 在类中定义`bark`方法，必须有`self`参数

### 2.3 创建对象

```python
dog1 = Dog("a", 10)
dog2 = Dog("b", 20)

>>> dog1.name
a
```

- 创建的对象中必须包含两个参数

