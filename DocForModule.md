- [代码模块说明](#代码模块说明)
  - [ForkingPicker](#forkingpicker)
  - [dataclass](#dataclass)
  - [typing](#typing)
  - [NamedTuple](#namedtuple)
  - [py::call\_guard装饰器](#pycall_guard装饰器)
  - [@classmethod注解](#classmethod注解)
  - [CSR格式](#csr格式)
- [开发方法](#开发方法)
  - [在Python中调用C++模块的方式](#在python中调用c模块的方式)
  - [缓存命中率测试流程](#缓存命中率测试流程)

# 代码模块说明

## ForkingPicker
ForkingPickler 是 Python 中的一个序列化库，用于在 multiprocessing 模块中将 Python 对象序列化（即将其转换为字节流），以便可以在不同的进程之间传输。它是对 Python 的标准 pickle 库的扩展，它支持在进程间共享文件描述符，它的主要特点是在使用 fork() 系统调用时可以更快地序列化和反序列化对象。  
在 multiprocessing 模块中，可以通过传递对象的序列化版本来实现进程间通信，这个过程需要使用 ForkingPickler 对象将对象序列化为一个字节流。在子进程中，可以使用 pickle.loads() 方法将字节流反序列化回 Python 对象。  


```python

#在上面的示例中，我们首先定义了一个名为 MyData 的自定义类，它包含一个 data 属性。然后，我们定义了两个函数 reduce_mydata 和 rebuild_mydata，分别用于将 MyData 对象序列化为一个元组并重建该对象。接下来，我们调用 ForkingPickler.register 方法，将 MyData 类和序列化和重建的函数进行注册。最后，在主程序中，我们创建了一个 MyData 对象，并将其传递给 worker_func 函数，以验证该对象能够在多进程中正确传递。
import multiprocessing
from multiprocessing.reduction import ForkingPickler

class MyData:
    def __init__(self, data):
        self.data = data

def reduce_mydata(data):
    return (rebuild_mydata, (data.data, ))

def rebuild_mydata(data):
    return MyData(data)

ForkingPickler.register(MyData, reduce_mydata)

def worker_func(data):
    print(data.data)

if __name__ == '__main__':
    pool = multiprocessing.Pool()
    my_data = MyData('Hello, world!')
    pool.apply(worker_func, args=(my_data, ))
```

## dataclass
dataclasses 是Python 3.7引入的一个模块，它可以用来自动化生成类，并自动添加一些常用的方法和属性，减少编写代码的工作量。这个模块在处理数据类型这样的简单类时非常方便。

使用 dataclasses 可以通过简单的声明，来创建具有一些预定义特性的类，如自动属性赋值、__repr__()、__eq__()等。

使用 dataclasses 可以将类的声明从下面这样： 
```python
class MyClass:
    def __init__(self, attr1, attr2):
        self.attr1 = attr1
        self.attr2 = attr2

    def my_method(self):
        pass
```
简化为： 
```python
from dataclasses import dataclass

@dataclass
class MyClass:
    attr1: str
    attr2: int

    def my_method(self):
        pass
```
这里，使用 @dataclass 装饰器来生成一个类，只需要定义每个属性的类型和名称。它还为我们自动添加了 __init__()、__repr__()、__eq__()等方法。

通过使用 dataclasses，我们可以更容易地定义简单的数据类，并自动获得一些常用的方法。同时，由于它们使用了 Python 3.7 引入的一些新特性，使得代码更加简洁易读。 

## typing
在 Python 中，typing 模块是在 Python 3.5 中引入的，它提供了类型提示和类型注释的功能，使得 Python 支持静态类型检查，可以提高代码的可读性、可维护性和可重构性。

typing 模块定义了许多类、函数和装饰器，用于实现类型注释和类型检查，包括但不限于以下内容：

- 类型变量（TypeVar）：用于泛型类型的声明，例如 List[T]。
- 泛型类型（Generic）：用于定义泛型类。
- 类型别名（TypeAlias）：用于定义类型别名。
- 泛型函数（Callable）：用于定义泛型函数。
- 类型检查器（TypeChecker）：用于检查类型注释是否正确。
- 类型推断器（TypeVar）：用于推断变量的类型。
- 类型转换器（cast）：用于将一个对象转换为指定的类型。 
 
除此之外，typing 模块还定义了许多基本的数据类型和容器类型，例如 Tuple、List、Set、Dict 等，以及一些高级数据类型和容器类型，例如 Union、Optional、Any、Callable 等，这些类型的定义可以帮助开发者更加方便地实现类型注释和类型检查。 

## NamedTuple
NamedTuple是Python标准库typing中的一种数据类型，用于创建具有命名属性的元组。它类似于普通元组，但可以为每个元素分配名称，这使得代码更加清晰易读。与普通元组不同，NamedTuple实例可以像普通对象一样访问其属性，而不必使用索引。它还可以继承和进行类型注释。

创建NamedTuple的方式与创建类似，需要定义类型名称和字段名称及其类型。例如： 
```python
from typing import NamedTuple

class Point(NamedTuple):
    x: float
    y: float
```
上述代码创建了一个名为Point的NamedTuple类型，该类型包含两个命名属性x和y，类型都是float。可以像普通元组一样使用Point类型创建实例，并像普通对象一样访问它们的属性：
```python
p = Point(1.0, 2.0)
print(p.x)  # output: 1.0
print(p.y)  # output: 2.0
```
## py::call_guard装饰器 
在Python中，有一个全局解释器锁（GIL），它的作用是确保同一时刻只有一个线程可以执行Python代码。这是为了防止多线程竞争和数据不一致的问题。但是，如果使用Python调用C++代码时，如果C++代码中有长时间运行的任务，例如I/O操作或GPU计算，会阻塞Python线程，因此会降低应用程序的并发性能。

为了解决这个问题，pybind11库提供了"py::call_guard"装饰器，它允许在调用C++代码时释放GIL，以便其他Python线程可以继续执行。这样可以提高应用程序的并发性能。

"py::call_guard"装饰器可以与pybind11中的函数一起使用，例如在C++中定义的Python类或函数。当使用该装饰器时，需要将它作为参数传递给C++函数，然后在函数内部使用它来装饰Python函数，以确保在调用C++函数时，GIL能够被释放。 

## @classmethod注解
classmethod 是一个 Python 内置的装饰器（decorator），它用于修饰类方法。classmethod 修饰的方法在调用时不需要创建类实例，它的第一个参数是类本身，通常用 cls 表示。

在 Python 中，调用类方法有两种方式：

- 通过实例对象调用
- 通过类对象调用 
 
classmethod 能够保证上述两种调用方式返回的结果相同，因为第一个参数都是类对象。因此，它通常用于实现与类相关的工厂方法，或者创建可替代类构造函数的方法。


## CSR格式
CSR (Compressed Sparse Row) 是一种用于稀疏矩阵存储的格式，它可以有效地压缩矩阵中的空白元素，从而节省存储空间。在 CSR 格式中，矩阵被表示为三个数组：行指针数组、列索引数组和值数组。  

具体来说，行指针数组 row_ptr 记录了每一行在列索引数组和值数组中的起始位置和结束位置，例如 row_ptr[i] 表示第 i 行在列索引数组和值数组中的起始位置，row_ptr[i+1] 表示第 i 行的结束位置。这个数组的长度为矩阵的行数加一。  

列索引数组 col_idx 记录了每个非空元素所在的列的索引，例如 col_idx[j] 表示值数组中的第 j 个元素所在的列的索引。这个数组的长度为矩阵中非空元素的个数。  

值数组 values 记录了每个非空元素的值，例如 values[j] 表示第 j 个非空元素的值。这个数组的长度也为矩阵中非空元素的个数。  

举个例子，假设有如下的稀疏矩阵：
```
1 0 0
0 2 0
0 0 3
0 4 5
```
那么它的 CSR 表示就是：
```python
row_ptr = [0, 1, 2, 3, 5]
col_idx = [0, 1, 2, 1, 2]
values = [1, 2, 3, 4, 5]
```
其中，`row_ptr`表示每一行的起始位置和结束位置，第一行从位置 0 开始，第二行从位置 1 开始，第三行从位置 2 开始，第四行从位置 3 开始，最后一行从位置 5 结束。`col_idx`表示每个非空元素所在的列的索引，第一个非空元素在第 0 列，第二个在第 1 列，第三个在第 2 列，第四个在第 1 列，最后一个在第 2 列。values 表示每个非空元素的值，依次为 1、2、3、4、5。
# 开发方法
模块调试

## 在Python中调用C++模块的方式
1. 导入pybind11头文件：
   ```c++
   #include <pybind11/pybind11.h>
   ```
2. 在需要导出的C++模块的文件中，使用PYBIND11_MODULE宏定义一个模块：
   ```c++
   PYBIND11_MODULE(module_name, module_variable) {
    // Code to expose C++ functions, classes, etc. to Python
    }
    ```
    module_name是模块的名称，将用于Python导入模块时的引用。  
    module_variable是一个C++变量的名称，该变量将在Python中表示整个模块。
3. 在PYBIND11_MODULE块内，使用pybind11库中的函数来将C++函数、类等导出到Python中：  
    - 将C++函数导出到Python：
        ```c++
        m.def("function_name", &function_name, "docstring");
        ```
      - `m.def`是pybind11库的函数，用于将C++函数导出到Python中。  
      - `"function_name"`是在Python中使用的函数名称。
      - '&function_name'是C++函数的地址。  
      - '"docstring"'是在Python中使用的文档字符串。  
    - 将C++类导出到Python：
        ```c++
        py::class_<Class>(m, "Class")
            .def(py::init<>())
            .def("member_function", &Class::member_function, "docstring")
            .def_readwrite("member_variable", &Class::member_variable, "docstring");
        ```
        - py::class_是pybind11库的类模板，用于将C++类导出到Python中。
        - Class是C++类的名称。
        - m是Python模块的变量名。
        - .def(py::init<>())是构造函数的定义。
        - .def("member_function", &Class::member_function, "docstring")是将成员函数导出到Python中。
        - .def_readwrite("member_variable", &Class::member_variable, "docstring")是将成员变量导出到Python中。
4. 编译C++代码，并将其链接到Python解释器中。
    ```r
    g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` example.cpp -o example`python3-config --extension-suffix`
    ```
1. 在Python中使用导出的模块：

    ```python
    import module_name

    # Call a C++ function
    module_name.function_name()

    # Create a C++ class instance
    obj = module_name.Class()

    # Call a member
    ```

## py::call_guard<py::gil_scoped_release>()
`py::call_guard<py::gil_scoped_release>()`是Pybind11库中用于在C++函数调用期间释放全局解释器锁（GIL）的函数调用包装器。

GIL是CPython（Python的参考实现）中的一种机制，它确保只有一个线程在任何时刻执行Python字节码。这可能会导致多线程应用程序的性能瓶颈，特别是当大量时间花费在I/O绑定任务上时。

通过使用`py::call_guard<py::gil_scoped_release>()`，Pybind11在调用C++函数之前释放GIL，使得其他Python线程可以并发执行。一旦函数调用完成，Pybind11会自动重新获取GIL。这可以显著提高使用Pybind11通过C++扩展的多线程Python应用程序的性能。

值得注意的是，在某些情况下释放GIL可能不安全，因为某些Python对象和函数需要保持GIL。因此，应谨慎使用此功能，并在生产环境中彻底测试其使用情况。

## edge_index 
`edge_index`是一个二维张量，用于表示图中所有边的连接关系。它的大小为2 x E，其中E是边的数量，每列代表一条边。例如，`edge_index[0][i]`和`edge_index[1][i]`表示第i条边连接的两个节点的索引。在PyG中，`edge_index`通常作为图数据的一个属性，可以通过data.edge_index访问。 
## node_idx
`node_idx`是一个一维张量，用于表示需要进行计算的节点的索引。在GNN训练中，通常只对一部分节点进行计算，而不是对所有节点进行计算。这个一部分节点的索引就由`node_idx`来表示。在PyG中，`node_idx`通常作为一个参数传递给GNN模型的forward方法。

## data.x
在图神经网络（GNN）的训练中，`data.x`通常用于表示节点的特征矩阵。它是一个二维张量，大小为N x F，其中N是节点的数量，F是每个节点的特征向量的维度。在PyG中，data.x通常作为图数据的一个属性，可以通过`data.x`访问。

节点的特征向量通常包含关于节点的各种信息，例如节点的内容、节点的位置、节点的度等等。在GNN模型中，节点的特征向量通常作为输入进行处理，通过节点之间的连接关系和节点的特征向量，模型可以学习到节点之间的关系以及对节点进行分类、预测等任务。

在GNN训练中，通常需要对节点的特征矩阵进行一些预处理，例如对节点的特征进行归一化、将节点的特征投影到低维空间等等，以便模型能够更好地学习节点之间的关系。预处理的具体方法取决于应用场景和数据集的特点。

## adj
在这段代码中，adjs代表了一批子图，其中每个子图由多个邻接矩阵组成，每个邻接矩阵都是一个元组(edge_index, e_id, size)，其中edge_index表示邻接矩阵的边索引，e_id表示邻接矩阵的边索引相对于全局边索引的偏移量，size表示邻接矩阵的大小。

具体来说，adjs的长度等于len(sizes)，也就是我们在定义NeighborSampler时指定的邻居采样大小数组的长度。在这个例子中，sizes=[25, 10]，所以adjs的长度为2。

每个元组(edge_index, e_id, size)代表一个邻接矩阵，edge_index是一个形状为(2, num_edges)的张量，其中num_edges表示边的数量，edge_index[0]和edge_index[1]分别是源节点和目标节点的索引。e_id是一个整数，表示邻接矩阵的边索引相对于全局边索引的偏移量。size是一个整数，表示邻接矩阵的大小。

因此，adjs的长度是2，每个元素都是一个包含多个邻接矩阵的元组，每个邻接矩阵由一个边索引、一个偏移量和一个大小组成。

## 在调用multiprocessing.spawn函数时，参数args中为什么没有rank
multiprocessing.spawn 函数是 Python 中用于多进程编程的模块之一，它允许你启动新的 Python 解释器进程，并在每个进程中执行指定的函数。

当使用 multiprocessing.spawn 启动多个进程时，每个进程都可以访问它自己的进程编号，这个编号通常称为进程的 "rank"，但是在 args 参数中并没有直接指定进程的 rank。实际上，rank 是由 multiprocessing.spawn 在内部处理的。

当你调用 multiprocessing.spawn 函数时，你需要指定一个函数来在新的进程中执行。这个函数的第一个参数通常被称为 rank，它是一个整数，表示当前进程的编号。在 multiprocessing.spawn 中，这个参数是隐式传递的。也就是说，spawn 函数会自动为每个进程分配一个 rank，并将它作为第一个参数传递给指定的函数。

例如，假设你想要启动 4 个进程，并在每个进程中执行一个名为 my_function 的函数。你可以这样调用 spawn 函数：
```python
import multiprocessing

def my_function(rank):
    print("Hello from process", rank)

if __name__ == '__main__':
    multiprocessing.spawn(my_function, args=(,))
```
在这个例子中，spawn 函数会启动 4 个进程，并在每个进程中执行 my_function 函数。args 参数为空，因为我们不需要在启动进程时指定任何其他参数。每个进程都会自动获取一个 rank 参数，它将作为 my_function 函数的第一个参数传递。my_function 函数将在每个进程中打印出 "Hello from process" 后面跟随该进程的 rank 值。

## 缓存命中率测试流程
1. 定义GPU数量
2. 划分训练集并定义DataLoader列表
3. 定义采样器
4. 设定若干epoch，每个epoch中完成采样，统计节点频率，不需要模型计算

- 首先进行多个epoch的采样，获取不同节点在不同GPU的热度和总热度
- 得到节点的度排序
- 按照某种策略将特征缓存到两个GPU上，储存在gpu_chached_ids_list中
- 再进行若干轮epoch模拟训练，统计在gpu_cached_ids_list的命中数量，计算命中率并保存到文件中