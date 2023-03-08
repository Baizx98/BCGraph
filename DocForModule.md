- [代码模块说明](#代码模块说明)
  - [ForkingPicker](#forkingpicker)
  - [dataclass](#dataclass)
  - [typing](#typing)
  - [NamedTuple](#namedtuple)
  - [py::call\_guard装饰器](#pycall_guard装饰器)
  - [@classmethod注解](#classmethod注解)
- [开发方法](#开发方法)

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

# 开发方法
模块调试