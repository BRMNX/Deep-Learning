#DeepLearning #pytorch #python #深度学习 

引用torch库：`import torch`
# 1. tensor
用`torch.tensor()`函数创建一个张量，也就是高维数组.它们统称为张量.它返回的对象是`tensor`，而非python中的数组/列表.
## 1.1 创建张量
**标量**
- `torch.tensor(k)`：创建值为$k$的标量.

**向量**
- `torch.tensor([a,b,c,d])`：创建向量$[a,b,c,d]$.
- `torch.arange(k)`：输出向量$[0,1,\cdots,k-1]$.

**矩阵**
- `reshape(m,n)`：对向量的形状进行重排，得到$m\times n$矩阵.

> [!notion] Input
```python
A = torch.arange(20).reshape(5,4)
```
> [!success] Output

```python
A = [[ 0, 1, 2, 3],
 [ 4, 5, 6, 7],
 [ 8, 9,10,11],
 [12,13,14,15],
 [16,17,18,19]]
```
- `A.T`：矩阵$A$的转置.

**张量**
- `torch.zeros((m,n,p,q))`：创建一个$m\times n\times p\times q$维的零张量.
- `torch.ones((m,n,p,q))`：创建一个$m\times n\times p\times q$维的一张量.
- `torch.randn((m,n,p,q))`：创建每个元素都服从$N(0,1)$分布的张量.
- 也可通过`reshape()`创建张量.

> [!notion] Input

```python
X = torch.arange(24).reshape(2,3,4)
```
> [!success] Output

```python
X = [
[[0,1,2,3],[4,5,6,7],[8,9,10,11]],
[[12,13,14,15],[16,17,18,19],[20,21,22,23]]
]
```

## 1.2 张量的运算
- `x.shape`：张量各个轴的长度，返回一个`torch.Size`对象.比如当`x`是一个$m\times n$矩阵时，`x.shape`返回`torch.Size([m,n])`，其中`x.shape[0]`返回`m`，`x.shape[1]`返回`n`.
- `x.numel()`：返回张量的总元素个数.
- `x+y,x-y,x*y,x/y,x**y`：都是按元素计算.当$y$是标量时，$x$的每个元素都与$y$作运算.
- `x.sum()`：对张量的所有元素求和，输出一个tensor标量.
- `x.sum(axis = k)`：对张量的第$k$个轴求和，输出一个tensor，其第$k$轴消失.其中$k$可以是多个轴，比如`x.sum(axis = [0,1])`.可选参数`keepdims = True`，即求和后第$k$轴不消失，但是长度变为1，比如`x.sum(axis = 1, keepdims = True)`.
- `x.mean(axis = k)`：对张量的第$k$个轴求平均值，用法同`sum()`.
- `x.reshape()`：将张量的大小重排.一般重排为矩阵，可以用`x.reshape(-1,n)`来排为$n$列或用`x.reshape(m,-1)`来排为$m$行.**使用`reshape()`的前提是知道张量的总大小，否则会报错**(比如只有11个元素的向量无法重排成3行的矩阵).
- `x[-1]`：取张量的最后一个元素.
- `x[a:b]`：取张量的第$a$到$b-1$元素.如果`x`是矩阵，可以用`:`直接取到一整列或一整行，像matlab那样.
- `x[p,q,h]`：取张量某一个特定元素，或更改它的值.
- `torch.exp(x)`：将$x$的每个元素取$e$指数.
- `torch.cat((x,y),dim = k)`：将两个张量$x,y$按第$k$个轴连接.连接不会改变张量的轴数(比如矩阵连接后还是矩阵)，连接哪个轴，那个轴就会扩大.

> [!notion] Input

```python
x = torch.arange(12).reshape((3,4))
y = torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
torch.cat((x,y), dim = 0)
torch.cat((x,y), dim = 1)
```

> [!success] Output

```python
tensor([[0,1, 2, 3],
		[4,5, 6, 7],
		[8,9,10,11],
		[2,1, 4, 3],
		[1,2, 3, 4],
		[4,3, 2, 1]])
tensor([[0,1,2,3,2,1,4,3],
	    [4,5,6,7,1,2,3,4],
        [8,9,10,11,4,3,2,1]])
```
- `x == y`：当两个张量大小相等时，按元素判断它们是否相等.

> [!notion] Input

```python
x = torch.arange(12).reshape((3,4))
y = torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
print(x == y)
```
> [!success] Output

```python
tensor([[False,  True, False,  True],
        [False, False, False, False],
        [False, False, False, False]])
```

- **自赋值操作：** 想要实现`x = x + y`这样的操作时，**必须使用**`x[:] = x + y`或`x += y`.
- `B = A.clone()`：分配新内存，将`A`复制给`B`.
- `A*B`：张量的Hadamard积，将每个元素分别相乘，比如$$A*B = \begin{bmatrix}a_{11}b_{11} &  \cdots &  a_{1n}b_{1n}\\\cdots &  & \cdots\\a_{m1}b_{m1} & \cdots & a_{mn}b_{mn}\end{bmatrix}.$$
- **向量的内积：**`torch.dot(x,y)`或`torch.sum(x*y)`.
- **矩阵-向量积：**`torch.mv(A,x)`.
- **矩阵乘法：**`torch.mm(A,B)`.
- **向量范数：**`torch.norm(u)`，默认计算2-范数.

## 1.3 张量广播(broadcasting)
当两个张量的形状不匹配(但轴数相同)时，对它们进行运算，会调用**广播机制**：
1. 假定两个张量$a,b$，它们的第$k$个轴的长度分别为$a_{k},b_{k}$，取其中的最大值$new_{k} = \max(a_{k},b_{k})$.
2. 将$a,b$的第$k$个轴的长度扩大到$new_{k}$，并且复制原有的元素.
3. 对所有轴扩大完毕后，得到新张量$\hat a,\hat b$，它们形状一样.
4. 对$\hat a,\hat b$进行运算，返回一个tensor.

**广播的规则：** 从最后一个轴开始比较，要么长度相等，要么其中一个是1才能广播，否则报错.
下面是广播机制的一个实例：
> [!notion] Input

```python
a = torch.arange(3).reshape(3,1)
b = torch.arange(2).reshape(1,2)
print(a+b)
```
这里$a$是一个$3\times 1$矩阵，$b$是一个$1\times 2$矩阵，取每个轴的最大值，也就是$3\times 2$.将$a$按列复制，$b$按行复制，得到两个$3\times 2$矩阵，然后进行加号运算.
> [!success] Output

```python
tensor([[0, 1],
        [1, 2],
        [2, 3]])
```
## 1.4 梯度
假设$x$为$n$维向量，有以下规则：
- 对于所有$A\in \mathbb{R}^{m\times n}$，都有$\nabla Ax = A^{\mathrm{T}}$.
- 对于所有$A\in\mathbb{R}^{n\times m}$，都有$\nabla x^{\mathrm{T}}A = A$.
- 对于所有$A\in \mathbb{R}^{n\times n}$，都有$\nabla x^{\mathrm{T}}Ax = (A+A^{\mathrm{T}})x$.
- $\nabla||x||^{2} = \nabla x^{\mathrm{T}}x = 2x$.同样，对于任何矩阵$X$，都有$\nabla||X||_{F}^{2} = 2X$.

# 2. 自动微分
在深度学习框架（如 PyTorch）中，`backward()` 是实现**自动微分**（Automatic Differentiation, AD）的核心函数。它并不是通过符号微分或数值微分来计算梯度，而是利用 **反向传播算法**（Backpropagation）在**计算图**（Computation Graph）上高效地计算梯度。
当你用 PyTorch 进行张量运算（如加法、乘法、激活函数等）时，如果 `requires_grad=True`，PyTorch 会**动态构建一个有向无环图**（DAG），记录所有操作。
例如
>[!notion] Input

```python
x = torch.tensor(4.0)
x.requires_grad_(True)
y = x**2
z = y * 3
z.backward()
print(x.grad)
```
反向传播算法会从$z$开始，沿着计算路径反向遍历，先计算$\frac{\partial{z}}{\partial{y}} = 3$，再计算$\frac{\partial{y}}{\partial{x}} = 2x$，累积得到$\frac{z}{\partial{x}} = 6x$.
> [!success] Output

```python
tensor(24.0)
```
如果计算从$x$出发后有多个分支，那么不同分支的反向传播得到的结果会进行累加，比如这个例子
> [!notion] Input

```python
x = torch.tensor(4.0)
x.requires_grad_(True)
y = x ** 2
z = x ** 3
y.backward()
print(x.grad)
z.backward()
print(x.grad)
```
分别从$y$和$z$两个分支进行反向传播，然后叠加.
> [!success] Output
```python
tensor(8.)
tensor(56.)
```
如果需要消除之前累积的值，使用`x.grad.zero_()`即可.
## 2.1 标量函数
下面给一个例子，对函数$y = 2x^{\mathrm{T}}x$关于向量$x$求导，注意$y$是一个标量函数.正确结果为$4x$.
> [!notion] Input

```python
import torch
x = torch.arange(4.0)
x.requires_grad_(True) 
# 等价于x = torch.arange(4.0, requeires_grad = True)
print(x.grad) # 默认值是none
y = 2 * torch.dot(x,x)
# 通过调用反向传播函数来自动计算y关于x的每个分量的梯度
y.backward()
print(x.grad)
print(x.grad == 4 * x)
```
> [!success] Output

```python
None
tensor([ 0.,  4.,  8., 12.])
tensor([True, True, True, True])
```
现在计算$x$的另一个标量函数：
>[!notion] Input

```python
# 在默认情况下，PyTorch会累积梯度，需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)
```
>[!success] Output

```python
tensor([1., 1., 1., 1.])
```
## 2.2 非标量函数
`backward()`函数默认计算的是标量对所有参数的梯度.如果$y = y(x)$仍是一个张量，则必须指定参数`backward(gradient = ...)`才能反向传播.
> [!notion] Input

```python
x = torch.arange(4.0)
x.requires_grad_(True)
y = x * x
y.backward(gradient=torch.tensor([1,1,1,1]))
# 等价于y.sum().backward()
print(x.grad)
x.grad.zero_()
y = x * x
y.backward(gradient=torch.tensor([0,0,0,1]))
# 只关心最后一个分量y[3]
print(x.grad)
```
> [!success] Output

```python
tensor([0., 2., 4., 6.])
tensor([0., 0., 0., 6.])
```
# 练习
> [!question]
> 本节定义了形状为$(2,3,4)$的张量`X`，`len(X)`的输出结果是什么？

> [!notion] Input

```python
X = torch.arange(24).reshape(2,3,4)
print(len(X))
```
> [!success] Output

```python
2
```
> [!question]
> 运行`A/A.sum(axis = 1)`，看看会发生什么？分析原因.

> [!notion] Input

```python
A = torch.arange(20).reshape(5,4)
print(A/A.sum(axis = 1))
```
> [!success] Output
> 报错.因为`A.sum(axis = 1)`的第1轴消失，退化为一个长度为5的向量.它与`A`的轴数不同，不能触发广播机制运算.

> [!question] 
> 考虑一个形状为$(2,3,4)$的张量，在轴$0,1,2$上的求和输出是什么形状？

```python
x = torch.arange(24).reshape(2,3,4)
print(x)
print(x.sum(axis = 0)) #两个3x4的矩阵求和
print(x.sum(axis = 1)) #两个矩阵分别按列求和
print(x.sum(axis = 2)) #两个矩阵分别按行求和
```
> [!success] Output

```python
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
tensor([[12, 14, 16, 18],
        [20, 22, 24, 26],
        [28, 30, 32, 34]])
tensor([[12, 15, 18, 21],
        [48, 51, 54, 57]])
tensor([[ 6, 22, 38],
        [54, 70, 86]])
```