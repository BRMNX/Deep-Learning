#深度学习 #DeepLearning #分类 #神经网络 #softmax
# 1. Softmax回归
本节全是理论分析.
## 1.1 分类
假定我们有一张图像$X$，它有4个像素，用一个四维向量存储它的信息，即$X \in \mathbb{R}^{4\times 1}$.给定一些类别，比如$\{\mathrm{cat},\mathrm{dog},\mathrm{chicken}\}$，分类问题就是通过计算以分辨出$X$属于哪个类别.
最直接的想法是输出一个$3 \times 1$的向量（标签）$y$.如果$y = (1,0,0)$就表示$X$属于$\mathrm{cat}$类别，如果$y = (0,1,0)$就表示$X$属于$\mathrm{dog}$类别，如果$y = (0,0,1)$就表示$X$属于$\mathrm{chicken}$类.
但在实际情况中，我们并不这样做，而是将输出$y$的每个分量作为概率，哪个分量的值更大，就表示$X$属于那一类的概率更高.
## 1.2 线性模型
在上面的例子中，如果使用线性模型，可以表示为$$o_{1} = w_{11}x_{1}+w_{12}x_{2}+w_{13}x_{3}+w_{14}x_{4}+b_{1}$$$$o_{2} = w_{21}x_{1}+w_{22}x_{2}+w_{23}x_{3}+w_{24}x_{4}+b_{1}$$
$$o_{1} = w_{31}x_{1}+w_{32}x_{2}+w_{33}x_{3}+w_{34}x_{4}+b_{1}$$
其中输出$y = (o_{1},o_{2},o_{3})$.相应的神经网络如图所示，输出层是一个**全连接层**，用向量表示为$\mathbf{o = Wx+b}$.
![[Pasted image 20260307121343.png]]
但是这样的模型存在如下问题：
- $o_{j}$的和不一定为1.
- 无法保证每一个$o_{j}$非负.
- 无法保证每一个$o_{j}$不超过1.

因此需要一种机制来“压缩”输出，它就是softmax函数.
## 1.3 softmax函数
使用指数函数$\exp(o_{j})$来实现，使得$$\mathbf{\hat y} = \mathrm{softmax}(\mathbf{o}),\mathrm{where}~\hat y_{j} =  \frac{\exp(o_{j})}{\sum_{j}\exp(o_{j})}.$$
现在，假定有$n$个样本的小批量数据$\mathbf{X}\in \mathbb{R}^{n\times d}$，且输出类别有$q$种.那么权重$\mathbf{W}\in \mathbb{R}^{d\times q}$，偏置$\mathbf{b}\in \mathbb{R}^{q\times 1}$，且$$\mathbf{O = WX+b},\mathbf{Y = }\mathrm{softmax}(\mathbf{O}).$$
对于指数值可能会溢出的现象，深度学习框架会自动处理.
## 1.4 损失函数
前面说过,输出$\mathbf{\hat y}$的每个分量实际上是一个概率,例如$\mathbf{\hat y}_{1} = P(\mathbf{y} = \mathrm{cat}|\mathbf{x})$,现在需要一个损失函数来评估预测的正确性.由于$\mathbf{x}$的每个分量是独立的,因此$$P(\mathbf{Y}|\mathbf{X}) = \prod_{i=1}^{n}P(\mathbf{y}^{(i)}|\mathbf{x}^{(i)}).$$我们想让这个概率值最大化,引进负对数似然函数$$-\log P(\mathbf{Y}|\mathbf{X}) = \sum_{i=1}^{n}-\log P(\mathbf{y}^{(i)}|\mathbf{x}^{(i)}) = \sum_{i=1}^{n}l(\mathbf{\hat y}^{(i)},\mathbf{y}^{(i)}),$$其中$$l(\mathbf{\hat y},\mathbf{y}) = -\sum_{j=1}^{q}y_{j}\log \hat y_{j}.$$最大化原来的概率,等价于最小化$l(\mathbf{\hat y},\mathbf{y})$.这个函数称为**交叉熵损失**.它有以下几个性质:
1. 由于输出$\mathbf{\hat y}$的每个分量$\hat y_{j}$都是一个概率,值小于1,因此损失函数值非负,下界为0.
2. 不可能有绝对准确的预测,使得某一个$\hat y_{j} = 1$.因为相应的分量$\hat y_{j}$趋于1需要softmax函数的输入$o_{j}$趋于正无穷,而其他的$o_{i}(i\neq j)$都趋于负无穷.此时$\hat y_{i} = \mathrm{softmax}(o_{i}) \rightarrow 0$.代入到损失函数中会使得$-\log \hat y_{i} \rightarrow \infty$,这会使得损失无穷大,显然是不合理的.

> [!notion] 为什么不使用均方损失MSELoss()?
> 假定仍然使用$l(\hat y,y) = \frac{1}{n}\sum_{i} \frac{1}{2}(\hat y_{i}- y_{i})^{2}$来计算损失,那么当需要计算梯度更新参数时$$\frac{\partial{l}}{\partial{o}} = \frac{\partial{l}}{\partial{\hat y}}\frac{\hat y}{\partial{o}},$$其中$$\frac{\partial{l}}{\partial{\hat y}} = (\hat y - y),$$而$$\begin{align}\frac{\partial{\hat y}}{\partial{o}_{i}} &= \frac{\exp{(o_{i})}\sum\exp(o_{j}) - (\exp(o_{i}))^{2}}{(\sum\exp (o_{j}))^{2}}\\&=\hat y_{i} - \hat y_{i}^{2} \\ &= \hat y_{i}(1-\hat y_{i}),\end{align}$$如果模型的损失很大，有某些$\hat y_{i}$趋于0或1，但在这种方法下它的梯度却趋于0，模型不能正确的更新这个偏差的参数，导致模型的学习效率很慢，这就是典型的梯度消失.

# 2. Fashion-MNIST数据集
#FashionMNIST
本节所用到的库
```python
import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from d2l import torch as d2l
import matplotlib.pyplot as plt
```
## 2.1 加载数据
Fashion-MNIST数据集包含10个类别的图像，每张图像为$28\times28$像素.每个类别中有6000个训练数据和1000个验证数据，也就是整个数据集中有60000个训练数据和10000个验证数据.
由于Fashion-MNIST数据集非常有用，所有主流的框架都提供其预处理版本，在PyTorch中为`FashionMNIST`类.
**Input**
```python
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式的张量，并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
```
这些图像是灰度图（单通道），用`mnist_train[a][b = 0,1]`来读取第$a$张图像的第$b$个属性，`b = 0`表示该图像，`b = 1`表示对应的标签`y`.
**Input**
```python
print(mnist_train[0][0].shape)
print(mnist_train[0][1])
```
**Output**
```python
torch.Size([1, 28, 28])
9
```
**每一张图像用张量（通道数，高度，宽度）来存储**，标签用整数来对应类别（`d2l`类中提供了`get_fashion_mnist_labels()`方法实现对应）.
```python
labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat','sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
```
以下是训练集中18个样本的图像与标签.
**Input**
```python
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
d2l.show_images(X.reshape(18, 28, 28), 2, 9, titles=d2l.get_fashion_mnist_labels(y));
plt.show()
```
**Output**
![[所有图片/深度学习图/线性神经网络/Figure_2.png]]
## 2.2 读取小批量
我们使用内置的数据迭代器，而不是自己编写.
**Input**
```python
batch_size = 256
train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=d2l.get_dataloader_workers())
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')
```
其中`d2l.get_dataloader_workers()`直接返回整数4，即调用4个进程来读取数据.
**Output**
读取整个数据集的时间
```python
6.85 sec
```
下面我们整合所有组件，一次性得到`train_iter`和`test_iter`，并通过`resize`参数将图片展成$64\times 64$像素，方便以后使用.
**Input**
```python
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if(resize):
        trans.insert(0,transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=r'C:\Users\27093\Desktop\Deep Learning\data',train=True,transform=trans,download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root=r'C:\Users\27093\Desktop\Deep Learning\data',train=False,transform=trans,download=False)
    train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=d2l.get_dataloader_workers())
    test_iter = data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=d2l.get_dataloader_workers())
    return train_iter, test_iter

if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size,resize=64)
    for X,y in train_iter:
        print(X.shape,X.dtype,y.shape,y.dtype)
        break
```
PyTorch的`transforms.py`文件中提供了多种转换器，用来处理不同类型的数据，比如这里的`ToTensor`和`Resize`转换器.将不同的转换器添加到列表中，最后使用`Compose`将所有转化器结合起来.当创建数据集`dataset`实例时，会传入转换器.
- 在查阅了[torchvision.transforms API](https://docs.pytorch.org/vision/stable/transforms.html#:~:text=Torchvision%20supports%20common%20computer%20vision%20transformations%20in%20the,and%20augment%20data%2C%20for%20both%20training%20or%20inference.)后，官方更推荐调用`from torchvision.transforms import v2`，其功能更多且运行速度更快.此后所有用到`transforms`的地方均以`v2`替代.

**Output**
```python
torch.Size([256, 1, 64, 64]) torch.float32 torch.Size([256]) torch.int64
```
## 2.3 问题
> [!question]
> 将batch_size减小到1，会影响性能吗？

会影响性能，此时读取整个训练集需要16.89秒.
# 3. Softmax回归实现

## 3.1 手动实现
**Input（讲解都在注释里）**
手动实现与改错真的花了比较长的时间.
```python
import torch
import torchvision
from torchvision.transforms import v2
from torch.utils import data
from d2l import torch as d2l
from IPython import display
import matplotlib.pyplot as plt

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [v2.ToTensor()]
    if(resize):
        trans.insert(0,v2.Resize(resize))
    trans = v2.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=r'C:\Users\27093\Desktop\Deep Learning\data',train=True,transform=trans,download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root=r'C:\Users\27093\Desktop\Deep Learning\data',train=False,transform=trans,download=False)
    train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=d2l.get_dataloader_workers())
    test_iter = data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=d2l.get_dataloader_workers())
    return train_iter, test_iter
    
def softmax(X:torch.Tensor) -> torch.Tensor:
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=1,keepdim=True) # 按行求和
    return X_exp / partition

def net(X:torch.Tensor):
    """先将X的形状重新调整，前面自由分配，只要最后一个维度是特征数量w.shape[0]就行"""
    return softmax(torch.matmul(X.reshape((-1,w.shape[0])),w) + b)

def cross_entropy(y_hat:torch.Tensor,y:torch.Tensor):
    """range(len(y_hat))其实就是0,1,2,...,N，作用是当下标，表示y_hat取到了第几个样本"""
    """取到样本后，它有很多个分量，每个分量都是概率值，而y_hat[...,y]就是要把此样本属于y那个类别的概率值单独提取出来"""
    return - torch.log(y_hat[range(len(y_hat)),y])

def accuracy(y_hat:torch.Tensor, y:torch.Tensor):
    # 当y_hat不止一个类别并且不止一个样本时才需要找最大下标
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    # 通过比较，得到一个全是True或False的张量compare
    compare = (y_hat.type(y.dtype) == y)
    # 将类型转换为数字，求和后返回正确预测的数量
    return float(compare.type(y.dtype).sum())

def evaluate_accuracy(net,data_iter:data.DataLoader):
    # 设为评估模式，目前还用不到
    if isinstance(net, torch.nn.Module):
        net.eval()
    # 维护两个变量的累加器，记录测试准确率metric[0]和样本总数metric[1]
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(accuracy(net(X),y),y.numel()) # 向累加器中加入该批次的准确率和数量
    return metric[0] / metric[1]

def updater(batch_size):
    return d2l.sgd([w,b],lr=lr,batch_size=batch_size)

def train_epoch(net, train_iter, loss, updater):
    # 设置模型为训练模式，目前暂时用不到
    if isinstance(net,torch.nn.Module):
        net.train()
    # 维护三个变量的累加器，存储训练损失metric[0]，训练准确率metric[1]，样本总数metric[2]
    metric = d2l.Accumulator(3)
    for X,y in train_iter:
        y_hat = net(X) # 预测值
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用自制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()),accuracy(y_hat, y), y.numel()) #更新累加器
    return metric[0]/metric[2], metric[1]/metric[2]

def train(net, train_iter:data.DataLoader, test_iter:data.DataLoader, num_epochs, loss, updater):
    # d2l中辅助画图的工具，这样我不需要自己画图了
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net,train_iter,loss,updater) # 训练集记录损失与准确率
        test_acc = evaluate_accuracy(net,test_iter) # 测试集只记录准确率
        animator.add(epoch + 1, train_metrics + (test_acc,))
    # 后续的代码用于代码层面检查模型的学习效果，判断代码是否有误
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

num_inputs = 784
num_outputs = 10
num_epochs = 10
batch_size = 256
lr = 0.1
w = torch.normal(0,0.01,size=(num_inputs,num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

if __name__ == '__main__':
train_iter, test_iter = load_data_fashion_mnist(batch_size)

train(net=net,train_iter=train_iter,test_iter=test_iter,num_epochs=num_epochs,loss=cross_entropy,updater=updater)

plt.show()
```
**Output**
由于手动编写的东西太粗糙了，训练时间长且效果一般.
![[所有图片/深度学习图/线性神经网络/Figure_3.png]]
## 3.2 简易实现
实际上很多部分与手动实现是一样的.
**Input**
```python
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import matplotlib

# 强制使用 TkAgg 后端，这样可以在普通 Python 脚本中弹出窗口显示图片
matplotlib.use('TkAgg')

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
nn.init.normal_(net[1].weight, mean=0, std=0.01)
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
plt.show()
```
softmax函数在`nn.CrossEntropyLoss`中隐式地实现，它会一直深入底层C++函数计算，它不直接利用softmax的原始定义，而是减去每个样本中的$\max(o_{j})$，避免指数溢出.
==**源码解析**==
注意这里用来初始化参数的`nn.init`方法实际上是对参数`net[1].weight`的初始化的覆盖.当调用模型`nn.Linear`时，其构造函数就自动初始化了`weight`和`bias`（并且它们的值服从均匀分布$U(-\sqrt k,\sqrt k)$，其中$k$是`in_features`的倒数，在这里就是$1 / 784$）.手动调用`nn.init`方法实际上是覆盖了参数的初始化，并且`normal_`指定了`weight`服从正态分布.
查看`init.py`中的`normal_`方法实现：
```python
def normal_(tensor: Tensor, mean: float = 0., std: float = 1.) -> Tensor:
    r"""Fills the input Tensor with values drawn from the normal
    distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.normal_(w)
    """
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(normal_, (tensor,), tensor=tensor, mean=mean, std=std)
    return _no_grad_normal_(tensor, mean, std)
```
它首先检查传入的 `tensor` 是否是一个“特殊张量”（例如来自 `faiss`、`xla` 或其他重载了 torch 函数的库）.如果是，它会把控制权交给 `torch.overrides` 系统，让那个特殊的库去处理初始化逻辑.
**正常流程**：如果是一个普通的 PyTorch Tensor（绝大多数情况），它就忽略 `if` 块，直接向下执行，调用内部函数 `_no_grad_normal_`.
再来看`_no_grad_normal_`方法实现：
```python
def _no_grad_normal_(tensor, mean, std):
    with torch.no_grad():
        return tensor.normal_(mean, std)
```
它禁用梯度更新，并且调用`_TensorBase`类中的`normal_`方法，让张量真正地发生更新.此方法的实现：
```python
def normal_(self, mean: _float=0, std: _float=1, *, generator: Optional[Generator]=None) -> Tensor: ...
```
`tensor.normal_(...)` 让PyTorch调用**C++ 底层引擎**，Python 解释器会立刻跳转到 C++ 代码中的对应函数生成随机数.
**Output**
![[所有图片/深度学习图/线性神经网络/Figure_4.png]]
## 3.3 问题
> [!question]
> 增加训练轮次，为什么准确率会在一段时间后下降？如何解决？

令`max_epochs = 30`，得到
**Output**
![[Figure_12.png]]
验证准确率确实有小段的下跌，然后又调整回来.这种现象出现主要有以下几种原因：
**A. 小批量数据的随机性.** 某个特性的batch中的数据可能噪声影响过大，导致模型在训练完此batch后参数更新的步长较大，从而该轮次在验证集中表现不佳.**解决方法是增大`batch_size`，以减小梯度的随机噪声.**
**B. 学习率过大.** 虽然训练损失整体上在不断下降，但是可以看到它在局部反复震荡，这说明参数在不断的重复“接近-远离”最优值的过程.造成此现象的原因是随机梯度下降的步长（学习率）过大，导致参数无法平滑收敛.**可以尝试令`lr = 0.05`或`lr = 0.01`，或者对学习率使用权重衰减.**
> [!question]
> 增加学习率会发生什么？比较不同学习率的损失曲线，哪一个效果更好？

令`max_epochs = 10,lr = 0.5`，得到
**Output**
![[Figure_13.png|350]]
过大的学习率让参数在最优点附近大幅度的震荡，致使验证损失不能有效降低.
（为了得到效果最好的，要多次尝试）