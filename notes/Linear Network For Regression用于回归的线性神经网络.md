#深度学习 #DeepLearning #线性回归 #神经网络
# 1. 合成回归数据
以下是需要用到的库.
```python
import random
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
```
## 1.1 生成数据集
以下代码片段中生成了1000个样本,每个样本有两个特征,这两个特征都服从标准正态分布.得到的矩阵$X \in \mathbb{R}^{1000\times2}$.用线性函数来生成它们的标签,即$$y = Xw+b+\varepsilon,$$其中$w\in\mathbb{R}^{2\times1}$是权重向量,$b$是标量,$\varepsilon$是服从$N(0,0.01^{2})$的噪声.在本例中,将设$w = [2,-3.4]^{\mathrm{T}},b = 4.2$.
通过编写`synthetic_data()`方法来实现这一点（此方法已经保存在`d2l`中）.
**Input**
```python
if __name__ == '__main__':
    def synthetic_data(w, b, num_examples):
        X = torch.normal(0,0.01,(num_examples,len(w)))
        y = torch.mv(X,w) + b
        y += torch.normal(0,0.01,y.shape)
        return X, y.reshape((-1,1))
    true_w = torch.tensor([2,-3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w,true_b,1000)
    # 查看每一个feature的第二个维度与labels的关系
    d2l.set_figsize()
    d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1);
    plt.show()
```

**Output**
![[所有图片/深度学习图/线性神经网络/Figure_1.png]]

## 1.2 读取数据(数据加迭代器)

编写`data_iter`方法，用以读取数据.
**Input**
```python
def synthetic_data(w, b, num_examples):
    X = torch.normal(0,0.01,(num_examples,len(w)))
    y = torch.mv(X,w) + b
    y += torch.normal(0,0.01,y.shape)
    return X, y.reshape((-1,1))
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices], labels[batch_indices]
        
if __name__ == '__main__':
    true_w = torch.tensor([2,-3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w,true_b,1000)
    for X,y in data_iter(batch_size=10,features=features,labels=labels):
        print(X,'\n',y)
        break
```

**Output(Not Unique)**
```python
tensor([[-0.0154,  0.0027],
        [ 0.0003, -0.0139],
        [-0.0012,  0.0021],
        [-0.0031, -0.0060],
        [-0.0175, -0.0071],
        [ 0.0080, -0.0049],
        [-0.0003, -0.0222],
        [ 0.0181, -0.0142],
        [-0.0238, -0.0099],
        [ 0.0036,  0.0125]]) 
 tensor([[4.1758],
        [4.2276],
        [4.1742],
        [4.2139],
        [4.1808],
        [4.2387],
        [4.2934],
        [4.2870],
        [4.1672],
        [4.1641]])
```
这样的迭代效率很低，不如深度学习框架中内置的迭代器.
# 2. 实现线性回归
以下是需要用到的库.
```python
import random
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
```
## 2.1 手动实现
**初始化模型参数**
```python
w = torch.normal(0,0.01,size=(true_w.shape),requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```
**定义模型、损失与优化器**
```python
def linreg(X,w,b):
    return torch.matmul(X,w) + b
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```
注意在损失函数返回时，要用`y.reshape`防止因维度不匹配造成错误.随机梯度下降方法中`with torch.no_grad()`的作用是**暂时禁用梯度计算**，防止PyTorch追踪`param`的计算并更新梯度.
**实现训练**
```python
if __name__ == '__main__':
    true_w = torch.tensor([2,-3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w,true_b,1000)
    w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss
    batch_size = 10
    for epoch in range(num_epochs):
        for X,y in data_iter(batch_size=batch_size,features=features,labels=labels):
            l = loss(net(X,w,b),y)
            l.sum().backward()
            sgd([w,b],lr = lr, batch_size=batch_size)
        with torch.no_grad():
            train_loss = loss(net(features,w,b),labels)
            print(f'epoch {epoch + 1}, train loss {float(train_loss.mean()):f}')
```
在`l.sum().backward()`处，尽管能够成功实现反向传播，但是`l.sum()`有**溢出的风险**.在现代框架中，更多使用`l.mean().backward()`并在损失函数中将参数更新为`param -= lr * param.grad`，无需再除以`batch_size`.
## 2.2 简洁实现
```python
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn
import matplotlib.pyplot as plt
```
生成数据的部分与上节一样.
```python
true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w,true_b,1000)
```
对于数据迭代器，PyTorch将数据封装到`DataLoader`类中，它接收一个数据集`dataset`（包含特征，标签等）、批量大小以及是否打乱(shuffle)，并自动处理以下工作：
- **batching** (批处理): 自动将单个样本组合成批次（batch）.
- **shuffling** (打乱): 每个 epoch 自动打乱数据顺序.
- **parallel loading** (多进程加载): 利用多核 CPU 并行读取和预处理数据，加速训练.
- **memory pinning**: 加速数据从 CPU 到 GPU 的传输.
如果我们的数据都是`Tensor`，PyTorch还提供了更方便的`TensorDataset`类，将这些`Tensor`包装成一个`Dataset`，然后就可以传入`DataLoader`中.我们完全不需要单独写一个方法来获取数据迭代器，而是像这样：
```python
dataset = data.TensorDataset(features,labels)
data_iter = data.DataLoader(dataset,batch_size=10,shuffle=True)
```
Input
```python
if __name__ == '__main__':
    true_w = torch.tensor([2,-3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w,true_b,1000)
    dataset = data.TensorDataset(features,labels)
    data_iter = data.DataLoader(dataset,batch_size=10,shuffle=True)
    net = nn.Sequential(nn.Linear(in_features=2,out_features=1))
    net[0].weight.data.normal_(0,0.01)
    net[0].bias.data.fill_(0)
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(params=net.parameters(),lr = 0.03)
    num_epochs = 3
    for epoch in range(num_epochs):
        for X,y in data_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features),labels)
        print(f'epoch {epoch + 1}, loss {float(l):f}')
```
它不再像第一版一样需要指定`model`，而是指定网络`net`，损失函数`loss`，优化器`trainer`，随后用`trainer.step()`来更新.
`MSELoss`类计算均方误差$$l^{(i)}(w,b) = \frac{1}{n^{(i)}}(\hat y^{(i)}-y^{(i)})^{2},$$其中$n^{(i)}$是此批次的数量.实际上`MSELoss()`有如下可选参数:
- `reduce(bool)`:为`True`时,返回标量损失值.为`False`时,返回一个损失张量,每一个位置上都是对应的$(\hat y^{(i)}-y^{(i)})^{2}$.
- `size_average(bool)`:当`reduce`为`True`时有效.为`True`时,返回的`loss`为平均值.为`False`时,返回各样本的`loss`之和.
- `reduction(str)`:共有三种可选项.
- - `'mean'`(默认)：返回该批次的平均`loss`.
- - `'sum'`：返回该批次的`loss`之和.
- - `'none'`：直接返回`loss`张量，不进行聚合.

# 3. 权重衰减（待修改）
在随机梯度下降法中，通过损失函数对参数$({\mathbf w},b)$求梯度来调整参数值，其中损失函数$$L(\mathbf{w},b) = \frac{1}{n}\sum_{i=1}^{n} \frac{1}{2}(\mathbf{w}^{\mathrm{T}}\mathbf{x}^{(i)} + b - \mathbf{y}^{(i)})^{2},$$
**权重衰减**旨在控制参数的取值范围，以降低模型复杂度.在此处，权重衰减为了控制$\mathbf{w}$的范围，预期$||\mathbf{w}||$尽可能小，通过向损失中加入惩罚项$$\frac{\lambda}{2}||w||_{2}^{2}$$来实现这一目标.系数$\frac{1}{2}$是为了求导的方便，$\lambda$是一个超参数，用于控制这一惩罚的占比，如果$\lambda$设置的很大，则模型会专注于减小权重向量大小的惩罚，而不是减小训练误差.选择使用$\mathscr{l}_{2}$范数的原因是，$\mathbf{w}$中过大的分量会引起巨大的惩罚，这使算法偏向于将权重均匀分布在更多特征上.
权重衰减又叫做$\mathscr{l}_{2}-$正则化，小批量随机梯度下降法更新如下：$$\mathbf{w}\leftarrow \mathbf{w} - \eta\left(\lambda\mathbf{w} + \frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}}\mathbf{x}^{(i)}(\mathbf{w}^{\mathrm{T}}\mathbf{x}^{(i)}+b-\mathbf{y}^{(i)})\right).$$
# 4. 高维线性回归（待修改）
## 4.1 实现
下面通过一个实例来实现高维线性回归，同时展示如何通过权重衰减来优化过拟合现象.为了使过拟合现象更明显，只采取20个数据进行训练，每个数据都有200个维度.
创建数据利用$$y = 0.05 + \sum_{i=1}^{200}0.01x_{i}+\varepsilon, ~\varepsilon \sim N(0,0.01^{2})$$
**Input**
```python
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

class Data(d2l.DataModule): #用于生成数据的类
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        # num_inputs表示输入特征的维度
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, num_inputs) # X的初始值是带有一点点误差的
        noise = torch.randn(n, 1) * 0.01
        w, b = torch.ones(num_inputs, 1) * 0.01, 0.05
        self.y = torch.matmul(self.X, w) + b + noise
    def get_dataloader(self, train):
        i = slice(0,self.num_train) if train else (self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)

def l2_penalty(w): # l2范数罚函数
    return (w**2).sum() / 2

class WeightDecay(d2l.LinearRegressionScratch): # 权重衰减模型，继承'从零实现线性回归'类
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        # 增加了一个超参数lambd
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()
    def loss(self,y_hat, y): # 增加了惩罚的损失
        return super().loss(y_hat, y) + self.lambd*l2_penalty(self.w)


def train_scratch(lambd): # 此函数实现训练，调整lambda的值得到不同的结果
    model = WeightDecay(num_inputs=200, lambd=lambd, lr=0.01)
    model.board.yscale = 'log'
    trainer.fit(model, data)
    plt.show()
    print('L2 norm of w: ',float(l2_penalty(model.w)))
```
- 若不使用$\mathscr{l}_{2}-$正则化进行训练（即$\lambda = 0$），可以明显看见过拟合现象.

**Input**
```python
data = Data(num_train=20, num_val=100,num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)
train_scratch(0)
```
**Output**
![[Figure_6.png]]
```python
L2 norm of w:  0.010975048877298832
```
训练误差下降，但验证误差却没有下降.
- 若使用$\mathscr{l}_{2}-$正则化，令$\lambda = 3$.

**Input**
```python
data = Data(num_train=20, num_val=100,num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)
train_scratch(3)
```
**Output**
![[Figure_7.png]]
```python
L2 norm of w:  0.0017459624214097857
```
可见训练误差增加了，但验证误差减少了，这正是预期的效果.
## 4.2 权重衰减的简洁实现
默认情况下，PyTorch会为$\mathbf{w}$和$b$都进行权重衰减.PyTorch已经将权重衰减整合进算法中，使得我们在选择优化器时，可以直接指定对哪个参数进行权重衰减.
**Input**
```python
# 将上面的WeightDecay类修改为
class WeightDecay(d2l.LinearRegression): # 权重衰减模型，继承'线性回归'类
    def __init__(self,lambd,lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.lambd = lambd
    def configure_optimizers(self): # 选择优化器时，可以指定参数进行权重衰减
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay' : self.lambd},
            {'params': self.net.bias}
        ],lr=self.lr)
        
# 在主文件中直接调用
data = Data(num_train=20, num_val=100,num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)
model = WeightDecay(lambd=3, lr=0.01)
model.board.yscale = 'log'
trainer.fit(model, data)
plt.show()
print('L2 norm of w: ', float(l2_penalty(model.get_w_b()[0])))
```
**Output**
![[Figure_8.png]]
```python
L2 norm of w:  0.013831708580255508
```
# Exercise（待修改）
> [!question] 
> 查看框架文档，看看提供了那些损失函数.特别是，用Huber的鲁棒损失函数替换平方损失.$$l(y,y^{'}) = \begin{cases}&|y-y^{'}| - \frac{\sigma}{2}&,if |y-y^{'}|>\sigma\\ &\frac{1}{2\sigma}(y-y^{'})^{2}&,else\end{cases}.$$

实际上，除了平方损失函数`MSELoss()`，PyTorch中还提供了另外18种不同的损失函数.但是并未直接提供得到损失值的方法，需要自己手动求出.
**Input**
```python
# 在LinearRegression类中修改loss方法为
def loss(self, y_hat, y):
        """Defined in :numref:`sec_linear_concise`"""
        fn = nn.SmoothL1Loss()
        return fn(y_hat, y)
        
# 然后在主文件中输入
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
model = d2l.LinearRegression(lr=0.03) # 学习率
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
train_dataloader = data.get_dataloader(train=True)
trainer = d2l.Trainer(max_epochs=3) # 最大训练轮次
trainer.fit(model, data)
plt.show()
total_loss = 0
num_train = 0
for X,y in train_dataloader:
    y_hat = model(X)
    batch_mean_loss = model.loss(y_hat, y)
    total_loss += batch_mean_loss * len(y)
    num_train += len(y)
print(f'Loss: {total_loss / num_train}')
w, b = model.get_w_b()
print(f'error in estimating w: {data.w - w.reshape(data.w.shape)}')
print(f'error in estimating b: {data.b - b}')
```
**Output(Not unique)**
![[Figure_2.png]]
```python
Loss: 2.1387643814086914
error in estimating w: tensor([ 1.0581, -2.1611])
error in estimating b: tensor([2.1389])
```
用这个损失函数拟合的效果并不好.
> [!question] 
> 如何访问模型权重`w`的梯度？

注意到`LinearRegression`类中提供了`get_w_b()`方法，其返回值是`self.net.weight.data`以及`self.net.bias.data`.`weight`和`bias`都是`TensorBase`类的实例，ctrl+右键它们进入到`TensorBase`类中，发现里面提供了`grad`属性，于是在`LinearRegression`类中手动添加如下方法：
```python
# 添加到LinearRegression类中
def get_w_grad(self):
        return (self.net.weight.grad)
```
然后在主函数部分输入
**Input**
```python
"""生成数据和训练数据的部分同上题"""
w_grad = model.get_w_grad()
print(f'grad of w:{w_grad}')
```
**Output(Not unique)**
```python
grad of w:tensor([[-0.6527,  0.4935]])
```
值得注意的是`.grad`只会返回最后一轮训练中的最后一批次数据的梯度值，因为PyTorch中每进行完梯度的计算后都需要清零，否则梯度值会累加.在`Trainer`类中，每当调用`fit()`方法，大致会执行：
- 执行`loss = ...`（建立计算图，计算损失）
- 执行 `self.optim.zero_grad()` (清空旧梯度)
- 执行 `loss.backward()` (f反向传播计算出最后一个 Batch 的梯度，并填入 `model.net.weight.grad`)
- 执行 `self.optim.step()` (参数更新)

由于最后一次的梯度没有清零，所以可以被读取.但通常标准的写法是
```python
loss = ... # 计算图构建，计算损失
loss.backward() # 反向传播，计算梯度
self.optim.step() # 更新参数
self.optim.zero_grad() # 清空梯度
```

> [!question]
> 如果更改学习率和训练周期，对解决方案有什么影响，它会持续改进吗？

将学习率扩大十倍（从0.03改为0.3），采用`MSELoss()`方法计算损失，得到输出
![[Figure_3.png]]
```python
Loss: 0.00010045830276794732
error in estimating w: tensor([0.0018, 0.0010])
error in estimating b: tensor([-0.0016])
```
可见从第一次训练过后模型就几乎完美拟合了.
将学习率调回0.03，但训练轮次为10，得到输出
![[Figure_4.png]]
```python
Loss: 0.00010059300984721631
error in estimating w: tensor([ 0.0008, -0.0003])
error in estimating b: tensor([-0.0001])
```
效果比调整学习率更好，说明多次训练可以显著提升模型表现.
如果采用拟合效果稍差的损失函数`SmoothL1Loss()`，学习率为0.03，训练轮次为10，可以看到更明显的趋势
![[Figure_5.png]]
```python
Loss: 0.0013581446837633848
error in estimating w: tensor([ 0.0301, -0.0351])
error in estimating b: tensor([0.0240])
```
随着训练次数增加，模型的表现越来越好，但是它的拟合效果仍然不够好.
> [!question] 
> 在实现权重衰减时，更改权重$\lambda$的值，绘制训练损失与验证损失关于$\lambda$的函数，观察到了什么？

对于每一个$\lambda$，需要求出相应的`train_loss`和`val_loss`，这并不困难，只需要添加一个求损失的方法即可.
**Input**
```python
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
class Data(d2l.DataModule): #用于生成数据的类
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        # num_inputs表示输入特征的维度
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, num_inputs) # X的初始值
        noise = torch.randn(n, 1) * 0.01
        w, b = torch.ones(num_inputs, 1) * 0.01, 0.05
        self.y = torch.matmul(self.X, w) + b + noise
    def get_dataloader(self, train):
        i = slice(0,self.num_train) if train else (self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)
def l2_penalty(w): # l2范数罚函数
    return (w**2).sum() / 2
    
class WeightDecay(d2l.LinearRegression): # 权重衰减模型
    def __init__(self,lambd,lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.lambd = lambd
    def configure_optimizers(self): # 指定参数w进行权重衰减
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay' : self.lambd},
            {'params': self.net.bias}
        ],lr=self.lr)
        
def get_loss(model, data_loader): # 求损失的方法
    total_loss = 0
    total_num = 0
    for X,y in data_loader:
        y_hat = model(X)
        batch_loss = model.loss(y_hat, y)
        batch_size = X.shape[0]
        total_loss += batch_size * batch_loss
        total_num += batch_size
    return total_loss / total_num # 返回的是一维tensor

def evaluate_loss_with_different_lambd(lambd_values):
    train_loss = []
    val_loss = []
    for lambd in lambd_values:
        model = WeightDecay(lambd, lr = 0.01)
        model.board.yscale = 'log'
        trainer.fit(model, data)
        train_loader = data.get_dataloader(train=True)
        val_loader = data.get_dataloader(train=False)
        train_loss.append(
        get_loss(model,train_loader).detach().numpy()) 
        val_loss.append(
        get_loss(model,val_loader).detach().numpy())
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(lambd_values, train_loss, label='Training Loss', marker='o')
    plt.plot(lambd_values, val_loss, label='Validation Loss', marker='s')
    plt.xlabel('Lambda (Weight Decay Parameter)')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs Lambda')
    plt.xscale('log')  # 使用对数刻度显示lambda值
    plt.legend()
    plt.grid(True)
    plt.show()
    return train_loss, val_loss

data = Data(num_train=20, num_val=100,num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)
lambd_values = [0, 0.001, 0.01, 0.1, 1.0, 2.0, 5.0, 10.0]
evaluate_loss_with_different_lambd(lambd_values)
```
特别注意当创建`train_loss = []`和`val_loss = []`时，它们都是`numpy`对象.而在`get_loss()`方法中，所有的计算都是在张量的基础上进行的（比如`batch_loss = model.loss(y_hat, y)`，其结果不是一个数，而是一个一维张量）.所以在向`train_loss`和`val_loss`中加入数据时，要用`tensor.detach().numpy()`方法将张量转化为`numpy`对象，否则会报错.
**Output**
![[Figure_9.png]]
从结果中可以观察到，当$\lambda = 0$时发生过拟合，验证损失非常大.随着$\lambda$的增加，过拟合现象逐渐消失，验证损失呈减少趋势.随着$\lambda$变得过大，模型发生**欠拟合现象**，验证损失逐渐增加.这是因为惩罚项$||w||_{2}^{2}$逐渐占据主导，模型倾向于把$||w||^{2}_{2}$变得很小（趋于0）以至于丢失了$w$的真正信息（真正的$w = 0.01$），相当于丢失了数据中$X$的信息，使得整个模型几乎只依赖于偏置$b$.
> [!question] 
> 使用验证集找到$\lambda$的最优值，它真的是最优值吗？这重要吗？

如上题的结果所见，$\lambda$的最优值为0.1（仅在上题数据的情况下，训练样本为20，每个样本特征维度为200）.找到$\lambda$的最优值非常重要，此过程就是一种**超参数调优**.当$\lambda$过小或过大，模型会发生过拟合或欠拟合，只有找到最优的$\lambda$值，模型才能展现出良好的对于$\lambda$的鲁棒性和泛化能力.一般来说，选取对数间隔均匀分布的$\lambda$进行训练，然后通过验证集损失找到最优$\lambda$（就像上题那样）.最后再用测试集对所得的$\lambda$值再次验证，评估其在测试集上的表现.
> [!question] 
> 如果使用$\mathscr{l}_{1}$范数进行惩罚，权重$\mathbf{w}$的更新方程是什么？

由于$||\mathbf{w}||_{1}$对$\mathbf{w}$的梯度是$\mathrm{sign}(\mathbf{w})$，因此更新方程是$$\mathbf{w}\leftarrow \mathbf{w} - \eta\left(\lambda\mathrm{sign}(\mathbf{w}) + \frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}}\mathbf{x}^{(i)}(\mathbf{w}^{\mathrm{T}}\mathbf{x}^{(i)}+b-\mathbf{y}^{(i)})\right).$$
