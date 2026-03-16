#深度学习 #DeepLearning #LeNet #MNIST
# 1. LeNet
**LeNet**是最早发布的卷积神经网络之一，因其在计算机视觉任务（识别手写数字）中的高校性能而受到广泛关注.
总体来看LeNet(LeNet-5)由两个部分组成：
- 卷积编码层：由两个卷积块构成.
- 全连接层密集块：由三个全连接层组成.

![[Pasted image 20260315162129.png]]
每个卷积块的基本单元是一个卷积层，经过sigmoid激活函数，再经过平均汇聚层（实际上，ReLU函数+最大汇聚层的效果更好，但当时还没有发现）.
**输入图片**是单通道的`(28,28)`灰度图.**第一个卷积层**的卷积核是`(5,5)`的，填充为`2`以保持输出大小不变，输出通道有`6`个.而后的汇聚层有`6`个大小为`(2,2)`的汇聚核，步幅默认（`=2`），这样每个矩阵输出大小减少一半，变为`(14,14)`.**第二个卷积层**的卷积核也是`(5,5)`的，输入通道为`6`，输出通道为`16`，不填充.卷积后的矩阵大小是`(10,10)`.流经`16`个大小为`(2,2)`的平均汇聚层，输出矩阵的大小减小一半为`(5,5)`.为了将卷积块的输出传递到**全连接密集块**，仍然需要将张量展平，而后依次通过
```python
nn.Linear(16*5*5,120),nn.Sigmoid(),
nn.Linear(120,84),nn.Sigmoid(),
nn.Linear(84,10),nn.Sigmoid()
```
由于标签有`10`个类别，所以最后输出`10`个维度.
首先试验它在Fashion-MNIST数据集中的表现.
**Input**
```python
import matplotlib
import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')
def evaluate_accuracy_gpu(net, data_iter,device=None):
    if isinstance(net, nn.Module):
        net.eval()
    metric = d2l.Accumulator(2)
    if device is None:
        device = next(net.parameters()).device
    with torch.no_grad():
        for X,y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X),y), y.numel())
    return metric[0] / metric[1]

def train(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter, device)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    plt.show()
if __name__ == '__main__':
    net = nn.Sequential(
        nn.Conv2d(1,6,kernel_size = 5,padding = 2),nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2,stride=2),
        nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2,stride=2),
        nn.Flatten(),
        nn.Linear(16*5*5,120),nn.Sigmoid(),
        nn.Linear(120,84),nn.Sigmoid(),
        nn.Linear(84,10))
    
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    lr = 0.9
    num_epochs = 10
    device = d2l.try_gpu()
    train(net, train_iter, test_iter, num_epochs, lr, device)
```
==**源码解析**==
计算损失和梯度时直接使用`l.backward()`是因为`nn.CrossEntropyLoss`的默认参数`reduction = 'mean'`，即损失是标量，可以直接反向传播.
**Output**
```python
loss 0.473, train acc 0.822, test acc 0.822
65442.4 examples/sec on cuda:0
```
![[所有图片/深度学习图/卷积神经网络/Figure_3.png]]

# 2. 问题
> 将平均汇聚层换为最大汇聚层，激活函数换为ReLU，观察结果.

将`AvgPool2d`更换为`MaxPool2d`，仅将卷积块中的`Sigmoid`函数换为`ReLU`函数.
**Output**
整体损失降低约`40%`，训练准确率提升约`10%`，测试准确率提升不明显.训练速度略微降低.
```python
loss 0.281, train acc 0.894, test acc 0.853
46226.7 examples/sec on cuda:0
```
![[所有图片/深度学习图/卷积神经网络/Figure_4.png|544]]

> 在LeNet的基础上进行适当修改，并在MNIST数据集上训练.观察结果.

MNIST数据集由`70000`张手写数字图像组成，`60000`张用于训练，每张大小为`28x28`.
我将LeNet的卷积块和全连接块中的激活函数都换为`ReLU`，并且卷积块中使用最大汇聚层，学习率降低为`0.1`，其他不变.
**Input**
```python
from torchvision import datasets
from torchvision import transforms as v2

# 其他函数沿用之前的定义.
# Other functions(methods) are defined as before.

def load_data_mnist(batch_size, root):
    trans = v2.ToTensor()
    train_data = datasets.MNIST(root="xxx", train = True, download=False, transform=trans)
    test_data = datasets.MNIST(root="xxx", train = False, download=False, transform=trans)
    train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False)
    return train_iter, test_iter
def show_test_result(net,test_iter,num_show):
    net.eval()
    device = d2l.try_gpu()
    X,y = next(iter(test_iter))
    X = X.to(device)
    y = y.to(device)
    with torch.no_grad():
        y_hat = net(X)
        y_pred_class = torch.argmax(y_hat, dim = 1)
    fig, axes = plt.subplots(1, num_show, figsize=(15, 3))
    for i in range(num_show):
        axes[i].imshow(X[i].cpu().squeeze(), cmap='gray') # 显示图片 (去除通道维度)
        # 标题显示：真实值 / 预测值
        axes[i].set_title(f'True: {y[i].item()}\nPred: {y_pred_class[i].item()}')
        axes[i].axis('off') # 关闭坐标轴
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    net = nn.Sequential(
        nn.Conv2d(1,6,kernel_size = 5,padding = 2),nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),
        nn.Conv2d(6,16,kernel_size=5),nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),
        nn.Flatten(),
        nn.Linear(16*5*5,120),nn.ReLU(),
        nn.Linear(120,84),nn.ReLU(),
        nn.Linear(84,10))
    batch_size = 256
    train_iter, test_iter = load_data_mnist(batch_size, root = "xxx")
    lr = 0.1
    num_epochs = 10
    device = d2l.try_gpu()
    train(net, train_iter, test_iter, num_epochs, lr, device)
    num_show = 9
    show_test_result(net, test_iter, num_show)
```
**Output**
训练的效果非常好，准确率普遍能够达到`98.5%`.
```python
loss 0.035, train acc 0.989, test acc 0.984
65262.7 examples/sec on cuda:0
```
![[所有图片/深度学习图/卷积神经网络/Figure_5.png]]
![[所有图片/深度学习图/卷积神经网络/Figure_6.png]]