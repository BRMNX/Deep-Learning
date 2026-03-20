#深度学习 #DeepLearning #CNN
# 1. ImageNet数据集
ImageNet数据集有100万个训练样本，1000个类别.每个样本形如`(3,224,224)`.
# 2. AlexNet模型
#AlexNet
AlexNet比LeNet更复杂，它由5个卷积层，2个隐藏全连接层和1个输出全连接层组成.
![[Pasted image 20260316210317.png]]
后面的两个全连接层分别有`4096`个输出，拥有接近`1GB`的模型参数.即使在现代GPU上，直接训练ImageNet数据集也需要数小时甚至数天.因此我们仍然在Fashion-MNIST上训练AlexNet模型.

AlexNet通过暂退法来控制全连接层的模型复杂度.并在训练时增加了大量图像增强数据（如翻转、切割、裁剪、变色），有效减少了过拟合.

**Input**
```python
import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
import matplotlib
from thop import clever_format
from thop import profile
matplotlib.use('TkAgg')

net = nn.Sequential(
    # 这里使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10))

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=224)
num_epochs = 5
lr = 0.01
# 评估模型复杂度
net.eval()
model_name = 'AlexNet'
inputs = torch.randn(1,1,224,224)
flops, params = profile(net, inputs=(inputs,))
flops, params = clever_format([flops, params], "%.3f")
print("%s |%s |%s" % (model_name, flops, params))
# 训练
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
plt.show()
```

**Output**
```python
AlexNet |938.146M |46.765M
loss 0.418, train acc 0.848, test acc 0.854
1820.0 examples/sec on cuda:0
```
![[所有图片/深度学习图/现代卷积神经网络/Figure_1.png]]

# 3. VGG块
VGG块由`(3,3)`卷积核、填充为`1`的卷积层，ReLU激活函数以及一个`(2,2)`步幅为2的最大汇聚层组成.
```python
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)
```

VGG网络的前面部分由多个VGG块组成，最后三层是与AlexNet一样的全连接层.其中有超参数`conv_arch`，用来指定每个VGG块中的卷积层数量和输出通道数.
![[Pasted image 20260318092843.png]]
```python
def vgg(conv_arch):
    conv_blocks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blocks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(*conv_blocks, nn.Flatten(), 
                         nn.Linear(out_channels*7*7,4096),nn.ReLU(),nn.Dropout(0.5),
                         nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),nn.Linear(4096,10))
```

原始VGG网络有5个卷积块，其中前两个块各有一个卷积层，后三个块各包含两个卷积层。 第一个模块有64个输出通道，每个后续模块将输出通道数量翻倍，直到该数字达到512.由于该网络使用8个卷积层和3个全连接层，因此它通常被称为VGG-11.
```python
def vgg(conv_arch):
    conv_blocks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blocks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(*conv_blocks, nn.Flatten(), 
                         nn.Linear(out_channels*7*7,4096),nn.ReLU(),nn.Dropout(0.5),
                         nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),nn.Linear(4096,10))
                         
if __name__ == "__main__":
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)) #(num_convs, out_channels)
    net = vgg(conv_arch)
```

由于VGG-11比AlexNet计算量更大，所以我们把上述`conv_arch`的每一个`out_channels`都除以4.并将`Fashion-MNIST`数据集`resize = 224`进行训练.

**Input**
```python
if __name__ == "__main__":
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    ratio = 4
    small_conv_arch = [(pair[0], pair[1]//ratio) for pair in conv_arch]
    net = vgg(small_conv_arch)
    batch_size = 128
    lr = 0.05
    num_epochs = 5
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=224)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    plt.show()
```

**Output**
```python
loss 0.265, train acc 0.903, test acc 0.901
985.6 examples/sec on cuda:0
```
![[所有图片/深度学习图/现代卷积神经网络/Figure_2.png]]

> 与AlexNet相比，VGG的计算要慢得多，而且需要更多显存.分析原因?

**关于计算量大小**
- 在卷积块中，VGG采用小卷积核堆积的策略，将`(3,3)`卷积核堆叠多层.为了达到同样的感受野，VGG的深度快速增加，其输出通道数也快速增大到`512`.但在此基础上，VGG又保持了高分辨率的运算（在每个卷积核都填充了`1`），这导致浮点数运算量爆炸性增长.
**关于显存**
- 在传入全连接层将向量展平时，VGG有`512*7*7 = 25088`个分量，而AlexNet只有`6400`个，仅为VGG的`1/4`，这是VGG占用显存高的**主要原因**.
- 由于VGG在很深的地方才进行下采样（Pooling），这意味着在网络的前半部分和中间部分，它必须保存大量**高分辨率、多通道**的特征图用于反向传播.相比之下，AlexNet的下降速度快得多，深层的特征图尺寸很小，保存激活值所需的显存就少得多.

# 4. NiN（网络中的网络）模型
**NiN(Network in Network)** 的想法是在每个像素应用一个全连接层，为了做到这一点，需要忽略周围像素，因此使用多通道的`(1,1)`卷积核.

NiN块以一个普通卷积层开始，后面是两个`(1,1)`的卷积层.**这两个卷积层充当带有ReLU激活函数的逐像素全连接层**. 第一层的卷积窗口形状通常手动设置，随后的卷积窗口形状固定为`(1,1)`.NiN模型与其他模型的显著区别是，NiN取消了全连接层，并用**自适应平均汇聚层**替代它，并令输出通道数等于标签类别数.移除全连接层可减少过拟合，同时显著减少NiN的参数.
![[Pasted image 20260318094833.png]]

**Input**
```python
import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成二维的输出，其形状为(batch_size, 10, 1, 1)
    nn.Flatten())

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
plt.show()
```
其中`nn.AdaptiveAvgPool2d((output_size))`方法（自适应平均汇聚）将`(batch_size, in_channels, H, W)`大小的特征图转化为`(batch_size, in_channels, (output_size))`大小.这里指定`(output_size) = (1,1)`就是直接计算了每个类别特征图的平均值作为该类的得分.

**Output**
```python
loss 0.378, train acc 0.860, test acc 0.856
1379.0 examples/sec on cuda:0
```
![[所有图片/深度学习图/现代卷积神经网络/Figure_3.png]]

# 5. GoogLeNet并行连结网络
## 5.1 Inception块
在GoogLeNet中，基本卷积块被称为Inception块，它由四条并行路径组成.
![[Pasted image 20260319110536.png]]
超参数通常是每层的输出通道数.注意第四条通道的最大汇聚，附加的是填充1以及步幅1.
```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)
```

## 5.2 GoogLeNet模型
如图所示，GoogLeNet使用9个Inception块和一些其他卷积层的堆叠.它的前面部分类似于AlexNet和LeNet，最后使用**全局平均汇聚层**是参照了VGG.在Inception块之间的最大汇聚层是为了降低维度.
![[Pasted image 20260319111205.png]]
下面分不同模块来实现GoogLeNet.
```python
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                   
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                   
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                   
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                   
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
```
GoogLeNet的计算复杂， 且不易修改通道数，因此将Fashion-MNIST数据集中的图片大小调整为`(96,96)`.
```python
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```