#深度学习 #DeepLearning #CNN

# 1. 卷积运算
## 1.1 卷积核
用来进行互相关运算的张量称为**卷积核(kernel)**.在卷积层中，权重就是卷积核，偏置仍然是一个常数，它们都是学习的对象.
假定输入张量的大小是$n_{h}\times n_{w}$，卷积核的大小是$k_{h}\times k_{w}$，则输出的大小是$(n_{h}-k_{h}+1)\times(n_{w}-k_{w}+1)$.
用代码实现这样的互相关运算：
**Input**
```python
mport torch
from torch import nn
from d2l import torch as d2l
# X是输入，K是卷积核
def corr2d(X, K):
    Y = torch.zeros(X.shape[0]-K.shape[0]+1, X.shape[1]-K.shape[1]+1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+K.shape[0],j:j+K.shape[1]]*K).sum()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0,1], [2,3]])
Y = corr2d(X, K)
print(Y)
```
这里的算例使用
![[Pasted image 20260313174826.png]]
**Output**
```python
tensor([[19., 25.],
        [37., 43.]])
```
借助特别的卷积核，可以实现边缘检测.
**Input**
注意输入时，**卷积核至少是二维张量**，即`K = [[a,b]]`（**两个中括号**），因为我们需要利用`K.shape[0]`和`K.shape[1]`来计算输出的大小，一维的`K.shape`传入会报错.
```python
X = torch.ones((6, 8))
X[:, 2:6] = 0
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
print(Y)
```
这里`X`展开形如
```python
X = tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.]])
```
**Output**
实现了垂直方向上的边缘检测.
```python
tensor([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.]])
```
## 1.2 问题
>构建一个具有对角线边缘的$X$.
>(1)如果将$X$应用于$K = [-1,1]$，会发生什么？
>(2)将$X^{\mathrm{T}}$应用于$K$.
>(3)将$X$应用于$K^{\mathrm{T}}$.

**Input**
```python
X = torch.eye(8)
K = torch.tensor([[1.0,-1.0]])
print(f'Diagonal matrix X = \n{X}')
print(f'Apply corr2d(X, K) = \n{corr2d(X, K)}')
print(f'Apply corr2d(X.t(), K) = \n{corr2d(X.t(), K)}')
print(f'Apply corr2d(X, K.t()) = \n{corr2d(X, K.t())}')
```
**Output**
```python
Diagonal matrix X = 
tensor([[1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1.]])
Apply corr2d(X, K) =
tensor([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [-1.,  1.,  0.,  0.,  0.,  0.,  0.],
        [ 0., -1.,  1.,  0.,  0.,  0.,  0.],
        [ 0.,  0., -1.,  1.,  0.,  0.,  0.],
        [ 0.,  0.,  0., -1.,  1.,  0.,  0.],
        [ 0.,  0.,  0.,  0., -1.,  1.,  0.],
        [ 0.,  0.,  0.,  0.,  0., -1.,  1.],
        [ 0.,  0.,  0.,  0.,  0.,  0., -1.]])
Apply corr2d(X.t(), K) =
tensor([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [-1.,  1.,  0.,  0.,  0.,  0.,  0.],
        [ 0., -1.,  1.,  0.,  0.,  0.,  0.],
        [ 0.,  0., -1.,  1.,  0.,  0.,  0.],
        [ 0.,  0.,  0., -1.,  1.,  0.,  0.],
        [ 0.,  0.,  0.,  0., -1.,  1.,  0.],
        [ 0.,  0.,  0.,  0.,  0., -1.,  1.],
        [ 0.,  0.,  0.,  0.,  0.,  0., -1.]])
Apply corr2d(X, K.t()) =
tensor([[ 1., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  1., -1.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  1., -1.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  1., -1.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  1., -1.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  1., -1.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  1., -1.]])
```
>尝试对下面这张骨骼图像(灰度图)作边缘检测.

![[fractured_spine.png|603]]
利用卷积核$$K = \begin{bmatrix}1 & 0 & -1 \\1 & 0 & -1\\1 & 0 & -1\end{bmatrix}$$进行垂直边缘检测.再用其转置进行水平边缘检测.最后将两个结果相加（水平垂直各占1/2）得到混合边缘检测.
**Input**
```python
import torch
from torch import nn
from d2l import torch as d2l
import numpy as np
import matplotlib.pyplot as plt

img_path = '...'
img = plt.imread(img_path)
img_tensor = torch.tensor(img)
h, w = img_tensor.shape
kernel = torch.tensor([[1,0,-1],[1,0,-1],[1,0,-1]])
Y1 = corr2d(img_tensor, kernel)
Y2 = corr2d(img_tensor, kernel.t())
Y3 = 0.5 * Y1 + 0.5 * Y2
fig, axes = plt.subplots(1, 3, figsize=(15, 5)) 
axes[0].imshow(Y1, cmap='gray')
axes[0].set_title('Vertical Edge Detection')
axes[1].imshow(Y2, cmap='gray')
axes[1].set_title('Horizontal Edge Detection')
axes[2].imshow(Y3, cmap='gray')
axes[2].set_title('Mixed Edge Detection')
plt.tight_layout()
plt.show()
```
**Output**
![[所有图片/深度学习图/卷积神经网络/Figure_1.png]]
# 2. 二维卷积层实现
下面实现二维卷积层，需要借助`nn.Conv2d`作为`net`.
**Input**
```python
#传入的样本必须是四维的，(批量大小、通道、高度、宽度)
X = X.reshape((1, 1, 6, 8))
K = K.reshape((1, 1, 1, 2))
# in_channels是输入图像的通道数
# out_channels是经过卷积后预期的输出图像通道数
# kernel_size只填一个数时生成方阵，填两个数则按规定生成.
net = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = (1, 2), bias = False)
for i in range(10):
    Y_hat = net(X)
    l = (Y_hat - Y) ** 2
    net.zero_grad()
    l.sum().backward()
    net.weight.data -= lr * net.weight.grad
    if (i+1)%2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')
print(f'学习后的卷积核K = ', net.weight.data.reshape((1, 2)))
```
**Output**
```python
epoch 2, loss 9.229
epoch 4, loss 2.337
epoch 6, loss 0.715
epoch 8, loss 0.252
epoch 10, loss 0.097
学习后的卷积核K =  tensor([[ 0.9624, -1.0132]])
```
学习后的卷积核已经与原始`K = [1,-1]`非常接近.
# 3. 特征映射和感受野
- **特征映射**：在含有多个卷积层的网络中，隐藏层全体记作$\mathbf{H}$，称为**隐藏表示**.隐藏表示可以看成是一系列具有多个二维张量的通道，这些通道被称为**特征映射(feature maps)**.每个通道都为后续层提供一组学习特征.（比如说，一个三通道的图像进行卷积，可以看作是在每个通道R/G/B上分别做卷积，反映三种不同的特征，最后再合并）
- **感受野**：对于某一层的任一元素$x$，其**感受野(receptive field)** 是指在前向传播期间可能影响$x$计算的所有元素.

# 4. 填充与步幅
## 4.1 填充
在经过多个卷积层的累积后，输出的尺寸可能会过度缩小，以至于丢失边界的信息.使用**填充(padding)** 来解决这个问题.
>**填充**就是向输入添加行和列，元素均为0.对于输入$n_{h}\times n_{w}$和卷积核$k_{h}\times k_{w}$，如果在输入中填充$p_{h}$行和$p_{w}$列，那么输出的大小为$$(n_{h}+p_{h}-k_{h}+1)\times(n_{w}+p_{w}-k_{w}+1)$$

很多时候，我们希望*输出与输入大小一致(Input and output have the same size)*，因此可取$$(p_{h},p_{w}) = (k_{h}-1,k_{w}-1)$$为此，我们通常**取卷积核的大小$k_{h},k_{w}$都是奇数**，这样填充的行数与列数$p_{h},p_{w}$都是偶数，便于在输入的上下两侧和左右两侧等量地填充.使用奇数大小卷积核的好处还在于，输出$Y[i,j]$与输入$X[i,j]$的位置一一对应.
**Input**
```python
# 此方法便于初始化参数，增减输入/输出到合适的维度
def comp_conv2d(net, X):
    X = X.reshape((1,1) + X.shape)
    Y = net(X)
    # Y形如(1, 1, h, w)，其中h和w分别是输出的高和宽
    # Y.shape[2:]只保留h和w，舍去Y的batch_size和out_channels维度
    # 但是只有在总元素个数相同时(也就是被舍去的轴维度都是1)才能这样reshape
    return Y.reshape(Y.shape[2:])

# 这里padding=1让每边都填充了一行或一列，一共填充2行2列
net = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, padding=1)
X = torch.rand(size=(8,8))
# 查询输出的大小
print(comp_conv2d(net, X).shape)
```
**Output**
输出与输入的大小一致.
```python
torch.Size([8, 8])
```
## 4.2 步幅
有时候为了高效计算和减少采样次数，卷积窗口每次滑动多个元素，将每次滑动元素的数量称为**步幅(stride)**.下图是水平步幅为2，垂直步幅为3的例子（从左下角开始）.
![[Pasted image 20260314133431.png]]
通常，若步幅为$s_{h}\times s_{w}$，则输出的大小为$$\lfloor(n_h-k_h+p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+p_w+s_w)/s_w\rfloor.$$
继续使用前面代码中定义的`X`，`Y`和卷积核大小`3`，但是将卷积核的步幅设置为2.
**Input**
```python
net = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, padding=1, stride=2)
print(comp_conv2d(net, X).shape)
```
**Output**
输出的大小满足上面的公式.
```python
torch.Size([4, 4])
```
## 4.3 小结
- 填充可以增加输出的高度和宽度，通常使输入输出大小一致.
- 步幅可以减少输出的高度和宽度.
- 在实践中，我们很少使用长宽不一致的填充和步幅.

# 5. 多输入与多输出
## 5.1 多通道输入
当输入的图像有$c$个（$c\geqslant1$）通道时，相应的卷积核也应该有$c\times k_{h}\times k_{w}$的大小.各个通道的计算是**相互独立**的，在每个通道上计算二维卷积，最后将每个通道的输出相加得到输出.
![[Pasted image 20260314134820.png]]
下面的代码实现上图的计算.
**Input**
```python
def corr2d_multi_in(X, K):
    # zip()函数把X的每个通道和K的对应通道一一配对，然后for循环遍历每个通道
    # 每次返回的x都是该通道上的二维张量，k是该通道上的二维卷积核
    # zip()函数默认以第0个维度的长度为准进行配对，也就是通道数.如果X和K的通道数不一样，则配对完最短的为止
    return sum(d2l.corr2d(x,k) for x,k in zip(X,K))

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
print(corr2d_multi_in(X, K))
```
**Output**
```python
tensor([[ 56.,  72.],
        [104., 120.]])
```
## 5.2 多通道输出
用$c_{i}$和$c_{0}$分别表示输入和输出的通道数，则卷积核形如$c_{i}\times c_{0}\times k_{h}\times k_{w}$.
![[26884e37bd1297676900eac49f1e5eeb.jpg]]
`torch.stack((a,b,c),0)`可以将三个大小完全相同的张量`a,b,c`放到一个新的维度中，各自占据一个位置.比如`[a,b,c]->[[a],[b],[c]]`.
**Input**
```python
def corr2d_multi_in_out(X,K):
    return torch.stack([corr2d_multi_in(X,k) for k in K],0)

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
# 手动修改K使得输出具有三个通道
K = torch.stack((K,K+1,K+2),0)
print(f'2通道输入3通道输出的corr2d(X,K) = \n{corr2d_multi_in_out(X,K)}')
```
**Output**
```python
2通道输入3通道输出的corr2d(X,K) =
tensor([[[ 56.,  72.],
         [104., 120.]],

        [[ 76., 100.],
         [148., 172.]],

        [[ 96., 128.],
         [192., 224.]]])
```
第一个通道的输出就是先前单通道的输出.
## 5.3 1x1卷积层
1x1卷积层就是指卷积核的长宽都为1.
![[Pasted image 20260314144822.png]]
对每个像素而言，1x1卷积层相当于全连接层.
下面的函数相当于先前实现的互相关函数`corr2d_multi_in_out`.
```python
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))
X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
print(f'1x1卷积核的输出 = \n{Y1}')
print(f'一般卷积核的输出 = \n{Y2}')
print(f'是否认为它们相等:{abs((Y1- Y2).sum()) < 1e-6}')
```
**Output**
```python
1x1卷积核的输出 = 
tensor([[[ 0.3311,  1.0936, -0.9430],
         [-2.2279,  0.1160, -3.0578],
         [ 2.8219, -0.1265, -0.5526]],

        [[ 0.9893, -1.9366,  0.6550],
         [ 1.9815,  0.3860,  3.1253],
         [-2.7913,  0.5407,  0.3937]]])
一般卷积核的输出 =
tensor([[[ 0.3311,  1.0936, -0.9430],
         [-2.2279,  0.1160, -3.0578],
         [ 2.8219, -0.1265, -0.5526]],

        [[ 0.9893, -1.9366,  0.6550],
         [ 1.9815,  0.3860,  3.1253],
         [-2.7913,  0.5407,  0.3937]]])
是否认为它们相等:True
```
# 6. 汇聚层
> **汇聚层(pooling)** （或称池化层）本质上就是一个卷积核，只不过这个核中不包含参数，而是含一个算子.每当此窗口滑过输入，就返回该算子的计算结果.通常，汇聚窗口的算子取$\max$或平均值，分别称为**最大汇聚层(maximum pooling)** 和 **平均汇聚层(average pooling)**.

易见，**最大汇聚层可以保持输入平移不变性**.即当输入的像素发生小范围的偏移，汇聚层的结果不会改变.
![[Pasted image 20260314150832.png]]
**Input**
```python
def pool2d(X,pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0]-p_h+1,X.shape[1]-p_w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i,j] = X[i:i+p_h,j:j+p_w].max()
            elif mode == 'avg':
                Y[i,j] = X[i:i+p_h,j:j+p_w].mean()
    return Y

X = torch.tensor([[0,1,2],[3,4,5],[6,7,8]])
print(f'2x2最大汇聚层的输出 = \n{pool2d(X,(2,2))}')
```
**Output**
```python
2x2最大汇聚层的输出 = 
tensor([[4., 5.],
        [7., 8.]])
```

与卷积层一样，汇聚层也有填充和步幅.可以通过`nn.MaxPool2d(pool_size)`调用深度学习框架中的二维最大汇聚层.
**Input**
```python
X = torch.arange(16, dtype=torch.float32).reshape((1,1,4,4))
print(f'X = \n{X}')

pool = nn.MaxPool2d(3)
print(f'3x3最大汇聚层的输出 = \n{pool(X)}')

pool = nn.MaxPool2d(3, padding=1, stride=2)
print(f'3x3最大汇聚层的输出(步幅2，填充1) = \n{pool(X)}')
```
1. ❌PyTorch的底层实现规定**所有神经网络只接受浮点数输入**.`X = torch.arange(16)`类型是`torch.int64`(等价地`Long`），如果不使用类型转换`dtype = torch.float32`，会报错`RuntimeError: "max_pool2d" not implemented for 'Long'`
2. ✅汇聚层的步幅默认为`pool_size`，也就是PyTorch默认让每个汇聚窗口互不相交.

**Output**
```python
X = 
tensor([[[[ 0.,  1.,  2.,  3.],
          [ 4.,  5.,  6.,  7.],
          [ 8.,  9., 10., 11.],
          [12., 13., 14., 15.]]]])
3x3最大汇聚层的输出 =
tensor([[[[10.]]]])
3x3最大汇聚层的输出(步幅2，填充1) =
tensor([[[[ 5.,  7.],
          [13., 15.]]]])
```

对于多通道输入，汇聚层会将每个通道的结果分别输出，而不是像卷积层那样加起来.因此，**汇聚层的输入通道数=输出通道数**.

**Input**
```python
# 在输入通道上添加X+1，X+2，形成3通道输入
X = torch.cat((X, X+1, X+2),1)
pool = nn.MaxPool2d(3, padding=1, stride=2)
print(f'3x3最大汇聚层3通道输入的输出(步幅2，填充1) = \n{pool(X)}')
```
**Output**
```python
3x3最大汇聚层3通道输入的输出(步幅2，填充1) = 
tensor([[[[ 5.,  7.],
          [13., 15.]],

         [[ 6.,  8.],
          [14., 16.]],

         [[ 7.,  9.],
          [15., 17.]]]])
```