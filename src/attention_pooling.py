import torch
from torch import nn
from d2l import torch as d2l
import torch.nn.functional as F
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    """显示矩阵热图"""
    # matrices的形状是(子图行数，子图列数，查询的数目，键的数目)
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    # axes的形状是(子图行数，子图列数)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            # 对每一列，ax是当前的子图轴对象，matrix是具体要显示的一个矩阵
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap, interpolation='bilinear')
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    plt.show()

def f(x):
    return 2*torch.sin(x) + x**0.8
#真实数据
x_range = torch.arange(0,5,0.1)
y_true = f(x_range)
# 数据量
n_train, n_test = 50, len(x_range) # 50, 50
# 带噪声的训练数据(键-值对)
x_train,_ = torch.sort(torch.rand(n_train)*5)
y_train = f(x_train) + torch.normal(0,0.5,(n_train,))

def plot_kernel_regression(y_pred):
    plt.plot(x_range.detach().numpy(), y_true.detach().numpy(), 'b-', label='True', linewidth=2)
    plt.plot(x_range.detach().numpy(), y_pred.detach().numpy(), 'r--', label='Pred', linewidth=2)
    plt.scatter(x_train.detach().numpy(), y_train.detach().numpy(), s = 30, c = 'orange',alpha = 0.5, marker = 'o')
    plt.xlim([0, 5])
    plt.ylim([-1, 5])
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
# x_query 的形状:(n_test, n_train),每一行都是某个固定的query点重复n_train次
x_query = x_range.repeat_interleave(n_train).reshape((-1,n_train))
attention_weights = F.softmax(-(x_query - x_train)**2 / 2, dim = 1)
y_pred = torch.matmul(attention_weights,y_train)
plot_kernel_regression(y_pred)
show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),xlabel='Sorted training inputs', ylabel='Sorted testing inputs')

class NWKernelRegression(nn.Module):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        # 可学习的常数参数w
        self.w = nn.Parameter(torch.rand(1,),requires_grad=True)

    def forward(self, queries, keys, values):
        # keys的形状为(n_train,n_train - 1)，将queries的每个元素复制n_train-1次，再按n_train-1列重排
        # 最终的大小就是(n_train,n_train-1)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1,keys.shape[1]))
        self.attention_weights = F.softmax(-((queries - keys)*self.w)**2 / 2, dim = 1)
        return torch.bmm(self.attention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)
# x_tile的每行都是完整的训练输入，重复x_train行
x_tile = x_train.repeat((n_train,1))
y_tile = y_train.repeat((n_train,1))
# keys和values在x_tile,y_tile中每行移除自身的对角元，形状为(n_train, n_train-1)
# 这样做的目的是为了避免计算注意力时查询x_i遇到键x_i，相减为0，从而softmax后占据了最大的注意力导致过拟合
keys = x_tile[(1-torch.eye(n_train)).type(torch.bool)].reshape((n_train,-1))
values = y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
plt.show()

# keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
keys = x_train.repeat((n_test, 1))
# value的形状:(n_test，n_train)
values = y_train.repeat((n_test, 1))
y_pred = net(x_range, keys, values).unsqueeze(1).detach()
plot_kernel_regression(y_pred)
show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
    