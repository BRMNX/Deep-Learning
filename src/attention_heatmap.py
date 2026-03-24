import torch
from d2l import torch as d2l
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

X = torch.randn((3,3,10,10))
ylabel = 'Querys'
xlabel = 'Keys'
titles = ['a','b','c','d','e','f','g','h','i','j']
show_heatmaps(X, xlabel, ylabel,titles)
