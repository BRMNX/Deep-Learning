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
    # 当y_hat不止一个类别并且不止一个样本时才进行
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    # 通过比较，得到一个全是True或False的张量compare
    compare = (y_hat.type(y.dtype) == y)
    return float(compare.type(y.dtype).sum())
def evaluate_accuracy(net,data_iter:data.DataLoader):
    # 设为评估模式，目前还用不到
    if isinstance(net, torch.nn.Module):
        net.eval()
    # 维护两个变量的累加器，记录测试准确率metric[0]和样本总数metric[1]
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(accuracy(net(X),y),y.numel())
            # 向累加器中加入该批次的准确率和数量
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
