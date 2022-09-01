import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
# 生成随机数据
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1) # class0_x data
y0 = torch.zeros(100) # class0_y data
x1 = torch.normal(-2*n_data, 1) # class1_x data
y1 = torch.ones(100) # class1_y data
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden) # 隐藏层
        self.out = torch.nn.Linear(n_hidden, n_output) # 输出层
    def forward(self, x):
        x = torch.sigmoid(self.hidden(x)) # 激活函数
        x = self.out(x)
        return x #定义网络、优化器与损失函数
net = Net(n_feature=2, n_hidden=10, n_output=2)
print(net) # net architecture
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss() # the target label is NOT an one-hotted

#可视化
plt.ion()
for t in range(100):
    out = net(x) # 进行前向传播
    loss = loss_func(out, y)# 计算损失函数
    optimizer.zero_grad() # 将梯度清零
    loss.backward() # 进行反向传播
    optimizer.step() # 更新所有参数
    if t % 2 == 0:
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y,s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) /float(target_y.size)
        plt.text(0.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20,
'color': 'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()