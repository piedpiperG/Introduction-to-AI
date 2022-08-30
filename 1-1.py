"""
本实验需要提前了解的知识：
1、线性回归
2、偏导数
"""
from sklearn.datasets import make_regression#导入 make_regression()函数，用#来生成回归模型数据
import matplotlib.pyplot as plt#导入 matplotlib.pyplot，并且重命名为 plt
import numpy as np#导入 numpy 库，并且重命名为 np
"""
需要提前了解的知识：
1、线性回归
2、偏导数
"""
"""
1、函数 make_regression()：用来生成回归模型数据。
2、参数说明：
 n_samples：样本数
 n_features：特征数
 noise：噪音
 bias：偏差
3、X : array of shape [n_samples, n_features]
 y : array of shape [n_samples] or [n_samples, n_targets]
4、下面的语句的作用为:生成一组数据集{(x1,y1),(x2,y2),……,(x100,y100)}，后面我
们将学习一个线性模型来尽可能的拟合此数据集。
"""
X, y= make_regression(n_samples=100, n_features=1, noise=0.4, bias=50)
"""
1、定义一个名为 plotLine()的函数，用来画出生成数据集的散点图和拟合线性模型
(y=k*x+b)
2、参数说明：
 theta0:即 y=k*x+b 中的参数 b
 theta1:即 y=k*x+b 中的参数 k
 X:数据集的横坐标（列表类型）
 y:数据集的纵坐标（列表类型）
3、np.linspace(start, stop, num)函数：用来返回 num 个等间距的样本，在区间[start, stop]中。
4、plt.plot(x,y,color,label)：可视化函数
 参数说明：x:x 轴上的数值；y:y 轴上的数值;color:用来设置线条的颜色，color='r'表示红色(b 表示蓝色)；label 用于指定标签
5、plt.scatter(x,y)：用来画散点图。
 参数说明:x:x 轴上的数值；y:y 轴上的数值。
6、plt.axis(）函数用来指定坐标轴的范围。
 参数需要以列表的形式给出。
7、plt.show()：将图像显示出。
"""
def plotLine(theta0, theta1, X, y):
    max_x = np.max(X) + 100 #np.max(X)用来取出 X 中的最大值
    min_x = np.min(X) - 100 #np.min(X)用来取出 X 中的最小值
    xplot = np.linspace(min_x, max_x, 1000) #在区间[min_x,max_x]中返回 1000个等间隔的样本
    yplot = theta0 + theta1 * xplot #将 x 带入线性方程 y=k*x+b 中求得 y
    #xxplot,yyplot = derivatives(theta0, theta1, X, y)

    print("目前的参数 b=",theta0) #打印参数 theta0
    print("目前的参数 k=",theta1) #打印参数 theta1
    plt.plot(xplot, yplot, color='g', label='Regression Line') #画出线性模型，参数依次表示：横坐标，纵坐标，颜色，标签
    #plt.plot(xxplot, yyplot, color='g', label='Regression Line') #画出线性模型，参数依次表示：横坐标，纵坐标，颜色，标签
    plt.scatter(X,y) #画散点图，参数依次表示横坐标、纵坐标
    plt.axis([-10, 10, 0, 200]) #设置横坐标范围为【-10，10】，纵轴范围为【0，200】
    plt.show() #显示可视化图像
"""
1、定义一个名为 hypothesis()的函数,根据给定的 x 值预测 y 的值，计算公式为:y=theta0 
+ (theta1*x)
"""
def hypothesis(theta0, theta1, x):
    return theta0 + (theta1*x)
"""
1、定义一个计算损失值的函数，采用最小二乘法来计算损失。
2、zip(x,y)函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，
然后返回由这些元组组成的列表。
 譬如：x={x1,x2,x3};y={y1,y2,y3};则 zip(x,y)=[(x1,y1),(x2,y2),(x3,y3)]
3、y**2:表示 y 的平方。
"""
def cost(theta0, theta1, X, y): #计算损失
    costValue = 0
    for (xi, yi) in zip(X, y): #使用 zip()函数，包为元组的列表
        costValue += 0.5 * ((hypothesis(theta0, theta1, xi) - yi)**2) #使用最小二乘法来计算损失
    return costValue #返回损失值
"""
1、定义名为 derivatives()的函数，用来计算参数的梯度。
2、len()函数：用来返回对象（字符、列表、元组等）长度或项目个数。其参数可以是字符、
列表、元组等。
"""
def derivatives(theta0, theta1, X, y): #derivative:导数
    dtheta0 = 0 #dtheta0：参数 theta0 的梯度，初始化为 0
    dtheta1 = 0 #dtheta0：参数 theta0 的梯度，初始化为 0
    for (xi, yi) in zip(X, y): #使用 zip()函数依次取出(xi,yi)
        dtheta0 += hypothesis(theta0, theta1, xi) - yi #计算公式为：损失函数对参数 dtheta0 求偏导。
        dtheta1 += (hypothesis(theta0, theta1, xi) - yi)*xi #计算公式为：损失函数对参数 dtheta1 求偏导。
    dtheta0 /= len(X) #求平均梯度，len(X)函数用来计算 X 中的样本数
    dtheta1 /= len(X) #求平均梯度
    return dtheta0, dtheta1
"""
1、定义一个名为 updateParameters()的函数，用来对参数进行更新。
 参数说明：
 theta0 和 theta1 为待更新参数。
 X、 y 分别表示横轴和纵轴的数值。
 alpha：学习率。
2、参数的更新：
 对于参数 w，其更新方式为：w=w-学习率*梯度值。其中学习率是一个超参数。
"""
def updateParameters(theta0, theta1, X, y, alpha): #参数的更新，alpha 表示学习率
    dtheta0, dtheta1 = derivatives(theta0, theta1, X, y) #dtheta0, dtheta1分别表示参数 theta0，theta1 的梯度值。
    theta0 = theta0 - (alpha * dtheta0) #依据参数更新方式更新参数 theta0
    theta1 = theta1 - (alpha * dtheta1) #依据参数更新方式更新参数 theta1
    return theta0, theta1 #返回更新好的参数

"""
1、定义一个名为 LinearRegression()的线性回归函数。
 参数说明：
 X：表示给定数据集的横坐标。
 y：表示给定数据集的纵坐标。
2、np.random.rand()函数：用来返回一个或一组服从“0~1”均匀分布的随机样本值。随机
样本取值范围是[0,1)，
 不包括 1。当不给定参数时，返回的是一个[0，1)区间内的随机数。）
"""
def LinearRegression(X, y):
    theta0 = np.random.rand() #给 theta0 赋一个随机初始值。
    theta1 = np.random.rand() #给 theta1 赋一个随机初始值。
    for i in range(0, 1000): #进行 1000 次参数的更新，每隔 100 次跟新打印一次图片
        if i % 100 == 0: #只有当 i 整除 100 时才进行一次图片打印
            plotLine(theta0, theta1, X, y)
            print(cost(theta0, theta1, X, y))
        theta0, theta1 = updateParameters(theta0, theta1, X, y, 0.005) #
    #调用参数更新函数来对参数进行更新，其中学习率指定为：0.005.


LinearRegression(X, y) #调用线性回归函数。


#plotLine(1, 1, X, y)
#a = hypothesis(1, 1, X)
#print(a)