#加载飞桨、NumPy和相关类库
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np
import os
import random

# 1.5.2.1 数据处理
def load_data():
    # 从文件导入数据
    datafile = './work/housing.data'
    data = np.fromfile(datafile, sep=' ', dtype=np.float32)

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原始数据集拆分成训练集和测试集
    # 使用80%的数据做训练，20%的数据做测试，测试集和训练集不能存在交集
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算训练集的最大值和最小值
    maximums, minimums = training_data.max(axis=0), training_data.min(axis=0)
    
    # 记录数据的归一化参数，在预测时对数据进行归一化
    global max_values
    global min_values
    
    max_values = maximums
    min_values = minimums
    
    # 对数据进行归一化处理
    for i in range(feature_num):
        data[:, i] = (data[:, i] - min_values[i]) / (maximums[i] - minimums[i])

    # 划分训练集和测试集
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data
# 验证数据读取的正确性
# training_data, test_data = load_data()
# print(training_data.shape)
# print(training_data[1,:])

# 1.5.2.2 模型设计
class Regressor(paddle.nn.Layer):

    # self代表类的实例自身
    def __init__(self):
        # 初始化父类中的一些参数
        super(Regressor, self).__init__()
        
        # 定义一层全连接层，输入维度是13，输出维度是1
        self.fc = Linear(in_features=13, out_features=1)
    
    # 网络的前向计算
    def forward(self, inputs):
        x = self.fc(inputs)
        return x

# 1.5.2.3 训练配置
# 声明定义好的线性回归模型
model = Regressor()
# 开启模型训练模式，模型的状态设置为train
model.train()
# 使用load_data加载训练集数据和测试集数据
training_data, test_data = load_data()
# 定义优化算法，采用随机梯度下降SGD
# 学习率设置为0.01
opt = paddle.optimizer.SGD(learning_rate=0.005, parameters=model.parameters())

# 1.5.2.4 训练过程

epoch_num = 20   # 设置模型训练轮次
batch_size = 10  # 设置批大小，即一次模型训练使用的样本数量

# 定义模型训练轮次epoch（外层循环）
for epoch_id in range(epoch_num):
    # 在每轮迭代开始之前，对训练集数据进行样本乱序
    np.random.shuffle(training_data)
    # 对训练集数据进行拆分，batch_size设置为10
    mini_batches = [training_data[k:k+batch_size] for k in range(0, len(training_data), batch_size)]
    # 定义模型训练（内层循环）
    for iter_id, mini_batch in enumerate(mini_batches):
        x = np.array(mini_batch[:, :-1]) # 将当前批的房价影响因素的数据转换为np.array格式
        y = np.array(mini_batch[:, -1:]) # 将当前批的标签数据（真实房价）转换为np.array格式
        # 将np.array格式的数据转为张量tensor格式
        house_features = paddle.to_tensor(x, dtype='float32')
        prices = paddle.to_tensor(y, dtype='float32')
        
        # 前向计算
        predicts = model(house_features)

        # 计算损失，损失函数采用平方误差square_error_cost
        loss = F.square_error_cost(predicts, label=prices)
        avg_loss = paddle.mean(loss)
        if iter_id%20==0:
            print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))
        
        # 反向传播，计算每层参数的梯度值
        avg_loss.backward()
        # 更新参数，根据设置好的学习率迭代一步
        opt.step()
        # 清空梯度变量，进行下一轮计算
        opt.clear_grad()

# 保存模型参数，文件名为LR_model.pdparams
paddle.save(model.state_dict(), 'LR_model.pdparams')
print("模型保存成功, 模型参数保存在LR_model.pdparams中")