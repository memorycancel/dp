import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np
import os
import random

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

def load_one_example():
    # 从测试集中随机选择一条作为推理数据
    idx = np.random.randint(0, test_data.shape[0])
    idx = -10
    one_data, label = test_data[idx, :-1], test_data[idx, -1]
    # 将数据格式修改为[1,13]
    one_data =  one_data.reshape([1,-1])

    return one_data, label

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

# 将模型参数保存到指定路径中
model = Regressor()

model_dict = paddle.load('LR_model.pdparams')
model.load_dict(model_dict)
# 将模型状态修改为.eval
model.eval()

training_data, test_data = load_data()
one_data, label = load_one_example()
# 将数据格式转换为张量 
one_data = paddle.to_tensor(one_data,dtype="float32")
predict = model(one_data)

# 对推理结果进行后处理
predict = predict * (max_values[-1] - min_values[-1]) + min_values[-1]
# 对label数据进行后处理
label = label * (max_values[-1] - min_values[-1]) + min_values[-1]

print("Inference result is {}, the corresponding label is {}".format(predict.numpy(), label))