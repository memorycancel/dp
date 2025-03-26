#加载飞桨、NumPy和相关类库
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np
import os
import random


from data import load_data

# 1.5.2.2 模型设计
from design import Regressor


# 1.5.2.3 训练配置
# 声明定义好的线性回归模型
model = Regressor()
# 开启模型训练模式，模型的状态设置为train
model.train()
# 使用load_data加载训练集数据和测试集数据
training_data, test_data, max_values, min_values = load_data()
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
paddle.save(model.state_dict(), 'model/LR_model.pdparams')
print("模型保存成功, 模型参数保存在LR_model.pdparams中")