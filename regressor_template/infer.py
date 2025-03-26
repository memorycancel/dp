import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np
import os
import random

from data import load_data
from data import load_one_example
from design import Regressor

# 将模型参数保存到指定路径中
model = Regressor()

model_dict = paddle.load('model/LR_model.pdparams')
model.load_dict(model_dict)
# 将模型状态修改为.eval
model.eval()

training_data, test_data, max_values, min_values = load_data()
one_data, label = load_one_example(test_data)
# 将数据格式转换为张量 
one_data = paddle.to_tensor(one_data,dtype="float32")
predict = model(one_data)

# 对推理结果进行后处理
predict = predict * (max_values[-1] - min_values[-1]) + min_values[-1]
# 对label数据进行后处理
label = label * (max_values[-1] - min_values[-1]) + min_values[-1]

print("Inference result is {}, the corresponding label is {}".format(predict.numpy(), label))