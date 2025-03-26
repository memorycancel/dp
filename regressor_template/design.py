import paddle
from paddle.nn import Linear
import paddle.nn.functional as F

class Regressor(paddle.nn.Layer):
    def __init__(self):
        super(Regressor, self).__init__()
        
        # 定义一层全连接层，输入维度是13，输出维度是1
        self.fc = Linear(in_features=3, out_features=1)
    
    # 网络的前向计算
    def forward(self, inputs):
        x = self.fc(inputs)
        return x