# 导入PyTorch的神经网络模块
import torch.nn as nn

# 定义深度Q网络类，继承自PyTorch的nn.Module
class DeepQNetwork(nn.Module):
    def __init__(self):
        # 调用父类的初始化方法
        super(DeepQNetwork, self).__init__()

        # 第一层：全连接层（输入4个特征，输出64个神经元）+ ReLU激活函数
        # 输入的4个特征是：消除的行数、空洞数、凹凸度、总高度
        # inplace=True表示直接修改输入，节省内存
        self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        # 第二层：全连接层（输入64个神经元，输出64个神经元）+ ReLU激活函数
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        # 第三层（输出层）：全连接层（输入64个神经元，输出1个值）
        # 输出值表示该状态-动作对的Q值（预期累积奖励）
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

        # 调用权重初始化方法
        self._create_weights()

    def _create_weights(self):
        # 遍历网络中的所有模块
        for m in self.modules():
            # 如果模块是全连接层（Linear层）
            if isinstance(m, nn.Linear):
                # 使用Xavier均匀分布初始化权重
                # 作用是在训练开始前，给网络中的全连阶层赋予合理的初始数值，避免权重过大或过小导致梯度消失或爆炸
                # Xavier初始化有助于保持信号在网络中传播时的方差稳定，让每一层的输出方差与输入方差相同，从而促进更快的收敛
                nn.init.xavier_uniform_(m.weight)
                # 将偏置初始化为0
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 前向传播：通过第一层（全连接 + ReLU）
        x = self.conv1(x)
        # 通过第二层（全连接 + ReLU）
        x = self.conv2(x)
        # 通过第三层（输出层，只有全连接层，无激活函数）
        x = self.conv3(x)

        # 返回Q值预测
        return x
