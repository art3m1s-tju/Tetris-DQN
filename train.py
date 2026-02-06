# 导入argparse库，用于解析命令行参数
import argparse
# 导入os库，用于操作系统相关功能（如文件路径操作）
import os
# 导入shutil库，用于高级文件和文件夹操作（如删除目录树）
import shutil
# 从random模块导入随机数生成函数
# random: 生成[0,1)之间的随机浮点数
# randint: 生成指定范围内的随机整数
# sample: 从序列中随机抽取指定数量的元素
from random import random, randint, sample

# 导入numpy库，用于高效的数值计算和数组操作
import numpy as np
# 导入PyTorch主库，用于深度学习
import torch
# 导入PyTorch的神经网络模块
import torch.nn as nn
# 导入TensorBoardX，用于可视化训练过程
from tensorboardX import SummaryWriter

# 从src.deep_q_network模块导入深度Q网络模型
from src.deep_q_network import DeepQNetwork
# 从src.tetris模块导入俄罗斯方块游戏环境
from src.tetris import Tetris
# 从collections模块导入双端队列，用于实现经验回放缓冲区
from collections import deque


# 定义函数：获取并解析命令行参数
def get_args():
    # 创建参数解析器，设置程序描述
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    # 添加游戏板宽度参数（默认10列）
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    # 添加游戏板高度参数（默认20行）
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    # 添加方块像素大小参数（默认30像素）
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    # 添加批量大小参数（默认512）- 每次训练使用的经验样本数量
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    # 添加学习率参数（默认0.001）- 控制模型参数更新的步长
    parser.add_argument("--lr", type=float, default=1e-3)
    # 添加折扣因子gamma（默认0.99）- 用于计算未来奖励的折扣
    parser.add_argument("--gamma", type=float, default=0.99)
    # 添加初始探索率epsilon（默认1.0）- 开始时完全随机探索
    parser.add_argument("--initial_epsilon", type=float, default=1)
    # 添加最终探索率epsilon（默认0.001）- 训练后期主要利用已学知识
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    # 添加探索率衰减轮数（默认2000轮）- epsilon从初始值线性衰减到最终值的轮数
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    # 添加总训练轮数（默认3000轮）
    parser.add_argument("--num_epochs", type=int, default=3000)
    # 添加模型保存间隔（默认每1000轮保存一次）
    parser.add_argument("--save_interval", type=int, default=1000)
    # 添加经验回放池大小（默认30000）- 存储历史经验用于训练
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Number of epoches between testing phases")
    # 添加TensorBoard日志保存路径（默认"tensorboard"目录）
    parser.add_argument("--log_path", type=str, default="tensorboard")
    # 添加训练模型保存路径（默认"trained_models"目录）
    parser.add_argument("--saved_path", type=str, default="trained_models")

    # 解析命令行参数
    args = parser.parse_args()
    # 返回解析后的参数对象
    return args


# 定义训练函数：实现深度Q学习算法训练AI玩俄罗斯方块
def train(opt):
    # 设置随机种子以确保结果可复现
    if torch.cuda.is_available():
        # 如果有GPU可用，设置CUDA随机种子
        torch.cuda.manual_seed(123)
    else:
        # 如果只有CPU，设置CPU随机种子
        torch.manual_seed(123)

    # 准备TensorBoard日志目录
    if os.path.isdir(opt.log_path):
        # 如果日志目录已存在，删除旧的日志目录
        shutil.rmtree(opt.log_path)
    # 创建新的日志目录
    os.makedirs(opt.log_path)
    # 初始化TensorBoard写入器，用于记录训练过程
    writer = SummaryWriter(opt.log_path)

    # 创建俄罗斯方块游戏环境，使用命令行参数指定的尺寸
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    # 初始化深度Q网络模型
    model = DeepQNetwork()
    # 创建Adam优化器，用于更新模型参数
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    # 定义均方误差损失函数，用于计算预测Q值与目标Q值之间的差异
    criterion = nn.MSELoss()

    # 重置游戏环境，获取初始状态
    state = env.reset()
    # 如果GPU可用，将模型和状态移到GPU上加速计算
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()

    # 创建经验回放缓冲区（双端队列），用于存储历史经验
    # maxlen参数确保缓冲区大小不超过指定值，旧经验会被自动删除
    replay_memory = deque(maxlen=opt.replay_memory_size)
    # 初始化训练轮数计数器
    epoch = 0

    # 主训练循环：持续训练直到达到指定轮数
    while epoch < opt.num_epochs:
        # 获取当前状态下所有可能的下一步动作及其对应的状态
        next_steps = env.get_next_states()

        # 计算当前的探索率epsilon（epsilon-greedy策略）
        # epsilon从initial_epsilon线性衰减到final_epsilon
        # 探索率决定了随机选择动作的概率
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        # 生成一个[0,1)之间的随机数，用于决定是探索还是利用
        u = random()
        # 如果随机数小于等于epsilon，则进行探索（随机选择动作）
        random_action = u <= epsilon
        # 将所有可能的动作和状态分离成两个列表
        # 把所有下一状态输入模型进行批量预测，得到每个动作对应的Q值，找到最大的Q值之后再回溯到对应的动作上
        next_actions, next_states = zip(*next_steps.items())
        # 将状态列表堆叠成一个张量批次，方便批量预测
        next_states = torch.stack(next_states)
        # 如果使用GPU，将状态张量移到GPU
        if torch.cuda.is_available():
            next_states = next_states.cuda()

        # 将模型设置为评估模式（关闭dropout等）
        model.eval()
        # 使用torch.no_grad()禁用梯度计算，节省内存和加速推理
        with torch.no_grad():
            # 使用模型预测所有可能动作的Q值
            # [:, 0]提取第一列，因为模型输出形状是(batch_size, 1)
            predictions = model(next_states)[:, 0]
        # 将模型切换回训练模式
        model.train()

        # 根据epsilon-greedy策略选择动作
        if random_action:
            # 探索：随机选择一个动作索引
            index = randint(0, len(next_steps) - 1)
        else:
            # 利用：选择Q值最大的动作索引
            index = torch.argmax(predictions).item()
            # .item()将单元素张量转换为Python数值

        # 获取选中的下一个状态
        # [index, :]表示选择第index行的所有列，即选中的下一个状态
        next_state = next_states[index, :]
        # 获取选中的动作
        action = next_actions[index]

        # 在环境中执行选中的动作
        # render=True表示渲染游戏画面
        # 返回奖励和游戏是否结束的标志
        reward, done = env.step(action, render=True)

        # 如果使用GPU，将下一个状态移到GPU
        if torch.cuda.is_available():
            next_state = next_state.cuda()

        # 将经验(状态, 奖励, 下一状态, 是否结束)存入经验回放缓冲区
        replay_memory.append([state, reward, next_state, done])

        # 如果游戏结束
        if done:
            # 记录本局游戏的最终得分
            final_score = env.score
            # 记录本局游戏放置的方块总数
            final_tetrominoes = env.tetrominoes
            # 记录本局游戏消除的总行数
            final_cleared_lines = env.cleared_lines
            # 重置游戏环境，开始新的一局
            state = env.reset()
            # 如果使用GPU，将新状态移到GPU
            if torch.cuda.is_available():
                state = state.cuda()
        else:
            # 如果游戏未结束，更新当前状态为下一个状态
            state = next_state
            # 跳过本次循环的剩余部分（不进行训练）
            continue

        # 如果经验回放缓冲区还没有积累足够的经验（少于容量的10%）
        # 则跳过训练，继续收集经验
        if len(replay_memory) < opt.replay_memory_size / 10:
            continue

        # 训练轮数加1
        epoch += 1

        # 从经验回放缓冲区中随机采样一批经验用于训练
        # 采样数量为batch_size和当前缓冲区大小的较小值
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        # 将批次数据拆分为状态、奖励、下一状态和结束标志
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        # 将状态列表堆叠成张量批次
        state_batch = torch.stack(tuple(state for state in state_batch))
        # 将奖励列表转换为numpy数组，再转换为PyTorch张量，并增加一个维度，使其形状为(batch_size, 1)，与预测Q值形状一致，方便目标Q值计算
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        # 将下一状态列表堆叠成张量批次
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        # 如果使用GPU，将所有批次数据移到GPU
        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        # 使用模型预测当前状态批次的Q值
        q_values = model(state_batch)

        # 将模型设置为评估模式
        model.eval()
        # 禁用梯度计算
        with torch.no_grad():
            # 预测下一状态批次的Q值
            next_prediction_batch = model(next_state_batch)
        # 切换回训练模式
        model.train()

        # 计算目标Q值（使用Bellman方程）
        # 如果游戏结束(done=True)，目标Q值 = 当前奖励
        # 如果游戏未结束(done=False)，目标Q值 = 当前奖励 + gamma * 下一状态的最大Q值
        # 遍历reward_batch、done_batch和next_prediction_batch，计算每个样本的目标Q值，并将结果堆叠成一个张量批次
        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        # 清空优化器的梯度
        optimizer.zero_grad()
        # 计算预测Q值与目标Q值之间的均方误差损失
        loss = criterion(q_values, y_batch)
        # 反向传播，计算梯度
        loss.backward()
        # 使用优化器更新模型参数
        optimizer.step()

        # 打印当前训练进度信息
        print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
            epoch,
            opt.num_epochs,
            action,
            final_score,
            final_tetrominoes,
            final_cleared_lines))
        # 将得分记录到TensorBoard
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        # 将方块数记录到TensorBoard
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        # 将消除行数记录到TensorBoard
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        # 如果达到模型保存间隔，保存当前模型
        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))

    # 训练结束后，保存最终模型
    torch.save(model, "{}/tetris".format(opt.saved_path))


# 主程序入口
if __name__ == "__main__":
    # 获取命令行参数
    opt = get_args()
    # 开始训练深度Q网络
    train(opt)
