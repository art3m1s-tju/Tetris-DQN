# 俄罗斯方块 DQN (Deep Q-Network)

基于深度 Q 网络（DQN）的俄罗斯方块 AI 智能体，使用人工特征进行强化学习训练。

![Demo](demo/tetris.gif)

## 📋 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [技术细节](#技术细节)
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [使用方法](#使用方法)
- [项目结构](#项目结构)
- [训练参数](#训练参数)
- [效果展示](#效果展示)

## 项目简介

本项目实现了一个基于深度强化学习的俄罗斯方块 AI 智能体。与传统的基于像素输入的 DQN 不同，本项目采用**人工设计的特征**作为状态表示，使得训练更加高效且收敛更快。

## 核心特性

- ✅ **人工特征提取**：使用 4 个精心设计的特征表示游戏状态
- ✅ **深度 Q 学习**：基于 DQN 算法进行训练
- ✅ **经验回放**：使用经验回放机制提高样本利用效率
- ✅ **Epsilon-Greedy 策略**：平衡探索与利用
- ✅ **TensorBoard 可视化**：实时监控训练过程
- ✅ **视频录制**：自动保存游戏过程为视频文件

## 技术细节

### 状态表示（人工特征）

本项目使用 4 个人工设计的特征来表示游戏状态，而非原始像素：

1. **消除行数** (lines_cleared)：当前步骤消除的行数
2. **空洞数量** (holes)：被方块覆盖下方的空位数量
3. **凹凸度** (bumpiness)：相邻列高度差的绝对值之和
4. **总高度** (height)：所有列高度的总和

```python
# 状态特征提取 (src/tetris.py)
def get_state_properties(self, board):
    lines_cleared, board = self.check_cleared_rows(board)
    holes = self.get_holes(board)
    bumpiness, height = self.get_bumpiness_and_height(board)
    return torch.FloatTensor([lines_cleared, holes, bumpiness, height])
```

### 网络架构

简单的全连接神经网络：

```
输入层：4 个特征
隐藏层 1：64 个神经元 + ReLU 激活函数
隐藏层 2：64 个神经元 + ReLU 激活函数
输出层：1 个神经元（Q 值）
```

```python
# 网络结构 (src/deep_q_network.py)
self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
self.conv3 = nn.Sequential(nn.Linear(64, 1))
```

### DQN 算法

- **经验回放**：存储历史经验 (state, reward, next_state, done)，随机采样进行训练
- **目标网络更新**：使用 Bellman 方程计算目标 Q 值
- **Epsilon-Greedy**：探索率从 1.0 线性衰减到 0.001

## 环境要求

- Python 3.7+
- PyTorch 1.0+
- CUDA（可选，用于 GPU 加速）

## 安装步骤

1. **克隆仓库**

```bash
git clone https://github.com/ar3m1s-tju/Tetris-DQN.git
cd Tetris-DQN
```

2. **安装依赖**

```bash
pip install torch torchvision
pip install numpy opencv-python pillow matplotlib
pip install tensorboardX
```

或者使用以下命令一次性安装：

```bash
pip install torch numpy opencv-python pillow matplotlib tensorboardX
```

## 使用方法

### 训练模型

使用默认参数训练：

```bash
python train.py
```

自定义训练参数：

```bash
python train.py --width 10 --height 20 --num_epochs 3000 --batch_size 512 --lr 0.001
```

训练过程中会：
- 在终端显示实时训练进度
- 在 `tensorboard/` 目录保存训练日志
- 每 1000 轮保存一次模型到 `trained_models/` 目录

### 查看训练过程

使用 TensorBoard 可视化训练过程：

```bash
tensorboard --logdir=tensorboard
```

然后在浏览器中打开 `http://localhost:6006`

### 测试模型

使用训练好的模型进行测试：

```bash
python test.py
```

自定义测试参数：

```bash
python test.py --saved_path trained_models --output output.mp4 --fps 300
```

测试完成后会生成游戏视频文件 `output.mp4`

## 项目结构

```
Tetris_DQN/
├── src/
│   ├── tetris.py              # 俄罗斯方块游戏环境
│   └── deep_q_network.py      # DQN 网络模型
├── train.py                   # 训练脚本
├── test.py                    # 测试脚本
├── trained_models/            # 保存的模型文件
│   └── tetris                 # 训练好的模型
├── tensorboard/               # TensorBoard 日志
├── demo/                      # 演示文件
│   └── tetris.gif            # 演示 GIF
└── README.md                  # 项目说明文档
```

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--width` | 10 | 游戏板宽度（列数） |
| `--height` | 20 | 游戏板高度（行数） |
| `--block_size` | 30 | 方块像素大小 |
| `--batch_size` | 512 | 批量大小 |
| `--lr` | 0.001 | 学习率 |
| `--gamma` | 0.99 | 折扣因子 |
| `--initial_epsilon` | 1.0 | 初始探索率 |
| `--final_epsilon` | 0.001 | 最终探索率 |
| `--num_decay_epochs` | 2000 | 探索率衰减轮数 |
| `--num_epochs` | 3000 | 总训练轮数 |
| `--save_interval` | 1000 | 模型保存间隔 |
| `--replay_memory_size` | 30000 | 经验回放池大小 |

## 效果展示

训练好的 AI 智能体能够：
- 自动选择最优的方块放置位置和旋转角度
- 尽可能消除更多行以获得高分
- 避免产生空洞和过高的堆叠
- 保持游戏板表面平整

## 算法原理

### 奖励函数

```python
# 基础分 1 分 + (消除行数^2 × 游戏板宽度)
score = 1 + (lines_cleared ** 2) * width
# 游戏结束惩罚 -2 分
if gameover:
    score -= 2
```

这种奖励设计鼓励智能体：
- 同时消除多行（奖励呈平方增长）
- 避免游戏结束

### Bellman 方程

```python
# 如果游戏结束
Q_target = reward
# 如果游戏未结束
Q_target = reward + gamma * max(Q(next_state))
```

## 技术亮点

1. **高效的状态表示**：使用 4 个人工特征代替高维像素输入，大幅降低计算复杂度
2. **快速收敛**：相比基于像素的 DQN，训练速度提升数倍
3. **稳定训练**：经验回放机制打破样本相关性，提高训练稳定性
4. **可解释性强**：人工特征具有明确的物理意义，便于理解和调试

## 未来改进方向

- [ ] 实现 Double DQN 减少 Q 值过估计
- [ ] 添加 Dueling DQN 架构
- [ ] 实现优先经验回放（Prioritized Experience Replay）
- [ ] 尝试基于像素的端到端学习
- [ ] 添加多智能体对战模式

## 参考资料

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)

## 许可证

MIT License

## 致谢

感谢所有为深度强化学习领域做出贡献的研究者和开发者。

---

如有问题或建议，欢迎提交 Issue 或 Pull Request！
