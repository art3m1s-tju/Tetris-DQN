# 导入argparse库，用于解析命令行参数
import argparse
# 导入PyTorch库，用于加载和运行深度学习模型
import torch
# 导入OpenCV库，用于视频录制
import cv2
# 从src.tetris模块导入Tetris游戏类
from src.tetris import Tetris


# 定义函数：获取命令行参数
def get_args():
    # 创建参数解析器，设置程序描述
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")

    # 添加游戏板宽度参数（默认10列）
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    # 添加游戏板高度参数（默认20行）
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    # 添加方块大小参数（默认30像素）
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    # 添加视频帧率参数（默认300 FPS）
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    # 添加模型保存路径参数（默认"trained_models"目录）
    parser.add_argument("--saved_path", type=str, default="trained_models")
    # 添加输出视频文件名参数（默认"output.mp4"）
    parser.add_argument("--output", type=str, default="output.mp4")

    # 解析命令行参数
    args = parser.parse_args()
    # 返回解析后的参数对象
    return args


# 定义测试函数：加载训练好的模型并运行游戏
def test(opt):
    # 如果CUDA可用（有GPU）
    if torch.cuda.is_available():
        # 设置CUDA随机种子，确保结果可复现
        torch.cuda.manual_seed(123)
    else:
        # 如果只有CPU，设置CPU随机种子
        torch.manual_seed(123)
    # 加载训练好的模型
    if torch.cuda.is_available():
        # 如果有GPU，直接加载模型到GPU
        model = torch.load("{}/tetris".format(opt.saved_path), weights_only=False)
    else:
        # 如果只有CPU，使用map_location将模型加载到CPU
        model = torch.load("{}/tetris".format(opt.saved_path), map_location=lambda storage, loc: storage, weights_only=False)
    # 将模型设置为评估模式（关闭dropout等训练特性）
    model.eval()
    # 创建Tetris游戏环境，使用命令行参数指定的尺寸
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    # 重置游戏环境到初始状态
    env.reset()
    # 如果CUDA可用，将模型移到GPU
    if torch.cuda.is_available():
        model.cuda()
    # 创建视频写入对象，用于录制游戏过程
    # 参数：输出文件名、编码格式(MJPG)、帧率、视频尺寸(宽度为1.5倍游戏板宽度以包含信息区域)
    out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"MJPG"), opt.fps,
                          (int(1.5*opt.width*opt.block_size), opt.height*opt.block_size))
    # 游戏主循环
    while True:
        # 获取所有可能的下一步状态
        next_steps = env.get_next_states()
        # 将字典拆分为动作列表和状态列表
        next_actions, next_states = zip(*next_steps.items())
        # 将状态列表堆叠成一个张量批次
        next_states = torch.stack(next_states)
        # 如果使用GPU，将状态张量移到GPU
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        # 使用模型预测每个状态的Q值
        # [:, 0]提取第一列（因为输出是(batch_size, 1)的形状）
        predictions = model(next_states)[:, 0]
        # 找到Q值最大的动作索引
        index = torch.argmax(predictions).item()
        # 获取对应的动作
        action = next_actions[index]
        # 执行动作，获取奖励和游戏是否结束的标志
        # render=True表示渲染游戏画面，video=out表示将画面写入视频
        _, done = env.step(action, render=True, video=out)

        # 如果游戏结束
        if done:
            # 释放视频写入对象，保存视频文件
            out.release()
            # 退出游戏循环
            break



# 主程序入口
if __name__ == "__main__":
    # 获取命令行参数
    opt = get_args()
    # 运行测试函数
    test(opt)
