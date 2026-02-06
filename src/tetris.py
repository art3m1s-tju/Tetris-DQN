# 导入numpy库，用于数组和矩阵运算
import numpy as np
# 导入PIL的Image模块，用于图像处理
from PIL import Image
# 导入OpenCV库，用于图像显示和视频处理
import cv2
# 导入matplotlib的style模块，用于设置绘图风格
from matplotlib import style
# 导入PyTorch库，用于深度学习和张量运算
import torch
# 导入random库，用于随机数生成
import random

# 使用ggplot风格的绘图样式
style.use("ggplot")


class Tetris:
    # 定义每个方块的颜色（RGB格式）
    # 索引0: 黑色（空白）
    # 索引1: 黄色（O型方块）
    # 索引2: 紫色（T型方块）
    # 索引3: 青绿色（Z型方块）
    # 索引4: 红色（S型方块）
    # 索引5: 浅蓝色（I型方块）
    # 索引6: 橙色（L型方块）
    # 索引7: 深蓝色（J型方块）
    piece_colors = [
        (0, 0, 0),
        (255, 255, 0),
        (147, 88, 254),
        (54, 175, 144),
        (255, 0, 0),
        (102, 217, 238),
        (254, 151, 32),
        (0, 0, 255)
    ]

    # 定义所有7种俄罗斯方块的形状
    # 每个方块用二维数组表示，数字对应piece_colors中的颜色索引
    pieces = [
        # O型方块（正方形）
        [[1, 1],
         [1, 1]],

        # T型方块
        [[0, 2, 0],
         [2, 2, 2]],

        # Z型方块
        [[0, 3, 3],
         [3, 3, 0]],

        # S型方块
        [[4, 4, 0],
         [0, 4, 4]],

        # I型方块（直线）
        [[5, 5, 5, 5]],

        # L型方块
        [[0, 0, 6],
         [6, 6, 6]],

        # J型方块
        [[7, 0, 0],
         [7, 7, 7]]
    ]

    def __init__(self, height=20, width=10, block_size=20):
        # 初始化游戏板的高度（默认20行）
        self.height = height
        # 初始化游戏板的宽度（默认10列）
        self.width = width
        # 初始化每个方块的像素大小（默认20像素）
        self.block_size = block_size
        # 创建额外的显示区域（用于显示分数等信息）
        # 大小为：高度*block_size 行，宽度*block_size/2 列，3个颜色通道
        # 填充为淡紫色 (204, 204, 255)
        self.extra_board = np.ones((self.height * self.block_size, self.width * int(self.block_size / 2), 3),
                                   dtype=np.uint8) * np.array([204, 204, 255], dtype=np.uint8)
        # 设置文本颜色为紫红色
        self.text_color = (200, 20, 220)
        # 调用reset方法初始化游戏状态
        self.reset()

    def reset(self):
        # 创建一个空的游戏板，所有位置初始化为0（空白）
        self.board = [[0] * self.width for _ in range(self.height)]
        # 初始化分数为0
        self.score = 0
        # 初始化已放置的方块数量为0
        self.tetrominoes = 0
        # 初始化已消除的行数为0
        self.cleared_lines = 0
        # 创建一个包含所有方块索引的"袋子"（用于随机生成方块），索引范围为0到6，对应7种方块，每一次打乱七种方块的顺序并从中取出一个方块，直到袋子空了再重新打乱生成新的袋子，保证每7个方块中包含所有7种不同的方块，避免长时间不出现某种方块的情况
        self.bag = list(range(len(self.pieces)))
        # 随机打乱袋子中的方块顺序
        random.shuffle(self.bag)
        # 从袋子中取出一个方块索引
        self.ind = self.bag.pop()
        # 根据索引获取对应的方块形状（深拷贝，不能直接对原数据进行操作）
        self.piece = [row[:] for row in self.pieces[self.ind]]
        # 设置当前方块的初始位置（水平居中，垂直在顶部），注意计算机的坐标原点在左上角，所以y坐标为0表示在顶部，x坐标根据方块宽度计算使其水平居中
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0}
        # 初始化游戏结束标志为False
        self.gameover = False
        # 返回当前游戏板的状态属性
        return self.get_state_properties(self.board)

    def rotate(self, piece):
        # 获取原始方块的行数（也是旋转后的列数）
        num_rows_orig = num_cols_new = len(piece)
        # 获取旋转后的行数（原始方块的列数）
        num_rows_new = len(piece[0])
        # 创建空列表存储旋转后的方块
        rotated_array = []

        # 遍历旋转后的每一行
        for i in range(num_rows_new):
            # 创建新行，初始化为0
            new_row = [0] * num_cols_new
            # 遍历旋转后的每一列
            for j in range(num_cols_new):
                # 顺时针旋转90度：新位置[i][j] = 原位置[num_rows_orig-1-j][i]
                new_row[j] = piece[(num_rows_orig - 1) - j][i]
            # 将新行添加到旋转后的数组中
            rotated_array.append(new_row)
        # 返回旋转后的方块
        return rotated_array

    def get_state_properties(self, board):
        # 检查并清除已填满的行，返回清除的行数和更新后的游戏板，返回新的游戏版来计算新的空洞数量、凹凸度和高度
        lines_cleared, board = self.check_cleared_rows(board)
        # 计算游戏板中的空洞数量
        holes = self.get_holes(board)
        # 计算游戏板的凹凸不平程度和总高度
        bumpiness, height = self.get_bumpiness_and_height(board)

        # 将这些状态属性转换为PyTorch张量并返回
        # 这4个特征用于深度Q学习的状态表示
        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])

    def get_holes(self, board):
        # 初始化空洞计数器
        num_holes = 0
        # 使用zip(*board)转置游戏板，按列遍历！
        for col in zip(*board):
            # 从顶部开始查找第一个非空方块
            row = 0
            # 跳过列顶部的所有空位置
            while row < self.height and col[row] == 0:
                row += 1
            # 计算第一个非空方块下方的所有空位置（这些是空洞）
            # 空洞定义：在某列中，位于已放置方块下方的空位置
            num_holes += len([x for x in col[row + 1:] if x == 0])
        # 返回总空洞数
        return num_holes

    def get_bumpiness_and_height(self, board):
        # 将游戏板转换为numpy数组以便进行向量化操作
        board = np.array(board)
        # 创建布尔掩码，标记所有非空位置
        mask = board != 0
        # 计算每列的"反向高度"（从顶部到第一个非空方块的距离）
        # 如果列中有方块，返回第一个非空方块的行索引；否则返回height
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        # 计算每列的实际高度（从底部算起）
        heights = self.height - invert_heights
        # 计算所有列的总高度
        total_height = np.sum(heights)
        # 获取除最后一列外的所有列高度
        currs = heights[:-1]
        # 获取除第一列外的所有列高度
        nexts = heights[1:]
        # 计算相邻列之间的高度差的绝对值
        diffs = np.abs(currs - nexts)
        # 计算总凹凸度（所有相邻列高度差之和）
        # 凹凸度越大，表示游戏板表面越不平整
        total_bumpiness = np.sum(diffs)
        # 返回总凹凸度和总高度
        return total_bumpiness, total_height

    def get_next_states(self):
        # 创建字典存储所有可能的下一个状态
        states = {}
        # 获取当前方块的ID
        piece_id = self.ind
        # 深拷贝当前方块
        curr_piece = [row[:] for row in self.piece]
        # 根据方块类型确定旋转次数
        if piece_id == 0:  # O型方块（正方形）
            num_rotations = 1  # 只有1种旋转状态（旋转后形状相同）
        elif piece_id == 2 or piece_id == 3 or piece_id == 4:  # Z型、S型、I型方块
            num_rotations = 2  # 有2种旋转状态
        else:  # T型、L型、J型方块
            num_rotations = 4  # 有4种旋转状态

        # 遍历所有可能的旋转状态
        for i in range(num_rotations):
            # 计算当前旋转状态下，方块可以放置的最大x坐标，防止方块超出游戏板右边界
            # 方块的坐标原点是方块的左上角，所以最大x坐标 = 游戏板宽度 - 方块宽度，最小x坐标为0
            valid_xs = self.width - len(curr_piece[0])
            # 遍历所有可能的x坐标位置
            for x in range(valid_xs + 1):
                # 深拷贝当前方块
                piece = [row[:] for row in curr_piece]
                # 设置方块的初始位置
                pos = {"x": x, "y": 0}
                # 让方块下落直到发生碰撞
                while not self.check_collision(piece, pos):
                    pos["y"] += 1
                # 截断方块，如果方块溢出游戏板顶部，truncate方法会删除方块的顶部行直到不再溢出，如果仍然无法放置，返回游戏结束标志
                self.truncate(piece, pos)
                # 将方块存储到游戏板上
                board = self.store(piece, pos)
                # 计算并存储这个状态的属性，键为(x坐标, 旋转次数)
                states[(x, i)] = self.get_state_properties(board)
            # 旋转方块以尝试下一个旋转状态
            curr_piece = self.rotate(curr_piece)
        # 返回所有可能的状态字典
        return states

    def get_current_board_state(self):
        # 深拷贝当前游戏板，正在下落的方块还没有固定到游戏板上，所以需要在副本上添加当前方块的位置来计算状态属性
        board = [x[:] for x in self.board]
        # 将当前正在下落的方块添加到游戏板副本中
        for y in range(len(self.piece)):
            for x in range(len(self.piece[y])):
                # 在游戏板上标记当前方块的位置
                board[y + self.current_pos["y"]][x + self.current_pos["x"]] = self.piece[y][x]
        # 返回包含当前方块的游戏板状态
        return board

    def new_piece(self):
        # 如果袋子为空（所有方块都已使用）
        if not len(self.bag):
            # 重新创建包含所有方块索引的袋子
            self.bag = list(range(len(self.pieces)))
            # 随机打乱袋子（确保方块随机出现）
            random.shuffle(self.bag)
        # 从袋子中取出一个方块索引
        self.ind = self.bag.pop()
        # 根据索引获取对应的方块形状（深拷贝）
        self.piece = [row[:] for row in self.pieces[self.ind]]
        # 设置新方块的初始位置（水平居中，垂直在顶部）
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2,
                            "y": 0
                            }
        # 检查新方块是否与已有方块发生碰撞，如果发生碰撞，说明游戏板已满，新方块连落都落不下来，游戏结束
        if self.check_collision(self.piece, self.current_pos):
            # 如果发生碰撞，说明游戏板已满，游戏结束
            self.gameover = True

    def check_collision(self, piece, pos):
        # 计算方块下一步的y坐标
        future_y = pos["y"] + 1
        # 遍历方块的每个单元格
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                # 检查是否发生碰撞：
                # 1. 方块超出游戏板底部，撞击地板 (future_y + y > self.height - 1)
                # 2. 方块与游戏板上已有方块重叠 (self.board[future_y + y][pos["x"] + x] and piece[y][x])
                if future_y + y > self.height - 1 or self.board[future_y + y][pos["x"] + x] and piece[y][x]:
                    return True  # 发生碰撞
        return False  # 未发生碰撞

    def truncate(self, piece, pos):
        # 初始化游戏结束标志
        gameover = False
        # 记录最后一个发生碰撞的行索引
        last_collision_row = -1
        # 遍历方块的每个单元格
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                # 检查方块是否与游戏板上已有方块重叠
                if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x]:
                    # 更新最后碰撞行
                    if y > last_collision_row:
                        last_collision_row = y

        # 如果方块溢出游戏板顶部（即使截断后仍然超出边界）
        if pos["y"] - (len(piece) - last_collision_row) < 0 and last_collision_row > -1:
            # 持续删除方块的顶部行，直到不再溢出
            while last_collision_row >= 0 and len(piece) > 1:
                gameover = True  # 标记游戏结束
                last_collision_row = -1
                del piece[0]  # 删除方块的第一行
                # 重新检查是否还有碰撞
                for y in range(len(piece)):
                    for x in range(len(piece[y])):
                        if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x] and y > last_collision_row:
                            last_collision_row = y
        # 返回游戏结束标志
        return gameover

    def store(self, piece, pos):
        # 每当方块固定到游戏板上时，调用这个方法将方块的形状存储到游戏板的对应位置
        # 深拷贝当前游戏板
        board = [x[:] for x in self.board]
        # 遍历方块的每个单元格
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                # 如果方块单元格非空且游戏板对应位置为空
                if piece[y][x] and not board[y + pos["y"]][x + pos["x"]]:
                    # 将方块存储到游戏板上
                    board[y + pos["y"]][x + pos["x"]] = piece[y][x]
        # 返回更新后的游戏板
        return board

    def check_cleared_rows(self, board):
        # 负责检查游戏板上是否有已填满的行，并将这些行删除，同时返回清除的行数和更新后的游戏板
        # 创建列表存储需要删除的行索引
        to_delete = []
        # 从底部向顶部遍历游戏板（使用[::-1]反转）
        for i, row in enumerate(board[::-1]):
            # 如果行中没有空位置（即行已填满）
            if 0 not in row:
                # 将该行的实际索引添加到删除列表
                to_delete.append(len(board) - 1 - i)
        # 如果有需要删除的行
        if len(to_delete) > 0:
            # 删除这些行
            board = self.remove_row(board, to_delete)
        # 返回清除的行数和更新后的游戏板
        return len(to_delete), board

    def remove_row(self, board, indices):
        # 负责删除游戏板上指定索引的行，并在顶部添加相同数量的空行，保持游戏板的高度不变
        # 从后向前遍历要删除的行索引（避免索引变化问题）
        for i in indices[::-1]:
            # 删除指定行
            del board[i]
            # 在顶部添加一行空行（所有位置为0）
            board = [[0 for _ in range(self.width)]] + board
        # 返回更新后的游戏板
        return board

    def step(self, action, render=True, video=None):
        # 解包动作：x坐标和旋转次数
        x, num_rotations = action
        # 设置当前方块的位置（x坐标由动作指定，y坐标从顶部开始）
        self.current_pos = {"x": x, "y": 0}
        # 根据动作指定的旋转次数旋转方块
        for _ in range(num_rotations):
            self.piece = self.rotate(self.piece)

        # 让方块下落直到发生碰撞
        while not self.check_collision(self.piece, self.current_pos):
            self.current_pos["y"] += 1  # 方块向下移动一格
            # 如果需要渲染，显示当前状态
            if render:
                self.render(video)

        # 截断方块（处理溢出情况）
        overflow = self.truncate(self.piece, self.current_pos)
        # 如果发生溢出，游戏结束
        if overflow:
            self.gameover = True

        # 将方块存储到游戏板上
        self.board = self.store(self.piece, self.current_pos)

        # 检查并清除已填满的行
        lines_cleared, self.board = self.check_cleared_rows(self.board)
        # 计算得分：基础分1分 + (消除行数^2 * 游戏板宽度)，消除行数越多得分越高，鼓励agent同时消除更多行
        score = 1 + (lines_cleared ** 2) * self.width
        # 更新总分
        self.score += score
        # 增加已放置的方块数量
        self.tetrominoes += 1
        # 增加已消除的总行数
        self.cleared_lines += lines_cleared
        # 如果游戏未结束，生成新方块
        if not self.gameover:
            self.new_piece()
        # 如果游戏结束，扣除2分作为惩罚
        if self.gameover:
            self.score -= 2

        # 返回本步得分和游戏结束标志
        return score, self.gameover

    def render(self, video=None):
        # 根据游戏状态选择要渲染的游戏板
        if not self.gameover:
            # 游戏进行中：获取包含当前下落方块的游戏板状态
            # 将每个位置的数字映射为对应的颜色
            img = [self.piece_colors[p] for row in self.get_current_board_state() for p in row]
        else:
            # 游戏结束：只显示已固定的方块
            img = [self.piece_colors[p] for row in self.board for p in row]
        # 将颜色列表转换为numpy数组，并重塑为(高度, 宽度, 3)的形状
        img = np.array(img).reshape((self.height, self.width, 3)).astype(np.uint8)
        # 将BGR颜色格式转换为RGB格式（OpenCV使用BGR，PIL使用RGB）
        img = img[..., ::-1]
        # 将numpy数组转换为PIL图像对象
        img = Image.fromarray(img, "RGB")

        # 将图像放大到指定的像素大小（每个方块block_size x block_size像素）
        img = img.resize((self.width * self.block_size, self.height * self.block_size), 0)
        # 将PIL图像转换回numpy数组
        img = np.array(img)
        # 在每个方块之间绘制水平网格线（设置为黑色）
        img[[i * self.block_size for i in range(self.height)], :, :] = 0
        # 在每个方块之间绘制垂直网格线（设置为黑色）
        img[:, [i * self.block_size for i in range(self.width)], :] = 0

        # 将额外的信息显示区域拼接到游戏板右侧
        img = np.concatenate((img, self.extra_board), axis=1)


        # 在额外区域显示"Score:"标签
        cv2.putText(img, "Score:", (self.width * self.block_size + int(self.block_size / 2), self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        # 显示当前分数
        cv2.putText(img, str(self.score),
                    (self.width * self.block_size + int(self.block_size / 2), 2 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        # 显示"Pieces:"标签
        cv2.putText(img, "Pieces:", (self.width * self.block_size + int(self.block_size / 2), 4 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        # 显示已放置的方块数量
        cv2.putText(img, str(self.tetrominoes),
                    (self.width * self.block_size + int(self.block_size / 2), 5 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        # 显示"Lines:"标签
        cv2.putText(img, "Lines:", (self.width * self.block_size + int(self.block_size / 2), 7 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        # 显示已消除的行数
        cv2.putText(img, str(self.cleared_lines),
                    (self.width * self.block_size + int(self.block_size / 2), 8 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        # 如果提供了视频写入对象，将当前帧写入视频
        if video:
            video.write(img)

        # 使用OpenCV显示图像窗口
        cv2.imshow("Deep Q-Learning Tetris", img)
        # 等待1毫秒（允许OpenCV处理窗口事件）
        cv2.waitKey(1)
