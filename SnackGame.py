import pygame
import random
import numpy as np
import sys
from enum import Enum
from collections import deque

# 初始化Pygame
pygame.init()

# 方向枚举
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 120, 255)
DARK_GREEN = (0, 180, 0)
GRAY = (40, 40, 40)
YELLOW = (255, 255, 0)

class SnakeGame:
    def __init__(self, width=800, height=600, grid_size=20, speed=10, ai_mode=False):
        """
        初始化贪吃蛇游戏
        
        参数:
            width: 游戏窗口宽度
            height: 游戏窗口高度
            grid_size: 网格大小
            speed: 游戏速度（帧率）
            ai_mode: 是否为AI训练模式
        """
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.speed = speed
        self.ai_mode = ai_mode
        
        # 计算网格行列数
        self.cols = width // grid_size
        self.rows = height // grid_size
        
        # 调整窗口大小以适应网格
        self.width = self.cols * grid_size
        self.height = self.rows * grid_size
        
        # 创建游戏窗口
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("贪吃蛇游戏 - AI训练模式" if ai_mode else "贪吃蛇游戏")
        
        # 游戏时钟
        self.clock = pygame.time.Clock()
        
        # 游戏字体
        self.font = pygame.font.SysFont('simhei', 36)
        self.small_font = pygame.font.SysFont('simhei', 24)
        
        # AI训练专用变量
        self.state_size = 15  # 基础状态向量大小
        if ai_mode:
            # 启用增强状态和奖励
            self.state_size = 15  # 增强状态向量大小
            self.steps_without_food = 0  # 记录多少步没吃到食物
            self.last_distance = None
            self.total_steps = 0
            self.warning_zones = {
                'wall': 2,      # 距离墙2格时警告
                'self': 1       # 距离自己身体1格时警告
            }
            # 原地转圈检测
            self.head_position_history = deque(maxlen=10)  # 最近10步的头部位置
            self.circle_detection_window = 10  # 检测窗口大小
        
        # 初始化游戏状态
        self.reset_game()
        
    def reset_game(self):
        """重置游戏状态"""
        # 蛇的初始位置（在屏幕中央）
        start_x = self.cols // 2
        start_y = self.rows // 2
        
        # 蛇身初始为3个格子
        self.snake = [(start_x, start_y), 
                     (start_x - 1, start_y), 
                     (start_x - 2, start_y)]
        
        # 初始方向向右
        self.direction = Direction.RIGHT
        
        # 生成第一个食物
        self.food = self.generate_food()
        
        # 游戏状态
        self.score = 0
        self.game_over = False
        self.paused = False
        self.frame_iteration = 0
        self.total_reward = 0
        
        # AI训练专用状态重置
        if self.ai_mode:
            self.steps_without_food = 0
            self.last_distance = None
            self.total_steps = 0
            self.head_position_history.clear()
        
        # 特殊状态（人类玩家模式）
        self.speed_boost = False
    
    def generate_food(self):
        """生成食物位置"""
        while True:
            # 随机生成食物位置
            food = (random.randint(0, self.cols - 1), 
                   random.randint(0, self.rows - 1))
            
            # 确保食物不在蛇身上
            if food not in self.snake:
                return food
    
    def change_direction(self, new_direction):
        """改变蛇的移动方向（不能直接反向移动）"""
        # 检查新方向是否与当前方向相反
        if (new_direction == Direction.UP and self.direction != Direction.DOWN) or \
           (new_direction == Direction.DOWN and self.direction != Direction.UP) or \
           (new_direction == Direction.LEFT and self.direction != Direction.RIGHT) or \
           (new_direction == Direction.RIGHT and self.direction != Direction.LEFT):
            self.direction = new_direction
    
    def is_collision(self, point=None):
        """
        检查是否发生碰撞
        
        参数:
            point: 要检查的点，如果为None则检查蛇头
            
        返回:
            bool: 是否发生碰撞
        """
        if point is None:
            point = self.snake[0]
        
        # 检查是否撞墙
        if (point[0] < 0 or point[0] >= self.cols or 
            point[1] < 0 or point[1] >= self.rows):
            return True
        
        # 检查是否撞到自己
        if point in self.snake[1:]:
            return True
        
        return False
    
    def move_snake(self):
        """移动蛇"""
        if self.game_over or self.paused:
            return
        
        # 增加帧迭代计数
        self.frame_iteration += 1
        
        # 获取蛇头当前位置
        head_x, head_y = self.snake[0]
        
        # 根据当前方向计算新的蛇头位置
        if self.direction == Direction.RIGHT:
            new_head = (head_x + 1, head_y)
        elif self.direction == Direction.LEFT:
            new_head = (head_x - 1, head_y)
        elif self.direction == Direction.UP:
            new_head = (head_x, head_y - 1)
        elif self.direction == Direction.DOWN:
            new_head = (head_x, head_y + 1)
        
        # 检查是否发生碰撞
        if self.is_collision(new_head):
            self.game_over = True
            return
        
        # 将新的蛇头添加到蛇身列表开头
        self.snake.insert(0, new_head)
        
        # 检查是否吃到食物
        if new_head == self.food:
            # 增加分数
            self.score += 1
            
            # 生成新的食物
            self.food = self.generate_food()
            
            # 吃到食物时给予短暂加速效果（仅人类玩家模式）
            if not self.ai_mode:
                self.speed_boost = True
                pygame.time.set_timer(pygame.USEREVENT, 2000)  # 2秒后加速效果结束
        else:
            # 如果没有吃到食物，移除蛇尾
            self.snake.pop()
        
        # 检查是否陷入无限循环（AI训练时使用）
        if self.ai_mode and self.frame_iteration > 100 * len(self.snake):
            self.game_over = True
    
    def get_state(self):
        """
        获取当前游戏状态，作为神经网络的输入
        
        返回:
            np.array: 状态向量
        """
        if not self.ai_mode:
            # 人类玩家模式返回基础状态
            return self._get_basic_state()
        else:
            # AI训练模式返回增强状态
            return self._get_enhanced_state()
    
    def _get_basic_state(self):
        """获取基础状态表示（11个特征）"""
        # 获取蛇头位置
        head = self.snake[0]
        
        # 计算相对食物的方向
        food_left = self.food[0] < head[0]
        food_right = self.food[0] > head[0]
        food_up = self.food[1] < head[1]
        food_down = self.food[1] > head[1]
        
        # 当前移动方向（布尔值）
        dir_left = self.direction == Direction.LEFT
        dir_right = self.direction == Direction.RIGHT
        dir_up = self.direction == Direction.UP
        dir_down = self.direction == Direction.DOWN
        
        # 计算危险：前方、左方、右方是否有障碍（墙或自身）
        # 根据当前方向计算前方、左方、右方的点
        if dir_right:
            point_straight = (head[0] + 1, head[1])
            point_left = (head[0], head[1] - 1)
            point_right = (head[0], head[1] + 1)
        elif dir_left:
            point_straight = (head[0] - 1, head[1])
            point_left = (head[0], head[1] + 1)
            point_right = (head[0], head[1] - 1)
        elif dir_up:
            point_straight = (head[0], head[1] - 1)
            point_left = (head[0] - 1, head[1])
            point_right = (head[0] + 1, head[1])
        else:  # dir_down
            point_straight = (head[0], head[1] + 1)
            point_left = (head[0] + 1, head[1])
            point_right = (head[0] - 1, head[1])
        
        danger_straight = self.is_collision(point_straight)
        danger_left = self.is_collision(point_left)
        danger_right = self.is_collision(point_right)
        
        # 组合成状态向量
        state = [
            # 危险方向
            danger_straight, danger_left, danger_right,
            # 移动方向
            dir_left, dir_right, dir_up, dir_down,
            # 食物方向
            food_left, food_right, food_up, food_down
        ]
        
        return np.array(state, dtype=int)
    
    def _get_enhanced_state(self):
        """获取增强的状态表示（17个特征）"""
        head = self.snake[0]
        
        # 1. 危险检测（增强版）
        dangers = self._get_enhanced_dangers(head)
        
        # 2. 移动方向
        dir_left = self.direction == Direction.LEFT
        dir_right = self.direction == Direction.RIGHT
        dir_up = self.direction == Direction.UP
        dir_down = self.direction == Direction.DOWN
        
        # 3. 食物方向（精确到8个方向）
        food_info = self._get_food_detailed_info(head)
        
        # 4. 距离信息
        distance = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        max_possible = self.cols + self.rows
        normalized_distance = distance / max_possible if max_possible > 0 else 0
        
        # 5. 身体长度信息
        normalized_length = len(self.snake) / 20.0  # 假设最大长度为20
        
        # 组合状态向量
        state = [
            # 危险（6个方向：前、前左、前右、左、右、后）
            *dangers,
            # 移动方向（4个）
            dir_left, dir_right, dir_up, dir_down,
            # 食物方向（2个：x方向、y方向）
            food_info['dx_norm'], food_info['dy_norm'],
            # 距离信息
            normalized_distance,
            # 身体长度
            normalized_length,
            # 饥饿度（步数没吃到食物）
            min(1.0, self.steps_without_food / 100.0)
        ]
        
        return np.array(state, dtype=float)
    
    def _get_enhanced_dangers(self, head):
        """获取增强的危险检测"""
        dangers = []
        
        # 检查8个方向：前、前左、前右、左、右、后左、后右、后
        directions = [
            (1, 0),   # 前
            (1, -1),  # 前左
            (1, 1),   # 前右
            (0, -1),  # 左
            (0, 1),   # 右
            (-1, -1), # 后左
            (-1, 1),  # 后右
            (-1, 0)   # 后
        ]
        
        # 根据当前方向旋转方向向量
        if self.direction == Direction.UP:
            rotation = [(dx, -dy) for dx, dy in directions]
        elif self.direction == Direction.DOWN:
            rotation = [(-dx, dy) for dx, dy in directions]
        elif self.direction == Direction.LEFT:
            rotation = [(-dy, -dx) for dx, dy in directions]
        else:  # RIGHT
            rotation = [(dy, dx) for dx, dy in directions]
        
        # 只取前6个方向（去掉后方的两个）
        for dx, dy in rotation[:6]:
            check_point = (head[0] + dx, head[1] + dy)
            # 检查是否碰撞
            danger = self.is_collision(check_point)
            # 如果是身体，检查距离
            if not danger and check_point in self.snake[1:]:
                # 计算到身体的精确距离
                body_dist = abs(dx) + abs(dy)
                danger = body_dist <= self.warning_zones['self']
            dangers.append(float(danger))
        
        return dangers
    
    def _get_food_detailed_info(self, head):
        """获取食物的详细信息"""
        dx = self.food[0] - head[0]
        dy = self.food[1] - head[1]
        
        # 归一化方向（-1, 0, 1）
        dx_norm = 0 if dx == 0 else (1 if dx > 0 else -1)
        dy_norm = 0 if dy == 0 else (1 if dy > 0 else -1)
        
        return {
            'dx': dx,
            'dy': dy,
            'dx_norm': dx_norm,
            'dy_norm': dy_norm,
        }
    
    def _get_manhattan_distance(self, point1, point2):
        """计算曼哈顿距离"""
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
    
    def _is_wall_collision(self):
        """检查是否是撞墙"""
        if not self.game_over:
            return False
        
        head = self.snake[0]
        return (head[0] < 0 or head[0] >= self.cols or 
                head[1] < 0 or head[1] >= self.rows)
    
    def _calculate_safety_reward(self, head):
        """计算安全奖励"""
        safety = 1.0
        
        # 检查距离墙壁的距离
        wall_dist_x = min(head[0], self.cols - 1 - head[0])
        wall_dist_y = min(head[1], self.rows - 1 - head[1])
        min_wall_dist = min(wall_dist_x, wall_dist_y)
        
        # 如果太靠近墙壁，减少安全分数
        if min_wall_dist <= self.warning_zones['wall']:
            safety -= 0.3 * (self.warning_zones['wall'] - min_wall_dist)
        
        # 检查距离身体的距离
        for i, body_part in enumerate(self.snake[1:], 1):
            dist = abs(head[0] - body_part[0]) + abs(head[1] - body_part[1])
            if dist <= self.warning_zones['self']:
                safety -= 0.2 * (self.warning_zones['self'] - dist)
        
        return max(0, safety)  # 确保非负
    
    def _detect_circling(self):
        """
        检测蛇是否在原地转圈
        
        返回:
            bool: True表示检测到原地转圈，False表示正常移动
        """
        if len(self.head_position_history) < self.circle_detection_window:
            return False
        
        # 获取历史位置列表
        history_list = list(self.head_position_history)
        
        # 计算最近N步内头部位置的重复次数
        # 如果在一个很小的区域内反复出现，说明在转圈
        unique_positions = len(set(history_list))
        
        # 如果最近10步内只访问了3个或更少的不同位置，认为是原地转圈
        if unique_positions <= 3:
            return True
        
        # 额外检查：计算头部位置的标准差
        positions_array = np.array(history_list)
        x_std = np.std(positions_array[:, 0])
        y_std = np.std(positions_array[:, 1])
        
        # 如果标准差都很小（在2格以内），认为是原地转圈
        if x_std < 1.5 and y_std < 1.5:
            return True
        
        return False
    
    def _reset_training_state(self):
        """重置训练状态"""
        self.last_distance = None
        self.steps_without_food = 0
    
    def _ai_action_to_direction(self, action):
        """将AI的动作转换为方向"""
        # action是一个三元数组，如 [1,0,0]直行，[0,1,0]右转，[0,0,1]左转
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):  # 直行
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):  # 右转
            new_dir = clock_wise[(idx + 1) % 4]
        else:  # [0, 0, 1] 左转
            new_dir = clock_wise[(idx - 1) % 4]
        
        self.direction = new_dir
    
    def play_step(self, action=None):
        """
        执行一步游戏
        参数:
            action: AI选择的动作，如果为None则由人类玩家控制
        返回:
            tuple: (奖励, 游戏是否结束, 当前分数)
        """
        # 初始化奖励
        reward = 0
        old_score = self.score
        if self.ai_mode and action is not None:
            # AI控制：将动作转换为方向
            self._ai_action_to_direction(action)
        
        # 处理事件（在AI模式下也处理退出事件）
        # self._handle_events()
        
        # 保存移动前的状态（仅AI模式）
        if self.ai_mode:
            head_before = self.snake[0]
            old_distance = self._get_manhattan_distance(head_before, self.food)
            if self.last_distance is None:
                self.last_distance = old_distance
        # 移动蛇
        self.move_snake()
        
        # 计算奖励（仅AI模式下）
        if self.ai_mode:
            # 记录当前头部位置用于原地转圈检测
            current_head = self.snake[0]
            self.head_position_history.append(current_head)
            # 更新步数计数器
            self.total_steps += 1
            self.steps_without_food += 1
            # 如果游戏结束，给予大的惩罚
            if self.game_over:
                # 死亡惩罚（根据原因调整）
                if self._is_wall_collision():
                    reward = -80  # 撞墙惩罚更重
                else:
                    reward = -30  # 撞自己
                # 如果是早期就死亡，惩罚更大
                if self.total_steps < 20:
                    reward *= 1.5
                self._reset_training_state()
            else:
                # 检查是否吃到食物
                ate_food = self.score > old_score
                if ate_food:
                    # 吃到食物的主要奖励
                    reward = 5000 
                    # 奖励效率：步数越少，奖励越高
                    efficiency_bonus = max(0, 5 - self.steps_without_food / 10)
                    reward += efficiency_bonus
                    # 重置饥饿计数器
                    self.steps_without_food = 0
                    # 根据身体长度给予额外奖励
                    length_bonus = len(self.snake) * 2
                    reward += length_bonus
                else:
                    # 密集奖励：基于距离变化
                    head_after = self.snake[0]
                    new_distance = self._get_manhattan_distance(head_after, self.food)
                    # 距离变化奖励（核心）防止蛇不吃食物只靠近食物
                    distance_change = self.last_distance - new_distance                    
                    if distance_change > 0:
                        # 距离缩短：给予与缩短距离成正比的奖励
                        reward = 2 * (1 + distance_change * 1)
                    elif distance_change < 0:
                        # 距离增加：惩罚
                        reward = -1 * (1 + abs(distance_change) * 0.3)
                    else:
                        # 距离不变：轻微惩罚
                        reward = -0.1                   
                    # 更新距离记录
                    self.last_distance = new_distance                
                    # 安全奖励：避免靠近墙壁和身体
                    safety_reward = self._calculate_safety_reward(head_after)
                    reward += safety_reward * 0.2  # 安全系数                 
                    # 检测原地转圈行为 - 巨大惩罚
                    if self._detect_circling():
                        circling_penalty = -100  # 原地转圈的巨大惩罚
                        reward += circling_penalty                   
                    # 探索奖励：鼓励探索新区域
                    if self.total_steps % 50 == 0 and len(self.snake) < 5:
                        reward += 0.5                  
                    # 饥饿惩罚：长时间没吃到食物
                    if self.steps_without_food > 50:
                        hunger_penalty = (self.steps_without_food - 50) * 0.01
                        reward -= hunger_penalty                  
                    # 存活奖励：每步微小奖励，鼓励存活
                    reward += 1
        # else:
        #     # 人类玩家模式：不计算奖励
        #     pass
        
        self.total_reward += reward
        
        return reward, self.game_over, self.score
    
    def _handle_events(self):
        """处理游戏事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
            # 处理加速效果结束事件（仅人类玩家模式）
            if event.type == pygame.USEREVENT and not self.ai_mode:
                self.speed_boost = False
        
            # 处理按键事件
            if event.type == pygame.KEYDOWN:
                # 方向控制（仅人类玩家模式）
                if not self.ai_mode:
                    if event.key == pygame.K_UP:
                        self.change_direction(Direction.UP)
                    elif event.key == pygame.K_DOWN:
                        self.change_direction(Direction.DOWN)
                    elif event.key == pygame.K_LEFT:
                        self.change_direction(Direction.LEFT)
                    elif event.key == pygame.K_RIGHT:
                        self.change_direction(Direction.RIGHT)
            
                # 空格键暂停/继续游戏
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
            
                # R键重新开始游戏
                elif event.key == pygame.K_r:
                    self.reset_game()
            
                # ESC键退出游戏
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
    
    def _draw_snake_eyes(self, head_rect):
        """绘制蛇的眼睛"""
        eye_size = max(self.grid_size // 8, 2)
        
        # 根据方向确定眼睛位置
        if self.direction == Direction.RIGHT:
            eye1_pos = (head_rect.right - eye_size*2, head_rect.top + eye_size*2)
            eye2_pos = (head_rect.right - eye_size*2, head_rect.bottom - eye_size*2)
        elif self.direction == Direction.LEFT:
            eye1_pos = (head_rect.left + eye_size, head_rect.top + eye_size*2)
            eye2_pos = (head_rect.left + eye_size, head_rect.bottom - eye_size*2)
        elif self.direction == Direction.UP:
            eye1_pos = (head_rect.left + eye_size*2, head_rect.top + eye_size)
            eye2_pos = (head_rect.right - eye_size*2, head_rect.top + eye_size)
        else:  # DOWN
            eye1_pos = (head_rect.left + eye_size*2, head_rect.bottom - eye_size)
            eye2_pos = (head_rect.right - eye_size*2, head_rect.bottom - eye_size)
        
        pygame.draw.circle(self.screen, WHITE, eye1_pos, eye_size)
        pygame.draw.circle(self.screen, WHITE, eye2_pos, eye_size)
    
    def draw(self):
        """绘制游戏画面"""
        # 绘制黑色背景
        self.screen.fill(BLACK)
        
        # 绘制网格线
        for x in range(0, self.width, self.grid_size):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, self.grid_size):
            pygame.draw.line(self.screen, GRAY, (0, y), (self.width, y), 1)
        
        # 绘制蛇
        for i, (x, y) in enumerate(self.snake):
            # 蛇头用不同颜色
            color = GREEN if i == 0 else DARK_GREEN
            rect = pygame.Rect(x * self.grid_size, y * self.grid_size, 
                              self.grid_size, self.grid_size)
            pygame.draw.rect(self.screen, color, rect)
            
            # 蛇头添加眼睛（仅人类玩家模式）
            if not self.ai_mode and i == 0:
                self._draw_snake_eyes(rect)
        
        # 绘制食物
        food_rect = pygame.Rect(self.food[0] * self.grid_size, 
                               self.food[1] * self.grid_size, 
                               self.grid_size, self.grid_size)
        pygame.draw.rect(self.screen, RED, food_rect)
        
        # 绘制分数
        score_text = self.font.render(f"分数: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # 绘制长度
        length_text = self.small_font.render(f"长度: {len(self.snake)}", True, WHITE)
        self.screen.blit(length_text, (10, 50))
        
        # 如果是AI模式，显示额外信息
        if self.ai_mode:
            # 绘制总奖励
            reward_text = self.small_font.render(f"总奖励: {self.total_reward:.2f}", True, YELLOW)
            self.screen.blit(reward_text, (10, 90))
            
            # 绘制帧数
            frame_text = self.small_font.render(f"帧数: {self.frame_iteration}", True, WHITE)
            self.screen.blit(frame_text, (10, 130))
        
        # 绘制加速效果提示（仅人类玩家模式）
        if self.speed_boost and not self.ai_mode:
            boost_text = self.small_font.render("加速中!", True, BLUE)
            self.screen.blit(boost_text, (self.width - 100, 10))
        
        # 如果游戏暂停，显示暂停文本
        if self.paused:
            pause_text = self.font.render("游戏暂停 - 按空格键继续", True, BLUE)
            text_rect = pause_text.get_rect(center=(self.width//2, self.height//2))
            self.screen.blit(pause_text, text_rect)
        
        # 如果游戏结束，显示游戏结束文本
        if self.game_over:
            # 半透明遮罩
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            # 游戏结束文本
            game_over_text = self.font.render("游戏结束!", True, RED)
            text_rect = game_over_text.get_rect(center=(self.width//2, self.height//2 - 40))
            self.screen.blit(game_over_text, text_rect)
            
            # 最终分数
            final_score_text = self.font.render(f"最终分数: {self.score}", True, WHITE)
            score_rect = final_score_text.get_rect(center=(self.width//2, self.height//2))
            self.screen.blit(final_score_text, score_rect)
            
            # 重新开始提示
            restart_text = self.small_font.render("按 R 键重新开始，ESC 键退出", True, BLUE)
            restart_rect = restart_text.get_rect(center=(self.width//2, self.height//2 + 40))
            self.screen.blit(restart_text, restart_rect)
        
        # 绘制操作说明
        if not self.game_over:
            if self.ai_mode:
                controls_text = self.small_font.render("AI训练中 - 空格键暂停，R键重置，ESC键退出", True, WHITE)
                self.screen.blit(controls_text, (self.width - 400, self.height - 30))
            else:
                controls_text1 = self.small_font.render("方向键控制移动，空格键暂停/继续", True, WHITE)
                controls_text2 = self.small_font.render("R键重新开始，ESC键退出", True, WHITE)
                self.screen.blit(controls_text1, (self.width - 300, self.height - 40))
                self.screen.blit(controls_text2, (self.width - 300, self.height - 20))
        
        # 更新显示
        pygame.display.flip()
    
    def update_display(self):
        """更新显示并控制帧率"""
        pygame.display.flip()
        current_speed = self.speed + 5 if self.speed_boost and not self.ai_mode else self.speed
        self.clock.tick(current_speed)