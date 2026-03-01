"""
snake_ai.py - 贪吃蛇强化学习AI
包含DQN模型、训练器和智能体（增强版）
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os
import matplotlib.pyplot as plt

class LinearQNet(nn.Module):
    """使用层归一化的网络"""
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearQNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)  # 层归一化，对批量大小不敏感
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln2 = nn.LayerNorm(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)
    
    def save(self, file_name='model.pth'):
        """保存模型"""
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        print(f"✅ 模型已保存: {file_name}")
    
    def load(self, file_name='model.pth'):
        """加载模型"""
        model_folder_path = './models'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_name):
            try:
                self.load_state_dict(torch.load(file_name, weights_only=True))
                print(f"✅ 模型已加载: {file_name}")
                return True
            except Exception as e:
                print(f"❌ 加载模型失败: {e}")
                return False
        else:
            print(f"❌ 模型文件不存在: {file_name}")
            return False

class QTrainer:
    """改进的Q学习训练器（支持目标网络）"""
    def __init__(self, model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.losses = []  # 记录损失
    
    def train_step(self, states, actions, rewards, next_states, dones):
        """训练一步（批量）"""
        # 转换为PyTorch张量
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float).unsqueeze(1)
        
        # 1: 计算当前Q值
        current_q_values = self.model(states).gather(1, actions.argmax(dim=1).unsqueeze(1))
        
        # 2: 使用目标网络计算下一状态的最大Q值
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
        
        # 3: 计算目标Q值（贝尔曼方程）
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 4: 计算损失
        loss = self.criterion(current_q_values, target_q_values)
        
        # 5: 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 记录损失
        self.losses.append(loss.item())
        
        return loss.item()
    
    def update_target_network(self):
        """更新目标网络参数"""
        self.target_model.load_state_dict(self.model.state_dict())

class Agent:
    """改进的强化学习智能体"""
    def __init__(self, state_size=15, action_size=3, hidden_size=256):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # 学习参数
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.998  # 探索率衰减
        self.lr = 0.0005  # 学习率
        self.target_update_freq = 100  # 目标网络更新频率
        
        # 主网络和目标网络
        self.model = LinearQNet(state_size, hidden_size, action_size)
        self.target_model = LinearQNet(state_size, hidden_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # 训练器
        self.trainer = QTrainer(self.model, self.target_model, self.lr, self.gamma)
        
        # 经验回放
        self.memory = deque(maxlen=200_000)
        self.batch_size = 512
        
        # 训练记录
        self.n_games = 0
        self.scores = []
        self.mean_scores = []
        self.total_score = 0
        self.record = 0
        self.training_steps = 0
        self.avg_rewards = []
        
        # 探索策略
        self.random_exploration = True
        
        # 创建保存目录
        os.makedirs('./models', exist_ok=True)
        os.makedirs('./records', exist_ok=True)
    
    def get_state(self, game):
        """从游戏获取状态"""
        return game.get_state()
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验到记忆回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        """从记忆回放中训练（批量）"""
        if len(self.memory) < self.batch_size:
            return 0
        
        # 随机采样批次
        indices = random.sample(range(len(self.memory)), self.batch_size)
        batch = [self.memory[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 训练
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        
        # 定期更新目标网络
        if self.training_steps % self.target_update_freq == 0:
            self.trainer.update_target_network()
        
        return loss
    
    def train_short_memory(self, state, action, reward, next_state, done):
        """单步训练"""
        loss = self.trainer.train_step([state], [action], [reward], [next_state], [done])
        return loss
    
    def get_action(self, state):
        """根据当前状态选择动作"""
        self.training_steps += 1
        
        # 动态调整探索率
        self._dynamic_epsilon_adjustment()
        
        # epsilon-greedy策略
        if self.random_exploration and random.random() < self.epsilon:
            # 安全的随机探索
            move = self._safe_random_action(state)
        else:
            # 利用模式：使用目标网络进行预测
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            
            # 添加噪声鼓励探索（即使在利用模式）
            if random.random() < 0.1:  # 10%的概率添加噪声
                with torch.no_grad():
                    prediction = self.model(state_tensor)
                    noise = torch.randn_like(prediction) * 0.1
                    prediction = prediction + noise
                move = torch.argmax(prediction).item()
            else:
                with torch.no_grad():
                    prediction = self.target_model(state_tensor)
                move = torch.argmax(prediction).item()
        
        # 创建动作向量
        action = [0, 0, 0]
        action[move] = 1
        
        return action
    
    def _dynamic_epsilon_adjustment(self):
        """动态调整探索率"""
        # 基于训练进度调整
        progress = min(1.0, self.n_games / 200.0)  # 假设200局为完整训练
        
        if self.n_games < 50:
            # 前50局：保持高探索率
            self.epsilon = max(self.epsilon_min, 0.9)
        elif self.n_games < 200:
            # 50-200局：线性衰减
            self.epsilon = max(self.epsilon_min, 0.9 - 0.0045 * (self.n_games - 50))
        else:
            # 200局后：指数衰减
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _safe_random_action(self, state):
        """安全的随机动作选择"""
        # 解析状态中的危险信息（前6个是危险信息）
        dangers = state[:6] if len(state) >= 6 else [0, 0, 0, 0, 0, 0]
        
        # 找出安全的方向
        safe_moves = []
        for i in range(3):  # 3个动作：直行、右转、左转
            # 检查动作是否危险
            if i == 0 and dangers[0] == 0:  # 直行安全
                safe_moves.append(i)
            elif i == 1 and dangers[2] == 0:  # 右转安全
                safe_moves.append(i)
            elif i == 2 and dangers[1] == 0:  # 左转安全
                safe_moves.append(i)
        
        # 如果有安全的方向，优先选择
        if safe_moves:
            return random.choice(safe_moves)
        else:
            # 没有安全方向，随机选择
            return random.randint(0, 2)
    
    def update_epsilon(self):
        """更新探索率（传统方法）"""
        if self.random_exploration:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_records(self, score):
        """更新训练记录"""
        self.n_games += 1
        self.scores.append(score)
        self.total_score += score
        
        if score > self.record:
            self.record = score
            print(f"🎉 新记录! 分数: {score}")
            # 保存最佳模型
            self.save_model(f"best_model_{score}.pth")
        
        # 计算平均分数（最近100局）
        recent_scores = self.scores[-100:] if len(self.scores) >= 100 else self.scores
        mean_score = sum(recent_scores) / len(recent_scores)
        self.mean_scores.append(mean_score)
        
        return mean_score
    
    def save_model(self, file_name=None):
        """保存模型"""
        if file_name is None:
            file_name = f"model_{self.n_games}_{self.record}.pth"
        self.model.save(file_name)
    
    def load_model(self, file_name='model.pth'):
        """加载模型"""
        return self.model.load(file_name)
    
    def plot_training(self):
        """绘制训练过程图表"""
        if len(self.scores) < 10:
            print("📊 训练数据不足，无法绘制图表")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 绘制分数
        plt.subplot(2, 2, 1)
        plt.plot(self.scores, label='分数', alpha=0.7)
        plt.plot(self.mean_scores, label='平均分数', linewidth=2, color='orange')
        plt.xlabel('游戏次数')
        plt.ylabel('分数')
        plt.title(f'训练分数 (最高: {self.record})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 绘制探索率
        plt.subplot(2, 2, 2)
        epsilon_values = []
        current_epsilon = 1.0
        for i in range(len(self.scores)):
            if i < 50:
                current_epsilon = max(self.epsilon_min, 0.9)
            elif i < 200:
                current_epsilon = max(self.epsilon_min, 0.9 - 0.0045 * (i - 50))
            else:
                current_epsilon = max(self.epsilon_min, current_epsilon * self.epsilon_decay)
            epsilon_values.append(current_epsilon)
        
        plt.plot(epsilon_values[:len(self.scores)])
        plt.xlabel('游戏次数')
        plt.ylabel('探索率')
        plt.title('探索率变化')
        plt.grid(True, alpha=0.3)
        
        # 绘制损失
        plt.subplot(2, 2, 3)
        if self.trainer.losses:
            # 计算移动平均损失
            window_size = min(100, len(self.trainer.losses))
            losses = self.trainer.losses
            moving_avg = []
            for i in range(len(losses)):
                if i < window_size:
                    moving_avg.append(sum(losses[:i+1]) / (i+1))
                else:
                    moving_avg.append(sum(losses[i-window_size+1:i+1]) / window_size)
            
            plt.plot(losses, alpha=0.3, label='损失')
            plt.plot(moving_avg, label=f'{window_size}次移动平均', linewidth=2, color='red')
            plt.xlabel('训练步数')
            plt.ylabel('损失')
            plt.title('训练损失')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 绘制分数分布
        plt.subplot(2, 2, 4)
        if len(self.scores) > 10:
            hist, bins = np.histogram(self.scores, bins=min(20, len(set(self.scores))))
            plt.bar(bins[:-1], hist, width=(bins[1]-bins[0])*0.8, alpha=0.7, color='green')
            plt.xlabel('分数')
            plt.ylabel('频次')
            plt.title('分数分布')
            plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 保存图表
        save_path = './records/training_plot.png'
        plt.savefig(save_path, dpi=150)
        print(f"📊 训练图表已保存: {save_path}")
        
        # 显示图表（非阻塞）
        plt.show(block=False)
        plt.pause(0.1)
    
    def save_records(self):
        """保存训练记录"""
        record_data = {
            'n_games': self.n_games,
            'scores': self.scores,
            'mean_scores': self.mean_scores,
            'record': self.record,
            'losses': self.trainer.losses
        }
        
        import pickle
        with open('./records/training_records.pkl', 'wb') as f:
            pickle.dump(record_data, f)
        
        print("💾 训练记录已保存")