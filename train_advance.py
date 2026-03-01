"""
train_advanced.py - 高级训练策略
包含课程学习、多阶段训练等
"""
import numpy as np
import pygame
import sys
from SnackGame import SnakeGame
from snakeai import Agent

class CurriculumLearning:
    def __init__(self):
        self.stages = [
            # 阶段1: 超简单（小地图，慢速）
            {'width': 300, 'height': 200, 'grid': 25, 'speed': 5, 'episodes': 100},
            # 阶段2: 简单（中等地图，中速）
            {'width': 400, 'height': 300, 'grid': 20, 'speed': 8, 'episodes': 150},
            # 阶段3: 中等（标准地图，标准速度）
            {'width': 500, 'height': 400, 'grid': 20, 'speed': 12, 'episodes': 200},
            # 阶段4: 困难（标准地图，快速）
            {'width': 600, 'height': 400, 'grid': 20, 'speed': 12, 'episodes': 200},
            # 阶段5: 专家（大地图，快速）
            {'width': 800, 'height': 600, 'grid': 20, 'speed': 12, 'episodes': 150},
        ]
        self.current_stage = 0
    
    def get_stage_config(self):
        """获取当前阶段配置"""
        return self.stages[self.current_stage]
    
    def should_advance(self, recent_scores, min_games=30):
        """检查是否应该进入下一阶段"""
        if len(recent_scores) < min_games:
            return False
        
        # 阶段特定目标
        stage_targets = [3, 5, 8, 12, 15]
        target = stage_targets[self.current_stage]
        
        # 计算最近N局的平均分
        recent_avg = np.mean(recent_scores[-min_games:])
        
        return recent_avg >= target
    
    def advance(self):
        """进入下一阶段"""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            return True
        return False
    
    def is_final_stage(self):
        """检查是否是最后阶段"""
        return self.current_stage == len(self.stages) - 1

def train_with_curriculum():
    """使用课程学习进行训练"""
    curriculum = CurriculumLearning()
    agent = None
    
    print("=" * 60)
    print("贪吃蛇AI课程学习训练")
    print("=" * 60)
    
    while curriculum.current_stage < len(curriculum.stages):
        stage = curriculum.stages[curriculum.current_stage]
        
        print(f"\n{'='*60}")
        print(f"阶段 {curriculum.current_stage + 1}/{len(curriculum.stages)}")
        print(f"配置: {stage['width']}x{stage['height']}, 网格: {stage['grid']}, 速度: {stage['speed']}")
        print(f"目标: {[3, 5, 8, 12, 15][curriculum.current_stage]}分")
        print(f"训练局数: {stage['episodes']}")
        print(f"{'='*60}")
        
        # 创建游戏实例
        game = SnakeGame(
            width=stage['width'],
            height=stage['height'],
            grid_size=stage['grid'],
            speed=stage['speed'],
            ai_mode=True
        )
        
        # 如果是第一阶段，创建新agent；否则继续使用现有agent
        if agent is None:
            agent = Agent(state_size=game.state_size, hidden_size=128)
            print(f"创建新Agent，状态大小: {game.state_size}")
        else:
            # 调整agent的状态大小
            if agent.state_size != game.state_size:
                print(f"调整Agent状态大小: {agent.state_size} -> {game.state_size}")
                # 重新创建Agent，但可以尝试加载之前阶段的模型
                old_record = agent.record
                agent = Agent(state_size=game.state_size, hidden_size=128)
                agent.record = old_record
        
        # 调整探索率（新阶段开始时增加探索）
        if curriculum.current_stage == 0:
            agent.epsilon = 0.9  # 第一阶段高探索
        else:
            agent.epsilon = max(agent.epsilon_min, 0.6)  # 后续阶段适当探索
        
        # 阶段训练
        stage_best = 0
        stage_scores = []
        stage_rewards = []
        
        for episode in range(1, stage['episodes'] + 1):
            # 训练一局
            game_over = False
            game.reset_game()
            state_old = agent.get_state(game)
            
            episode_reward = 0
            episode_steps = 0
            
            while not game_over:
                action = agent.get_action(state_old)
                reward, game_over, score = game.play_step(action)
                state_new = agent.get_state(game)
                
                # 短期训练
                agent.train_short_memory(state_old, action, reward, state_new, game_over)
                agent.remember(state_old, action, reward, state_new, game_over)
                state_old = state_new
                
                episode_reward += reward
                episode_steps += 1
            
            # 长期训练
            agent.train_long_memory()
            agent.update_records(score)
            agent.update_epsilon()
            
            stage_scores.append(score)
            stage_rewards.append(episode_reward)
            stage_best = max(stage_best, score)
            
            # 每10局显示一次进度
            if episode % 10 == 0:
                recent_avg = np.mean(stage_scores[-10:]) if len(stage_scores) >= 10 else np.mean(stage_scores)
                print(f"阶段{curriculum.current_stage+1} 局数:{episode:3d}/{stage['episodes']}, "
                      f"分数:{score:3d}, 阶段最佳:{stage_best:3d}, 最近10局平均:{recent_avg:.1f}")
            
            # 检查是否可以提前进入下一阶段
            if episode >= 30 and curriculum.should_advance(stage_scores, min_games=20):
                print(f"✓ 阶段{curriculum.current_stage+1} 提前完成! 平均分达到目标")
                break
        
        # 保存阶段模型
        agent.save_model(f"model_stage_{curriculum.current_stage+1}.pth")
        
        # 绘制当前阶段训练图
        if len(stage_scores) >= 10:
            agent.plot_training()
        
        # 检查是否进入下一阶段
        if curriculum.is_final_stage():
            print(f"\n已达到最终阶段，课程学习完成!")
            break
        elif curriculum.advance():
            print(f"\n 进入下一阶段: {curriculum.current_stage + 1}")
        else:
            print(f"\n 无法进入下一阶段，保持当前阶段")
        
        # 清除游戏对象
        del game
        pygame.quit()
        pygame.init()
    
    print("\n" + "="*60)
    print("课程学习完成!")
    print(f"最终最佳分数: {agent.record}")
    agent.save_model("model_curriculum_final.pth")
    agent.plot_training()
    
    pygame.quit()

def quick_train():
    """快速训练（用于调试）"""
    print("快速训练模式（用于调试）")
    
    # 创建超简单环境
    game = SnakeGame(
        width=200,
        height=200,
        grid_size=25,
        speed=5,
        ai_mode=True
    )
    
    # 创建简单Agent
    agent = Agent(state_size=game.state_size, hidden_size=64)
    
    # 快速训练
    for episode in range(1, 101):
        game_over = False
        game.reset_game()
        state_old = agent.get_state(game)
        
        while not game_over:
            action = agent.get_action(state_old)
            reward, game_over, score = game.play_step(action)
            state_new = agent.get_state(game)
            
            agent.train_short_memory(state_old, action, reward, state_new, game_over)
            agent.remember(state_old, action, reward, state_new, game_over)
            state_old = state_new
        
        agent.train_long_memory()
        agent.update_records(score)
        agent.update_epsilon()
        
        if episode % 10 == 0:
            print(f"局数:{episode:3d}, 分数:{score:2d}, 探索率:{agent.epsilon:.3f}")
    
    print(f"快速训练完成，最佳分数: {agent.record}")
    agent.save_model("model_quick.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='贪吃蛇AI高级训练')
    parser.add_argument('--mode', choices=['curriculum', 'quick', 'test'], default='curriculum')
    parser.add_argument('--model', type=str, default='model_final.pth')
    
    args = parser.parse_args()
    
    if args.mode == 'curriculum':
        train_with_curriculum()
    elif args.mode == 'quick':
        quick_train()
    else:
        # 测试模式
        from train import test_trained_model
        test_trained_model(args.model)