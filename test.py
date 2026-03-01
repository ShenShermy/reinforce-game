"""
train.py - 贪吃蛇AI训练程序（增强版）
"""
import pygame
import sys
import numpy as np
from SnackGame import SnakeGame
from snakeai import Agent

def train():
    """改进的训练函数"""
    # 训练参数
    train_episodes = 200  # 训练总次数
    
    print("=" * 60)
    print("贪吃蛇强化学习AI训练 - 增强版")
    print("=" * 60)
    
    # 创建游戏实例（调整为更容易学习的设置）
    game = SnakeGame(
        width=400,  # 更小的窗口，更容易学习
        height=300,
        grid_size=20,
        speed=15,  # 中等速度
        ai_mode=True
    )
    
    # 创建AI智能体
    agent = Agent(state_size=game.state_size, hidden_size=128)
    
    # 尝试加载已有模型
    load_success = agent.load_model()
    
    print(f"\n训练配置:")
    print(f"  游戏窗口: {game.width}x{game.height}")
    print(f"  网格大小: {game.grid_size}")
    print(f"  游戏速度: {game.speed}")
    print(f"  状态大小: {game.state_size}")
    print(f"  训练次数: {train_episodes}")
    print(f"  探索率: {agent.epsilon:.4f}")
    print(f"  当前记录: {agent.record}")
    print("-" * 60)
    
    # 训练统计
    best_score = 0
    consec_fails = 0  # 连续失败计数
    
    try:
        for episode in range(1, train_episodes + 1):
            # 重置游戏
            game_over = False
            game.reset_game()
            
            # 获取初始状态
            state_old = agent.get_state(game)
            
            # 每局游戏的统计
            episode_steps = 0
            episode_reward = 0
            
            # 是否显示游戏画面
            display_game = (episode % 50 == 0) or (episode <= 10)
            
            # 游戏循环
            while not game_over:
                # 获取动作
                action = agent.get_action(state_old)
                
                # 执行动作
                reward, game_over, score = game.play_step(action)
                episode_reward += reward
                episode_steps += 1
                
                # 获取新状态
                state_new = agent.get_state(game)
                
                # 短期记忆训练
                agent.train_short_memory(state_old, action, reward, state_new, game_over)
                
                # 记住经验
                agent.remember(state_old, action, reward, state_new, game_over)
                
                # 更新状态
                state_old = state_new
                
                # 显示游戏画面
                if display_game:
                    game.draw()
                    game.update_display()
                    
                    # 处理事件
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                print("\n训练被用户中断")
                                agent.save_model(f"model_interrupted_{episode}.pth")
                                agent.plot_training()
                                agent.save_records()
                                pygame.quit()
                                sys.exit()
            
            # 游戏结束，进行长期记忆训练
            loss = agent.train_long_memory()
            
            # 更新记录
            mean_score = agent.update_records(score)
            
            # 记录这局的表现
            agent.avg_rewards.append(episode_reward)
            
            # 更新探索率
            agent.update_epsilon()
            
            # 显示训练信息
            if episode % 1 == 0:
                print(f"局数: {episode:4d}, 分数: {score:3d}, 步数: {episode_steps:4d}, "
                      f"平均分: {mean_score:6.2f}, 探索率: {agent.epsilon:.4f}, "
                      f"奖励: {episode_reward:6.2f}, 损失: {loss if loss else 'N/A'}")
            
            # 适应性训练调整
            if score < 2:
                consec_fails += 1
                if consec_fails >= 10:
                    # 连续10局得分低于2，调整策略
                    print(" 检测到学习困难，增加探索...")
                    agent.epsilon = min(0.8, agent.epsilon + 0.1)
                    consec_fails = 0
            else:
                consec_fails = 0
            
            # 阶段性评估
            if episode % 100 == 0:
                print("\n" + "-" * 60)
                print(f"阶段评估 (第{episode}局):")
                print(f"  最高分: {agent.record}")
                print(f"  最近100局平均分: {mean_score:.2f}")
                recent_rewards = agent.avg_rewards[-100:] if len(agent.avg_rewards) >= 100 else agent.avg_rewards
                print(f"  平均每局奖励: {np.mean(recent_rewards):.2f}")
                print(f"  当前探索率: {agent.epsilon:.4f}")
                print("-" * 60)
                
                # 保存模型
                agent.save_model(f"model_epoch_{episode}.pth")
                
                # 绘制训练图表
                agent.plot_training()
                
                # 如果表现良好，减慢探索率衰减
                if mean_score > 5:
                    agent.epsilon_decay = 0.999  # 减慢衰减
            
            # 如果表现太差，重启训练
            if episode > 200 and agent.record < 3:
                print(" 训练效果不佳，重置模型重新训练...")
                agent = Agent(state_size=game.state_size, hidden_size=128)
                agent.epsilon = 0.9  # 重置探索率
                episode = 0  # 重置计数（会在循环末尾+1）
                continue
    
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 保存最终结果
        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"总游戏次数: {agent.n_games}")
        print(f"最高记录: {agent.record}")
        if agent.n_games > 0:
            print(f"最终平均分数: {agent.total_score / agent.n_games:.2f}")
        
        agent.save_model("model_final.pth")
        agent.plot_training()
        agent.save_records()
        
        pygame.quit()
        print("程序已退出")

def test_trained_model(model_path='model_final.pth'):
    """测试训练好的模型"""
    print("测试训练好的模型...")
    
    # 创建游戏实例
    game = SnakeGame(width=600, height=400, grid_size=20, speed=10, ai_mode=True)
    
    # 创建AI智能体
    agent = Agent(state_size=game.state_size, hidden_size=128)
    
    # 加载模型
    if agent.load_model(model_path):
        # 关闭随机探索
        agent.random_exploration = False
        agent.epsilon = agent.epsilon_min
        
        # 测试多次游戏
        test_games = 5
        test_scores = []
        
        print(f"\n测试 {test_games} 次游戏...")
        print("-" * 40)
        
        for i in range(test_games):
            game_over = False
            game.reset_game()
            state_old = agent.get_state(game)
            
            steps = 0
            while not game_over and steps < 500:  # 限制最大步数
                # 获取动作
                action = agent.get_action(state_old)
                
                # 执行动作
                reward, game_over, score = game.play_step(action)
                
                # 获取新状态
                state_new = agent.get_state(game)
                state_old = state_new
                
                steps += 1
                
                # 显示游戏画面
                game.draw()
                game.update_display()
                
                # 处理退出事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            sys.exit()
            
            test_scores.append(score)
            print(f"测试 {i+1}: 分数 = {score}, 步数 = {steps}")
        
        # 计算统计信息
        print("-" * 40)
        print(f"测试完成!")
        print(f"平均分数: {np.mean(test_scores):.2f}")
        print(f"最高分数: {np.max(test_scores)}")
        print(f"最低分数: {np.min(test_scores)}")
        print(f"标准差: {np.std(test_scores):.2f}")
    
    pygame.quit()

if __name__ == "__main__":
    # 检查命令行参数
    import argparse
    
    parser = argparse.ArgumentParser(description='贪吃蛇AI训练')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test'], help='运行模式: train或test')
    parser.add_argument('--model', type=str, default='model_final.pth',
                       help='要测试的模型路径')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='训练次数')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_episodes = args.episodes
        train()
    elif args.mode == 'test':
        test_trained_model(args.model)