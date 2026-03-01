"""
train.py - 贪吃蛇AI训练程序
"""
import pygame
import sys
import time
from snake import SnakeGame
from snake_ai import Agent
import matplotlib.pyplot as plt

def train():
    """训练AI智能体"""
    print("=" * 60)
    print("贪吃蛇强化学习AI训练")
    print("=" * 60)
    
    # 训练参数
    train_episodes = 500  # 训练总次数
    display_every = 100    # 每N次显示一次游戏画面
    save_every = 100       # 每N次保存一次模型
    plot_every = 100       # 每N次绘制一次训练图表
    
    # 创建游戏实例（AI模式）
    game = SnakeGame(width=600, height=400, grid_size=20, speed=20, ai_mode=True)
    
    # 创建AI智能体
    agent = Agent(state_size=game.state_size)
    
    # 尝试加载已有模型
    load_success = agent.load_model()
    
    # 训练循环
    print(f"\n开始训练，目标: {train_episodes} 次游戏")
    print(f"当前探索率: {agent.epsilon:.4f}")
    print(f"当前记录: {agent.record}")
    print("-" * 60)
    
    try:
        for episode in range(1, train_episodes + 1):
            # 重置游戏
            game_over = False
            game.reset_game()
            
            # 获取初始状态
            state_old = agent.get_state(game)
            
            # 是否显示游戏画面
            display_game = (episode % display_every == 0) or (episode <= 5)
            
            # 游戏循环
            while not game_over:
                if display_game:
                    # 处理退出事件
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
                
                # 获取动作
                action = agent.get_action(state_old)
                
                # 执行动作
                reward, game_over, score = game.play_step(action)
                
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
            
            # 游戏结束，进行长期记忆训练
            loss = agent.train_long_memory()
            
            # 更新记录
            mean_score = agent.update_records(score)
            
            # 更新探索率
            agent.update_epsilon()
            
            # 显示训练信息
            if episode % 1 == 0:
                print(f"游戏: {episode:4d}, 分数: {score:3d}, 记录: {agent.record:3d}, "
                      f"平均: {mean_score:6.2f}, 探索率: {agent.epsilon:.4f}, "
                      f"损失: {loss if loss else 'N/A'}")
            
            # 定期保存模型
            if episode % save_every == 0:
                agent.save_model(f"model_{episode}_{agent.record}.pth")
            
            # 定期绘制训练图表
            if episode % plot_every == 0 and episode > 10:
                agent.plot_training()
    
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    
    finally:
        # 保存最终模型和记录
        print("\n" + "-" * 60)
        print("训练完成!")
        print(f"总游戏次数: {agent.n_games}")
        print(f"最高记录: {agent.record}")
        print(f"平均分数: {agent.total_score / agent.n_games:.2f}")
        
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
    agent = Agent(state_size=game.state_size)
    
    # 加载模型
    if agent.load_model(model_path):
        # 关闭随机探索
        agent.random_exploration = False
        
        # 测试多次游戏
        test_games = 10
        test_scores = []
        
        print(f"\n测试 {test_games} 次游戏...")
        
        for i in range(test_games):
            game_over = False
            game.reset_game()
            state_old = agent.get_state(game)
            
            while not game_over:
                # 获取动作
                action = agent.get_action(state_old)
                
                # 执行动作
                reward, game_over, score = game.play_step(action)
                
                # 获取新状态
                state_new = agent.get_state(game)
                state_old = state_new
                
                # 显示游戏画面
                game.draw()
                game.update_display()
                
                # 处理退出事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
            
            test_scores.append(score)
            print(f"测试 {i+1}: 分数 = {score}")
        
        # 计算平均分数
        avg_score = sum(test_scores) / len(test_scores)
        print(f"\n平均分数: {avg_score:.2f}")
        print(f"最高分数: {max(test_scores)}")
        print(f"最低分数: {min(test_scores)}")
    
    pygame.quit()

if __name__ == "__main__":
    # 检查命令行参数
    import argparse
    
    parser = argparse.ArgumentParser(description='贪吃蛇AI训练')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test'], help='运行模式: train或test')
    parser.add_argument('--model', type=str, default='model_final.pth',
                       help='要测试的模型路径')
    parser.add_argument('--episodes', type=int, default=500,
                       help='训练次数')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_episodes = args.episodes
        train()
    elif args.mode == 'test':
        test_trained_model(args.model)