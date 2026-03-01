"""
play_ai.py - 使用训练好的模型玩贪吃蛇游戏
"""
import pygame
import sys
import os
import numpy as np
from SnackGame import SnakeGame
from snakeai import Agent

def play_ai(model_name='model_curriculum_final', num_games=1, speed=10, show_info=True):
    """
    使用AI模型玩贪吃蛇游戏
    
    参数:
        model_name: 要使用的模型文件名（在models文件夹中）
        num_games: 要玩的游戏局数
        speed: 游戏速度（1-20，数字越大速度越快）
        show_info: 是否显示游戏信息
    """
    print("=" * 60)
    print("贪吃蛇游戏 - AI模式")
    print("=" * 60)
    
    # 创建游戏实例
    game = SnakeGame(
        width=800,
        height=600,
        grid_size=20,
        speed=speed,
        ai_mode=True
    )
    
    # 创建AI智能体
    agent = Agent(state_size=game.state_size, hidden_size=128)
    
    # 加载模型
    print(f"\n正在加载模型: {model_name}")
    model_path = os.path.join('./models/test', model_name)
    
    if not os.path.exists(model_path):
        print(f" 模型文件不存在: {model_path}")
        print("\n可用的模型文件:")
        if os.path.exists('./models/test'):
            models = [f for f in os.listdir('./models/test') if f.endswith('.pth')]
            if models:
                for i, m in enumerate(models, 1):
                    print(f"  {i}. {m}")
            else:
                print("  (没有找到任何模型文件)")
        return
    
    # 尝试加载模型
    success = agent.load_model(model_name)
    if not success:
        print(" 模型加载失败")
        return
    
    # 设置为评估模式（不进行梯度计算）
    agent.model.eval()
    agent.target_model.eval()
    agent.epsilon = 0  # 关闭探索，只使用贪心策略
    
    print(f"\n 模型加载成功")
    print(f"游戏配置:")
    print(f"  窗口大小: {game.width}x{game.height}")
    print(f"  网格大小: {game.grid_size}")
    print(f"  游戏速度: {speed}")
    print(f"  游戏局数: {num_games}")
    print(f"  状态大小: {game.state_size}")
    print("-" * 60)
    
    # 游戏统计
    all_scores = []
    game_count = 0
    
    try:
        for game_num in range(1, num_games + 1):
            print(f"\n【第 {game_num}/{num_games} 局游戏】")
            
            # 重置游戏
            game.reset_game()
            game_over = False
            
            # 获取初始状态
            state = agent.get_state(game)
            
            # 游戏统计
            steps = 0
            food_eaten = 0
            
            # 游戏循环
            while not game_over:
                # 获取AI动作（使用贪心策略）
                import torch
                state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                with torch.no_grad():
                    prediction = agent.model(state_tensor)
                action_idx = torch.argmax(prediction).item()
                
                # 转换为动作向量
                action = [0, 0, 0]
                action[action_idx] = 1
                
                # 执行动作
                reward, game_over, score = game.play_step(action)
                
                # 更新统计
                steps += 1
                if reward > 0:
                    food_eaten += 1
                
                # 获取新状态
                state = agent.get_state(game)
                
                # 绘制游戏
                game.draw()
                game.update_display()
                
                # 处理事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\n游戏被用户中断")
                        pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            print("\n游戏被用户中断")
                            pygame.quit()
                            return
            
            # 游戏结束统计
            all_scores.append(score)
            game_count += 1
            print(f"  分数: {score}")
            print(f"  步数: {steps}")
            print(f"  吃掉食物: {food_eaten} 个")
            
            # 短暂延迟以便观看游戏结束的情况
            if game_num < num_games:
                pygame.time.wait(1000)
        
        # 显示汇总统计
        print("\n" + "=" * 60)
        print("游戏汇总统计")
        print("=" * 60)
        print(f"总游戏局数: {game_count}")
        print(f"平均分数: {sum(all_scores) / len(all_scores):.1f}")
        print(f"最高分数: {max(all_scores)}")
        print(f"最低分数: {min(all_scores)}")
        print(f"所有分数: {all_scores}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n游戏被用户中断")
    finally:
        pygame.quit()


if __name__ == "__main__":
    import argparse
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='使用AI模型玩贪吃蛇游戏')
    parser.add_argument('--model', type=str, default='model_curriculum_final.pth',
                        help='模型文件名 (默认: model_curriculum_final.pth)')
    parser.add_argument('--games', type=int, default=1,
                        help='游戏局数 (默认: 1)')
    parser.add_argument('--speed', type=int, default=10,
                        help='游戏速度 1-20 (默认: 10)')
    parser.add_argument('--list', action='store_true',
                        help='列出所有可用的模型文件')
    
    args = parser.parse_args()
    
    # 如果指定了 --list，显示可用模型
    if args.list:
        print("可用的模型文件:")
        if os.path.exists('./models'):
            models = sorted([f for f in os.listdir('./models') if f.endswith('.pth')])
            if models:
                for i, m in enumerate(models, 1):
                    print(f"  {i}. {m}")
            else:
                print("  (没有找到任何模型文件)")
        sys.exit(0)
    
    # 运行游戏
    play_ai(
        model_name=args.model,
        num_games=args.games,
        speed=args.speed,
        show_info=True
    )
