"""
evaluate_ai.py - 评估AI模型性能
用于快速评估模型分数，不显示游戏画面
"""
import os
import sys
import numpy as np
import torch
from SnackGame import SnakeGame
from snakeai import Agent

def evaluate_model(model_name='model_final.pth', num_games=10, show_progress=True):
    """
    评估模型性能
    
    参数:
        model_name: 要评估的模型文件名
        num_games: 评估的游戏局数
        show_progress: 是否显示进度信息
    
    返回:
        stats: 包含统计信息的字典
    """
    # 创建游戏实例（不显示）
    game = SnakeGame(
        width=400,
        height=300,
        grid_size=20,
        speed=100,  # 快速执行，不用渲染
        ai_mode=True
    )
    
    # 创建AI智能体
    agent = Agent(state_size=game.state_size, hidden_size=128)
    
    # 加载模型
    model_path = os.path.join('./models', model_name)
    if not os.path.exists(model_path):
        print(f" 模型文件不存在: {model_path}")
        return None
    
    success = agent.load_model(model_name)
    if not success:
        print(f" 模型加载失败")
        return None
    
    # 设置为评估模式
    agent.model.eval()
    agent.target_model.eval()
    agent.epsilon = 0
    
    if show_progress:
        print(f"\n 正在评估模型: {model_name}")
        print(f" 评估次数: {num_games}")
        print("-" * 50)
    
    # 评估
    scores = []
    steps_list = []
    foods_eaten_list = []
    
    try:
        for game_num in range(1, num_games + 1):
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
                # 获取AI动作
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
            
            # 记录统计
            scores.append(score)
            steps_list.append(steps)
            foods_eaten_list.append(food_eaten)
            
            if show_progress:
                status = "▓" if food_eaten > 0 else "░"
                print(f"  游戏 {game_num:2d}/{num_games:2d} {status} 分数: {score:3d} | 步数: {steps:4d} | 食物: {food_eaten:2d}")
        
    except KeyboardInterrupt:
        print("\n  评估被中断")
        if not scores:
            return None
    
    # 计算统计
    stats = {
        'model': model_name,
        'num_games': len(scores),
        'scores': scores,
        'steps': steps_list,
        'foods_eaten': foods_eaten_list,
        'avg_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': np.max(scores),
        'min_score': np.min(scores),
        'avg_steps': np.mean(steps_list),
        'avg_food': np.mean(foods_eaten_list),
    }
    
    if show_progress:
        print("-" * 50)
        print(f"\n评估结果:")
        print(f"  平均分数: {stats['avg_score']:.2f} ± {stats['std_score']:.2f}")
        print(f"  最高分数: {stats['max_score']}")
        print(f"  最低分数: {stats['min_score']}")
        print(f"  平均步数: {stats['avg_steps']:.1f}")
        print(f"  平均食物: {stats['avg_food']:.2f}")
    
    return stats


def compare_models(model_names=None, num_games=5):
    """
    对比多个模型的性能
    
    参数:
        model_names: 模型文件名列表。如果为None，则评估所有模型
        num_games: 每个模型的评估次数
    """
    if model_names is None:
        # 获取所有可用的模型
        if not os.path.exists('./models'):
            print(" models文件夹不存在")
            return
        model_names = sorted([f for f in os.listdir('./models') if f.endswith('.pth')])
    
    if not model_names:
        print(" 没有找到任何模型文件")
        return
    
    print("=" * 60)
    print(" 模型对比评估")
    print("=" * 60)
    
    results = []
    
    for model_name in model_names:
        stats = evaluate_model(model_name, num_games=num_games)
        if stats:
            results.append(stats)
    
    if not results:
        print(" 没有模型评估成功")
        return
    
    # 显示对比表格
    print("\n" + "=" * 80)
    print(" 模型对比汇总")
    print("=" * 80)
    print(f"{'模型名称':<35} {'平均分':<12} {'最高分':<10} {'平均食物':<10}")
    print("-" * 80)
    
    for stats in sorted(results, key=lambda x: x['avg_score'], reverse=True):
        model_name = stats['model'][:33] + ".." if len(stats['model']) > 35 else stats['model']
        print(f"{model_name:<35} {stats['avg_score']:>10.2f}  {stats['max_score']:>8d}  {stats['avg_food']:>8.2f}")
    
    print("=" * 80)
    
    # 找到最佳模型
    best_model = max(results, key=lambda x: x['avg_score'])
    print(f"\n 最佳模型: {best_model['model']} (平均分: {best_model['avg_score']:.2f})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='评估AI模型性能')
    parser.add_argument('--model', type=str, default=None,
                        help='要评估的模型文件名 (默认: 评估所有模型)')
    parser.add_argument('--games', type=int, default=5,
                        help='每个模型的评估次数 (默认: 5)')
    parser.add_argument('--compare', action='store_true',
                        help='对比所有模型')
    
    args = parser.parse_args()
    
    if args.compare or args.model is None:
        # 对比所有模型
        compare_models(num_games=args.games)
    else:
        # 评估单个模型
        evaluate_model(args.model, num_games=args.games)
