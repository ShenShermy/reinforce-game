"""
play_human.py - 人类玩家模式
"""
import pygame
import sys
from snake   import SnakeGame

def main():
    """人类玩家主函数"""
    print("贪吃蛇游戏 - 人类玩家模式")
    print("=" * 40)
    print("控制方式:")
    print("  方向键: 控制蛇移动")
    print("  空格键: 暂停/继续")
    print("  R键: 重新开始")
    print("  ESC键: 退出游戏")
    print("=" * 40)
    
    # 创建游戏实例（人类玩家模式）
    game = SnakeGame(width=800, height=600, grid_size=20, speed=10, ai_mode=False)
    
    # 游戏主循环
    running = True
    while running:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # 执行一步游戏（无动作参数，由键盘控制）
        reward, game_over, score = game.play_step()
        
        # 绘制游戏
        game.draw()
        
        # 更新显示并控制帧率
        game.update_display()
        
        # 如果游戏结束，等待用户按键
        if game_over:
            # 短暂延迟后检查按键
            pygame.time.wait(500)
            
            # 检查是否按R键重新开始
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                game.reset_game()
            elif keys[pygame.K_ESCAPE]:
                running = False
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()