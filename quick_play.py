#!/usr/bin/env python3
"""
quick_play.py - 快速开始：一键运行最佳模型
"""

import subprocess
import sys
import os

def main():
    """快速运行最佳模型"""
    print("=" * 60)
    print(" 贪吃蛇游戏 - AI快速演示")
    print("=" * 60)
    print()
    print("选择要执行的操作:")
    print()
    print("  1.   快速演示（用最佳模型玩1局，最快速度）")
    print("  2.   详细观看（用最佳模型玩3局，中等速度）")
    print("  3.  评估性能（对比所有模型）")
    print("  4.  运行最佳模型（玩5局，评估平均分）")
    print("  5.  列出所有模型")
    print("  6.  退出")
    print()
    
    choice = input("请选择 (1-6): ").strip()
    print()
    
    if choice == "1":
        print("启动快速演示...")
        os.system("python play_ai.py --model model_curriculum_final.pth --speed 20")
    
    elif choice == "2":
        print("启动详细观看...")
        os.system("python play_ai.py --model model_curriculum_final.pth --games 3 --speed 8")
    
    elif choice == "3":
        print("开始评估所有模型...")
        os.system("python evaluate_ai.py --compare --games 3")
    
    elif choice == "4":
        print("运行最佳模型评估...")
        os.system("python evaluate_ai.py --model model_curriculum_final.pth --games 5")
    
    elif choice == "5":
        print("可用的模型:")
        os.system("python play_ai.py --list")
    
    elif choice == "6":
        print("再见！")
        return
    
    else:
        print("无效的选择，请重试")
        main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已中断")
