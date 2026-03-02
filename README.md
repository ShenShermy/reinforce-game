**QlearningSnakeGame**

![Desktop 2026 03 02 - 18 59 47 02 DVR](https://github.com/user-attachments/assets/70636e1d-a2d3-4d2e-b983-0cbc4e943c00)


This project implements a classic Snake game with an AI agent trained using Deep Q-Learning (DQN) algorithm. The game supports both human player mode and AI training mode, allowing for comparison between human and AI performance.

reinforce-game/

├── SnackGame.py              # Game engine core (Pygame-based)

├── snakeai.py                # Neural network model and trainer

├── play_ai.py                # AI demo script

├── evaluate_ai.py            # Performance evaluation script

├── train_advance.py          # Advanced training script

├── quick_play.py             # Quick demo menu 

├── models/                   # Trained model storage

│   ├── best_model_1.pth      # Best models from training stages

│   ├── model_stage_1-5.pth   # Models from curriculum stages

│   └── model_curriculum_final.pth  # Final optimal model

└── requirements.txt          # Project dependencies
**Project Achievements**
This project successfully demonstrates how to use deep reinforcement learning to solve game AI problems. By combining DQN algorithms, curriculum learning, and layer normalization, we have trained a snake AI with stable performance and intelligent strategies.

**Practical Application Value**
While the snake game is simple, the technologies and methodologies used in this project can be extended to more complex problems:
- Decision systems in autonomous driving
- Robot control and navigation
- Game AI and intelligent NPCs
- Resource optimization and path planning

**Future Improvement Directions**
1. Adopt more advanced RL algorithms (e.g., A3C, PPO)
2. Use CNN to process game images instead of just feature vectors
3. Multi-agent adversarial learning
4. Transfer learning across different games
5. Real-time performance optimization and model compression

**Final Reflection**
This project fully demonstrates the application potential of artificial intelligence in gaming. Through continuous innovation and improvement, we can develop AI systems that are more intelligent and better adapted to complex environments.
