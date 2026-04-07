from TicTacToe import TicTacToeEnv, train_step, select_action, evaluate_vs_random
import random
import numpy as np

def train_resume(model, target, optimizer, buffer, start_episode=60000, num_episodes=10000):
    """Resume training from a checkpoint.
    
    Args:
        model: The trained QNetwork
        target: The target QNetwork
        optimizer: The optimizer state (or create new)
        buffer: The replay buffer (can be empty or pre-filled)
        start_episode: Episode number to resume from
        num_episodes: How many more episodes to train
    """
    env = TicTacToeEnv()
    
    steps = 0
    epsilon = 1.0
    
    losses = []
    ep_lengths = []
    
    def update_target():
        target.load_state_dict(model.state_dict())
    
    for episode in range(start_episode, start_episode + num_episodes):
        s = env.reset()
        ep_len = 0
        
        if env.mover == -1:
            env.board[random.choice(env.available_actions())] = -1
            s = env.board.copy()
        
        done = False
        while not done:
            avail = env.available_actions()
            epsilon = max(0.01, 1.0 - steps / 40000) if episode < 50000 else 0.0
            
            if episode > 30000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-4
            if episode > 50000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-6
            
            a = select_action(model, s, avail, epsilon)
            ns, r, done = env.step(a)
            next_avail = env.available_actions() if not done else []
            
            buffer.push(s, a, r, ns, done, next_avail)
            s = ns
            steps += 1
            ep_len += 1
            
            stats = train_step(model, target, optimizer, buffer)
            
            if stats is not None:
                losses.append(stats["loss"])
        
        ep_lengths.append(ep_len)
        
        if episode % 500 == 0:
            update_target()
        
        if episode % 2000 == 0 and episode > 0:
            w, d, l = evaluate_vs_random(model)
            
            print(
                f"[Ep {episode:6d}] "
                f"ε={epsilon:.2f} | "
                f"Win={w:.2%} Draw={d:.2%} Loss={l:.2%} | "
                f"AvgLen={np.mean(ep_lengths[-2000:]):.2f} | "
                f"Loss={np.mean(losses[-5000:]):.4f}"
            )
    
    return model