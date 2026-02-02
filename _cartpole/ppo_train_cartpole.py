import gym
from crl_lib.ppo import PPO 

from datetime import datetime
import numpy as np
import time
import logging
import sys
import os

REPETITIONS = 10
ENVIRONMENT = "CartPole-v1"
STATE_DIM = 4  # (position, velocity, angle, angular_velocity)
ACTION_DIM = 2  # (left, right)
ENV_LABEL = "cartpole"

EPISODES = 500
RENDER = False
SHOW_EVERY = 10
UPDATE_EVERY = 10
UPDATE_FREQUENCY = 5 

######################## FUNCTIONS

def clear_logger():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

def get_logger(repetition):
    logger = logging.getLogger()
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("output", "LOGS", ENV_LABEL, "PPO")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"PPO_logs_r{repetition}_{now_str}.log")
    logging.basicConfig(filename=log_file, format='%(asctime)s %(message)s')
    logger.setLevel(logging.DEBUG)
    logger.info("Started")
    return logger

######################## MAIN PROGRAM

cost_dir = os.path.join("output", "DATA", ENV_LABEL, "_cost")
os.makedirs(cost_dir, exist_ok=True)
with open(os.path.join(cost_dir, "times_PPO.csv"), "w", newline="") as f:
    f.write("rep,time\n")

for i in range(REPETITIONS):
    start_repetition = time.time()

    # ENVIRONMENT
    env = gym.make(ENVIRONMENT)

    # PPO AGENT
    ppo_agent = PPO(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    
    ep_rewards = []
    ep_lengths = []
    fail_count = []

    clear_logger()
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("output", "DATA", ENV_LABEL, "PPO")
    os.makedirs(results_dir, exist_ok=True)
    RESULTS_FILE = os.path.join(results_dir, f"PPO_r{i+1}_{now_str}.csv")
    with open(RESULTS_FILE, "w", newline="") as f:
        f.write("run,avg,max,min,avg_length,fr\n")

    logger = get_logger(i + 1)

    # cycle on EPISODES
    for episode in range(EPISODES):
        episode_reward = 0
        episode_length = 0
        
        # RESET
        obs, _ = env.reset()
        state = np.array(obs, dtype=np.float32)

        done = False
        fail = 0

        while not done:
            if episode % SHOW_EVERY == 0 and RENDER:
                env.render()

            # ACTION
            action, log_prob, value, _ = ppo_agent.get_action_and_value(state)

            # STEP
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_state = np.array(next_obs, dtype=np.float32)

            # +1 for each step in the episode in which the pole is balanced
            if not done:
                reward = 1.0
            else:
                # Penalty for falling before the episode length threshold
                if episode_length < 195:  # CartPole-v1 max is 500, consider <195 as failure
                    reward = -10.0
                    fail = 1
                else:
                    reward = 10.0
                    fail = 0
            
            # Check if the episode is a failure
            if done and episode_length < 500: 
                fail = 1

            ppo_agent.memorize(state, action, reward, log_prob, value, done)

            logger.info(f"{state.tolist()}#{action}#{reward}")
            episode_reward += reward
            episode_length += 1
            state = next_state

        if episode % UPDATE_FREQUENCY == 0 and episode > 0:
            ppo_agent.update()

        ep_rewards.append(episode_reward)
        ep_lengths.append(episode_length)
        fail_count.append(fail)

        if episode % UPDATE_EVERY == 0 and episode > 0:
            recent_rewards = ep_rewards[-UPDATE_EVERY:]
            recent_lengths = ep_lengths[-UPDATE_EVERY:]
            recent_fails = fail_count[-UPDATE_EVERY:]
            
            avg_reward = np.mean(recent_rewards)
            max_reward = np.max(recent_rewards)
            min_reward = np.min(recent_rewards)
            avg_length = np.mean(recent_lengths) # Average length of the last UPDATE_EVERY episodes
            failure_rate = np.mean(recent_fails)
            
            print(f'Episode: {episode:>5d}, avg reward: {avg_reward:>6.1f}, '
                  f'avg length: {avg_length:>6.1f}, fail rate: {failure_rate:>4.2f}, '
                  f'max reward: {max_reward:>6.1f}, min reward: {min_reward:>6.1f}')
            
            with open(RESULTS_FILE, "a", newline="") as f:
                f.write(f"{episode},{avg_reward},{max_reward},{min_reward},{avg_length},{failure_rate}\n")

    # Update the agent one last time after all episodes
    if len(ppo_agent.buffer) > 0:
        ppo_agent.update()

    # SAVE MODEL
    model_filename = f"{now_str}_r{i+1}_ep{episode}.pth"
    ppo_agent.save_model(model_filename, env_label=ENV_LABEL)

    # Final statistics
    final_avg_reward = np.mean(ep_rewards[-50:])
    final_avg_length = np.mean(ep_lengths[-50:])
    final_fail_rate = np.mean(fail_count[-50:])
    
    print(f"\nRepetition {i+1} completed:")
    print(f"  Final average reward (last 50 episodes): {final_avg_reward:.2f}")
    print(f"  Final average length (last 50 episodes): {final_avg_length:.2f}")
    print(f"  Final fail rate (last 50 episodes): {final_fail_rate:.2f}")
    print("-" * 60)

    end_repetition = time.time()
    elapsed = end_repetition - start_repetition
    with open(os.path.join(cost_dir, "times_PPO.csv"), "a", newline="") as f:
        f.write(f"{i+1},{elapsed}\n")

    env.close()

print(f"\nAll {REPETITIONS} repetitions completed!")