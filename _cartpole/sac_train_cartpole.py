import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from crl_lib.sac_cartpole import SAC  
from datetime import datetime
import time
import logging
import sys
import os

# Environment setup
REPETITIONS = 10
ENVIRONMENT = "CartPole-v1"

# CARTPOLE SETTINGS
FEATURES = 4
NUM_VEHICLES = 1
ACTION_VARIABLES = 2
OBS_VARIABLES = FEATURES
ENV_LABEL = "cartpole"

RESULTS_FILE = ""

######################## SAC PARAMETERS

# SAC specific hyperparameters
SAC_Q_LR = 1e-3  # da 3e-4 a 1e-3
SAC_POLICY_LR = 1e-3  # da 3e-4 a 1e-3
SAC_TARGET_ENTROPY_SCALE = 0.5  # da 1.0 a 0.5
SAC_AUTOTUNE_ALPHA = True
SAC_ALPHA = 0.2 
SAC_TAU = 0.01  # da 0.005 a 0.01
SAC_DISCOUNT = 0.995  # da 0.99 a 0.995
SAC_BATCH_SIZE = 128  # da 64 a 128
SAC_BUFFER_SIZE = 50000  # da 10000 a 50000
SAC_GRADIENT_STEPS = 2  # da 1 a 2
SAC_LEARNING_STARTS = 500  # da 1000 a 500

# Training parameters
EPISODES = 500
SHOW_EVERY = 10
UPDATE_EVERY = 10
RENDER = False

######################## FUNCTION DEFINITIONS

def clear_logger():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

def get_logger(repetition):
    logger = logging.getLogger()
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("output", "LOGS", ENV_LABEL, "SAC")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"SAC_logs_r{repetition}_{now_str}.log")
    logging.basicConfig(filename=log_file, format='%(asctime)s %(message)s')
    logger.setLevel(logging.DEBUG)
    logger.info("Started")
    return logger

def create_sac_agent():
    agent = SAC(
        state_size=OBS_VARIABLES,
        action_size=ACTION_VARIABLES,
        _q_learning_rate=SAC_Q_LR,
        _p_learning_rate=SAC_POLICY_LR,
        _target_entropy_scale=SAC_TARGET_ENTROPY_SCALE,
        _autotune_alpha=SAC_AUTOTUNE_ALPHA,
        _alpha=SAC_ALPHA,
        _tau=SAC_TAU,
        _discount=SAC_DISCOUNT,
        _batch_size=SAC_BATCH_SIZE,
        _buffer_size=SAC_BUFFER_SIZE,
        _gradient_steps=SAC_GRADIENT_STEPS,
        env_label=ENV_LABEL
    )
    return agent

######################## MAIN PROGRAM

cost_dir = os.path.join("output", "DATA", ENV_LABEL, "_cost")
os.makedirs(cost_dir, exist_ok=True)
with open(os.path.join(cost_dir, "times_SAC.csv"), "w", newline="") as f:
    f.write("rep,time\n")

for i in range(REPETITIONS):
    start_repetition = time.time()
    
    # ENVIRONMENT
    env = gym.make(ENVIRONMENT)
    
    # SAC agent
    sac_agent = create_sac_agent()
    
    ep_rewards = []
    ep_lengths = []
    fail_count = []
    
    clear_logger()
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("output", "DATA", ENV_LABEL, "SAC")
    os.makedirs(results_dir, exist_ok=True)
    RESULTS_FILE = os.path.join(results_dir, f"SAC_r{i+1}_{now_str}.csv")
    with open(RESULTS_FILE, "w", newline="") as f:
        f.write("run,avg,max,min,avg_length,fr\n")

    global_step = 0

    # cycle on EPISODES
    for episode in range(EPISODES):
        logger = get_logger(i + 1)
        episode_reward = 0
        step_count = 0

        # RESET
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
            
        done = False
        fail = 0

        while not done:
            if episode % SHOW_EVERY == 0 and RENDER:
                env.render()
            
            # Get action
            if len(sac_agent.rb) > SAC_LEARNING_STARTS:
                action = sac_agent.act(state.reshape(1, -1))
            else:
                # Random exploration only during initial phase
                action = np.random.randint(0, ACTION_VARIABLES)

            # STEP
            step_result = env.step(action)
            if len(step_result) == 5:  # New Gym API
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:  # Old Gym API
                next_state, reward, done, info = step_result

            # Reward calculation
            if not done:
                reward = 1.0
            else:
                # Penalize for early termination
                if step_count < 195:
                    reward = -10.0
                    fail = 1
                else:
                    reward = 10.0
                    fail = 0

            # Store experience in SAC replay buffer
            sac_agent.memorize(state, action, reward, next_state, done)
            
            # Train SAC if enough experiences collected
            global_step += 1
            if global_step > SAC_LEARNING_STARTS and len(sac_agent.rb) >= SAC_BATCH_SIZE:
                for _ in range(SAC_GRADIENT_STEPS):
                    sac_agent.replay()
                
                # Update target networks
                sac_agent.update_target()

            logger.info(f"{state.tolist()}#{action}#{reward}")
            episode_reward += reward
            step_count += 1
            state = next_state

        ep_rewards.append(episode_reward)
        ep_lengths.append(step_count)
        fail_count.append(fail)
        
        if not episode % UPDATE_EVERY:
            avg_reward = sum(ep_rewards[-UPDATE_EVERY:]) / UPDATE_EVERY
            avg_length = sum(ep_lengths[-UPDATE_EVERY:]) / UPDATE_EVERY
            avg_fails = sum(fail_count[-UPDATE_EVERY:]) / UPDATE_EVERY
            max_reward = max(ep_rewards[-UPDATE_EVERY:])
            min_reward = min(ep_rewards[-UPDATE_EVERY:])
            
            print(f'Episode: {episode:>5d}, avg_reward: {avg_reward:>6.1f}, avg_length: {avg_length:>6.1f}, failure_rate: {avg_fails:>4.1f}, max_reward: {max_reward:>6.1f}')
            
            with open(RESULTS_FILE, "a", newline="") as f:
                f.write(f"{episode},{avg_reward},{max_reward},{min_reward},{avg_length},{avg_fails}\n")

            # Save model periodicamente
            try:
                model_filename = f"{now_str}_r{i+1}_ep{episode}.pth"
                sac_agent.save_model(model_filename)
            except Exception as e:
                print(f"Error saving model at episode {episode}: {e}")

    # Save final model for this repetition
    try:
        final_model_filename = f"{now_str}_r{i+1}_final.pth"
        sac_agent.save_model(final_model_filename)
        print(f"Final model for repetition {i+1} saved")
    except Exception as e:
        print(f"Error saving final model for repetition {i+1}: {e}")

    end_repetition = time.time()
    elapsed = end_repetition - start_repetition
    with open(os.path.join(cost_dir, "times_SAC.csv"), "a", newline="") as f:
        f.write(f"{i},{elapsed}\n")

    env.close()

print("Training completed!")