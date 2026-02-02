import gymnasium
import highway_env


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  

from crl_lib.qtable import QTable
from crl_lib.dqn import DQN

from datetime import datetime
import numpy as np
import time
import logging
import sys 

REPETITIONS = 10
ENVIRONMENT = "intersectionMA-v0"  # highwayMA, intersectionMA


if ENVIRONMENT == "highwayMA-v0":
    FEATURES = 5
    NUM_VEHICLES = 2
    ACTION_VARIABLES = 5
    ENV_LABEL = "highway"
    EGO_MODEL = os.path.join("models", "EGO", "highway", "table_EGO_TRAIN_2000.txt")

elif ENVIRONMENT == "intersectionMA-v0":
    FEATURES = 7
    NUM_VEHICLES = 2
    ACTION_VARIABLES = 3
    ENV_LABEL = "intersection"
    EGO_MODEL = os.path.join("models", "EGO", "intersection", "table_EGO_TRAIN_8000.txt")

OBS_VARIABLES = NUM_VEHICLES * FEATURES

EPISODES = 500 # episode budget   


RENDER = False
SHOW_EVERY = 10  # How often the current solution is rendered
UPDATE_EVERY = 10  # How often the current progress is recorded

# Exploration settings
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 3
EPSILON_MIN = 0.1
EPSILON_DECAY_VALUE = START_EPSILON_DECAYING / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# DQN specific parameters
LEARNING_RATE = 0.001
DISCOUNT = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 10000
GRADIENT_STEPS = 1
TARGET_UPDATE_FREQUENCY = 50  # Update target network every 50 episodes

########################    FUNCTIONS

def clear_logger():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

def get_logger(repetition):
    logger = logging.getLogger()
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("output", "LOGS", ENV_LABEL, "DQN")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"DQN_logs_r{repetition}_{now_str}.log")
    logging.basicConfig(filename=log_file, format='%(asctime)s %(message)s')
    logger.setLevel(logging.DEBUG)
    logger.info("Started")
    return logger

########################    MAIN PROGRAM

cost_dir = os.path.join("output", "DATA", ENV_LABEL, "_cost")
os.makedirs(cost_dir, exist_ok=True)
with open(os.path.join(cost_dir, "times_DQN.csv"), "w", newline="") as f:
    f.write("rep,time\n")

for i in range(REPETITIONS):
    start_repetition = time.time()

    # ENVIRONMENT
    env = gymnasium.make(ENVIRONMENT, render_mode="rgb_array")

    # EGO AGENT (QTable)
    ego_agent = QTable(OBS_VARIABLES, ACTION_VARIABLES, env_label=ENV_LABEL)
    try:
        ego_agent._load_model(EGO_MODEL)
        print(f"Loaded EGO model: {EGO_MODEL}")
    except Exception as e:
        print(f"Warning: Could not load EGO model: {e}")
        print("Training will start with a fresh EGO model")

    # ADVERSARIAL AGENT (DQN)
    adv_agent = DQN(OBS_VARIABLES, ACTION_VARIABLES, LEARNING_RATE, DISCOUNT, BATCH_SIZE, BUFFER_SIZE, GRADIENT_STEPS)

    ep_rewards = []
    fail_count = []

    clear_logger()
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("output", "DATA", ENV_LABEL, "DQN")
    os.makedirs(results_dir, exist_ok=True)
    RESULTS_FILE = os.path.join(results_dir, f"DQN_r{i+1}_{now_str}.csv")
    with open(RESULTS_FILE, "w", newline="") as f:
        f.write("run,avg,max,min,fr,eps\n")

    epsilon = 1

    # cycle on EPISODES
    for episode in range(EPISODES):
        # ... (training loop code remains the same) ...
        logger = get_logger(i + 1)
        episode_reward = 0
        obs = env.reset()
        states = tuple(np.array(obs_i.reshape(-1)) for obs_i in obs[0])
        done = truncated = False
        fail = 0
        cnt = 0
        while not (done or truncated):
            if episode % SHOW_EVERY == 0 and RENDER:
                env.render()
            ego_action = ego_agent.choose_action(states[0])
            if np.random.random() > epsilon:
                adv_action = adv_agent.act(states[1])
            else:
                adv_action = np.random.randint(0, ACTION_VARIABLES)
            actions = [ego_action, adv_action]
            next_states, rewards, done, truncated, info = env.step((actions[0], actions[1]))
            next_states = tuple(np.array(obs_i.reshape(-1)) for obs_i in next_states)
            adv_agent.memorize(states[1], actions[1], rewards[1], next_states[1], done)
            if len(adv_agent.memory) > BATCH_SIZE:
                adv_agent.replay()
            if info["crashed"]:
                fail = 1
            logger.info(f"{states[1].tolist()}#{actions[1]}#{rewards[1]}")
            episode_reward += rewards[1]
            cnt += 1
            states = next_states
        if episode % TARGET_UPDATE_FREQUENCY == 0:
            adv_agent.update_target()
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= EPSILON_DECAY_VALUE
            epsilon = max(EPSILON_MIN, epsilon)
        episode_reward = episode_reward/cnt if cnt > 0 else 0
        ep_rewards.append(episode_reward)
        fail_count.append(fail)
        if not episode % UPDATE_EVERY:
            avg_r = sum(ep_rewards[-UPDATE_EVERY:]) / UPDATE_EVERY
            avg_fails = sum(fail_count[-UPDATE_EVERY:]) / UPDATE_EVERY
            max_r = max(ep_rewards[-UPDATE_EVERY:])
            min_r = min(ep_rewards[-UPDATE_EVERY:])
            print(f'Episode: {episode:>5d}, average fails: {avg_fails:>4.1f}, average reward: {avg_r:>4.1f}, max reward: {max_r:>4.1f}, min reward: {min_r:>4.1f}, current epsilon: {epsilon:>1.2f}')
            with open(RESULTS_FILE, "a", newline="") as f:
                f.write(f"{episode},{avg_r},{max_r},{min_r},{avg_fails},{epsilon}\n")


    # SAVE MODELS (Modified to save in .keras format)
    model_filename = f"ADV_DQN_r{i+1}_{now_str}"

    model_dir = os.path.join("models", "ADV", ENV_LABEL, "DQN")
    os.makedirs(model_dir, exist_ok=True)
    full_model_path = os.path.join(model_dir, f"{model_filename}.keras")

    # --- ADDED THIS LINE FOR DEBUGGING ---
    print("\n>>> DEBUG: ATTEMPTING TO SAVE COMPLETE MODEL IN .KERAS FORMAT... <<<\n")

    adv_agent.model.save(full_model_path)
    print(f"Complete model saved to: {full_model_path}")

    end_repetition = time.time()
    elapsed = end_repetition - start_repetition
    with open(os.path.join(cost_dir, "times_DQN.csv"), "a", newline="") as f:
        f.write(f"{i},{elapsed}\n")

    env.close()