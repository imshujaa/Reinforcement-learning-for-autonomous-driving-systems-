import gymnasium
import highway_env

from crl_lib.qtable import QTable
from crl_lib.ppo import PPO 

from datetime import datetime
import numpy as np
import time
import logging
import sys
import os


REPETITIONS = 10
ENVIRONMENT = "highwayMA-v0"  # highwayMA, intersectionMA


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
EPISODES = 500
RENDER = False
SHOW_EVERY = 10
UPDATE_EVERY = 10

########################    FUNCTIONS

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

########################    MAIN PROGRAM

cost_dir = os.path.join("output", "DATA", ENV_LABEL, "_cost")
os.makedirs(cost_dir, exist_ok=True)
with open(os.path.join(cost_dir, "times_PPO.csv"), "w", newline="") as f:
    f.write("rep,time\n")

for i in range(REPETITIONS):
    start_repetition = time.time()

    # ENVIRONMENT
    env = gymnasium.make(ENVIRONMENT, render_mode="rgb_array")

    # EGO AGENT
    ego_agent = QTable(OBS_VARIABLES, ACTION_VARIABLES, env_label=ENV_LABEL)
    try:
        ego_agent._load_model(EGO_MODEL)
    except Exception as e:
        print(e)
        sys.exit()

    # ADVERSARIAL AGENT
    adv_agent = PPO(state_dim=OBS_VARIABLES, action_dim=ACTION_VARIABLES)
    
    ep_rewards = []
    fail_count = []

    clear_logger()
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("output", "DATA", ENV_LABEL, "PPO")
    os.makedirs(results_dir, exist_ok=True)
    RESULTS_FILE = os.path.join(results_dir, f"PPO_r{i+1}_{now_str}.csv")
    with open(RESULTS_FILE, "w", newline="") as f:
        f.write("run,avg,max,min,fr\n")

    # cycle on EPISODES
    for episode in range(EPISODES):
        logger = get_logger(i + 1)
        episode_reward = 0

        # RESET
        obs = env.reset()
        states = tuple(np.array(obs_i.reshape(1, OBS_VARIABLES).tolist()[0]) for obs_i in obs[0])

        done = truncated = False
        fail = 0
        cnt = 0

        while not (done or truncated):
            if episode % SHOW_EVERY == 0 and RENDER:
                env.render()

			# ACTIONS
            actions = [ego_agent.choose_action(states[0])]
            adv_action, log_prob, value, _ = adv_agent.get_action_and_value(states[1])
            actions.append(adv_action)

			# STEP
            next_states, rewards, done, truncated, info = env.step((actions[0], actions[1]))
            next_states = tuple(np.array(obs_i.reshape(1, OBS_VARIABLES).tolist()[0]) for obs_i in next_states)

            adv_agent.memorize(states[1], adv_action, rewards[1], log_prob, value, done)

            if info["crashed"]:
                fail = 1

            logger.info(f"{states[1].tolist()}#{actions[1]}#{rewards[1]}")
            episode_reward += rewards[1]
            cnt += 1
            states = next_states

        adv_agent.update()

        episode_reward = episode_reward / cnt
        # AVG REWARD COMPUTATION
        ep_rewards.append(episode_reward)
        fail_count.append(fail)

        if not episode % UPDATE_EVERY:
            avg_r = sum(ep_rewards[-UPDATE_EVERY:]) / UPDATE_EVERY 
            avg_fails = sum(fail_count[-UPDATE_EVERY:]) / UPDATE_EVERY 
            max_r = max(ep_rewards[-UPDATE_EVERY:])
            min_r = min(ep_rewards[-UPDATE_EVERY:])
            print(f'Episode: {episode:>5d}, average fails: {avg_fails:>4.1f}, average reward: {avg_r:>4.1f}, max reward: {max_r:>4.1f}, min reward: {min_r:>4.1f}')
            with open(RESULTS_FILE, "a", newline="") as f:
                f.write(f"{episode},{avg_r},{max_r},{min_r},{avg_fails}\n")

    # SAVE MODELS
    model_filename = f"{now_str}_r{i+1}_ep{episode}.pth"
    adv_agent.save_model(model_filename, env_label=ENV_LABEL)

    end_repetition = time.time()
    elapsed = end_repetition - start_repetition
    with open(os.path.join(cost_dir, "times_PPO.csv"), "a", newline="") as f:
        f.write(f"{i},{elapsed}\n")

    env.close()
