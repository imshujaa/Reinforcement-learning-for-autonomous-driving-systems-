import gymnasium
import highway_env
import glob
import os
from crl_lib.qtable import QTable
from datetime import datetime
import numpy as np
import logging
import sys

ENVIRONMENT = "intersectionMA-v0" # highwayMA, intersectionMA

# FIXED
if ENVIRONMENT == "highwayMA-v0":
    FEATURES = 5
    NUM_VEHICLES = 2
    ACTION_VARIABLES = 5
    ENV_LABEL = "highway"
    EGO_MODEL = "models/EGO/highway/table_EGO_TRAIN_2000.txt"

elif ENVIRONMENT == "intersectionMA-v0":
    FEATURES = 7
    NUM_VEHICLES = 2
    ACTION_VARIABLES = 3
    ENV_LABEL = "intersection"
    EGO_MODEL = "models/EGO/intersection/table_EGO_TRAIN_8000.txt"

OBS_VARIABLES = NUM_VEHICLES*FEATURES

# VARIABLES
EPISODES = 100
RENDER = False
SHOW_EVERY = 1

# Paths
TECHNIQUE = "Q"  # CRL, Q, RS
# --- MODIFIED THIS LINE ---
ADV_MODELS_PATH = "models/ADV/intersection/Q/*.txt"    #add here the path to the adversarial models, e.g. "models/ADV/highway/*.txt" or "models/ADV/intersection/*.txt"


OUTPUT_FOLDER = f"output/DATA/{ENV_LABEL}/_test_pretrained/{TECHNIQUE}/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
RESULTS_FILE = os.path.join(OUTPUT_FOLDER, f"overall_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")

######################## FUNCTIONS

def clear_logger():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

# --- MODIFIED THIS FUNCTION ---
def get_logger(model_name):
    logger = logging.getLogger()
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f'output/LOGS/{ENV_LABEL}/_test_pretrained/{TECHNIQUE}/{model_name}_{now}.log'

    # Create the directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(filename=log_file, format='%(asctime)s %(message)s')
    logger.setLevel(logging.DEBUG)
    logger.info("Started")
    return logger

######################## MAIN PROGRAM

adv_model_files = glob.glob(ADV_MODELS_PATH)

# Write headers to single overall results file
with open(RESULTS_FILE, "w", newline="") as f:
    f.write("model,avg_reward,fail_rate\n")

for adv_model_path in adv_model_files:
    model_name = os.path.splitext(os.path.basename(adv_model_path))[0]

    # ENVIRONMENT SETUP
    env = gymnasium.make(ENVIRONMENT, render_mode="rgb_array")

    # Load EGO AGENT
    ego_agent = QTable(OBS_VARIABLES, ACTION_VARIABLES)
    try:
        ego_agent._load_model(EGO_MODEL)
    except Exception as e:
        print(e)
        sys.exit()

    # Load ADVERSARIAL AGENT
    adv_agent = QTable(OBS_VARIABLES, ACTION_VARIABLES)
    try:
        adv_agent._load_model(adv_model_path)
    except Exception as e:
        print(e)
        sys.exit()

    ep_rewards = []
    fail_count = []

    clear_logger()

    # Cycle on EPISODES
    for episode in range(EPISODES):
        logger = get_logger(model_name)
        episode_reward = 0

        obs = env.reset()
        states = tuple(np.array(obs_i.reshape(1, OBS_VARIABLES).tolist()[0]) for obs_i in obs[0])

        done = truncated = False
        fail = 0
        cnt = 0

        while not (done or truncated):
            if episode % SHOW_EVERY == 0 and RENDER:
                env.render()

            if TECHNIQUE != "RS":
                actions = [
                    ego_agent.choose_action(states[0]),
                    adv_agent.choose_action(states[1])
                ]
            else:
                actions = [
                    ego_agent.choose_action(states[0]),
                    np.random.randint(0, env.action_space[0].n)
                ]

            next_states, rewards, done, truncated, info = env.step((actions[0], actions[1]))
            next_states = tuple(np.array(obs_i.reshape(1, OBS_VARIABLES).tolist()[0]) for obs_i in next_states)

            if info["crashed"]:
                fail = 1

            logger.info(f"{states[1].tolist()}#{actions[1]}#{rewards[1]}")
            episode_reward += rewards[1]
            cnt += 1
            states = next_states

        ep_rewards.append(episode_reward / cnt)
        fail_count.append(fail)

    overall_avg_reward = np.mean(ep_rewards)
    overall_fail_rate = np.mean(fail_count)

    print(f'Model: {model_name}, Avg Reward: {overall_avg_reward:.3f}, Fail Rate: {overall_fail_rate:.3f}')

    # Append results to overall file
    with open(RESULTS_FILE, "a", newline="") as f:
        f.write(f"{model_name},{overall_avg_reward},{overall_fail_rate}\n")

    env.close()