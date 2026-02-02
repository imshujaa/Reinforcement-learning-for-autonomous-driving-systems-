import gymnasium
import highway_env
import tensorflow as tf
import numpy as np
import os
import sys
import glob
from datetime import datetime
import logging
from crl_lib.qtable import QTable # Still needed for the EGO agent

# --- CONFIGURATION ---
ENVIRONMENT = "highwayMA-v0" # Options: "highwayMA-v0", "intersectionMA-v0"
EPISODES = 100

# --- FIXED PARAMETERS (These change automatically) ---
if ENVIRONMENT == "highwayMA-v0":
    FEATURES = 5
    NUM_VEHICLES = 2
    ACTION_VARIABLES = 5
    ENV_LABEL = "highway"
    EGO_MODEL_PATH = "models/EGO/highway/table_EGO_TRAIN_2000.txt"
elif ENVIRONMENT == "intersectionMA-v0":
    FEATURES = 7
    NUM_VEHICLES = 2
    ACTION_VARIABLES = 3
    ENV_LABEL = "intersection"
    EGO_MODEL_PATH = "models/EGO/intersection/table_EGO_TRAIN_8000.txt"
OBS_VARIABLES = NUM_VEHICLES * FEATURES


# --- PATHS ---
# Set the technique label for output folders
TECHNIQUE = "DQN" # You can change this to "CRL_DQN" when you test those models

# --- MODIFIED THIS LINE TO BE HARDCODED ---
ADV_MODELS_PATH = "models/ADV/highway/DQN/*.keras"

# Setup output folders exactly like all_test.py
OUTPUT_FOLDER = f"output/DATA/{ENV_LABEL}/_test_pretrained/{TECHNIQUE}/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
RESULTS_FILE = os.path.join(OUTPUT_FOLDER, f"overall_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")


######################## FUNCTIONS (Copied from all_test.py)

def clear_logger():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

def get_logger(model_name):
    logger = logging.getLogger()
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    log_folder = f'output/LOGS/{ENV_LABEL}/_test_pretrained/{TECHNIQUE}/'
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, f'{model_name}_{now}.log')

    logging.basicConfig(filename=log_file, format='%(asctime)s %(message)s')
    logger.setLevel(logging.DEBUG)
    logger.info("Started")
    return logger

######################## MAIN PROGRAM

# Find all adversarial model files to test
adv_model_files = glob.glob(ADV_MODELS_PATH)
if not adv_model_files:
    print(f"Warning: No models found at the path: {ADV_MODELS_PATH}")
    sys.exit()

# Write header to the main CSV results file
with open(RESULTS_FILE, "w", newline="") as f:
    f.write("model,avg_reward,fail_rate\n")

# Loop through each adversarial model file
for adv_model_path in adv_model_files:
    model_name = os.path.splitext(os.path.basename(adv_model_path))[0]
    print(f"\n--- Starting test for model: {model_name} ---")

    # 1. Load the EGO Agent (from Q-Table)
    ego_agent = QTable(OBS_VARIABLES, ACTION_VARIABLES)
    try:
        ego_agent._load_model(EGO_MODEL_PATH)
    except Exception as e:
        print(f"Error loading EGO agent: {e}")
        sys.exit()

    # 2. Load the Adversarial Agent (from .keras file)
    try:
        adv_agent_dqn = tf.keras.models.load_model(adv_model_path)
    except Exception as e:
        print(f"Error loading Adversarial DQN model '{adv_model_path}': {e}")
        continue # Skip to the next model if one fails to load

    # 3. Setup the Environment
    env = gymnasium.make(ENVIRONMENT, render_mode="rgb_array")
    
    ep_rewards = []
    fail_count = []
    clear_logger()

    # Cycle on EPISODES
    for episode in range(EPISODES):
        logger = get_logger(model_name)
        episode_reward = 0
        obs, info = env.reset()
        states = tuple(np.array(obs_i.reshape(1, OBS_VARIABLES).tolist()[0]) for obs_i in obs)

        done = truncated = False
        fail = 0
        step_count = 0

        while not (done or truncated):
            ego_action = ego_agent.choose_action(states[0])
            
            adv_state_reshaped = np.reshape(states[1], [1, OBS_VARIABLES])
            q_values = adv_agent_dqn.predict(adv_state_reshaped, verbose=0)
            adv_action = np.argmax(q_values[0])
            
            next_states, rewards, done, truncated, info = env.step((ego_action, adv_action))
            next_states = tuple(np.array(obs_i.reshape(1, OBS_VARIABLES).tolist()[0]) for obs_i in next_states)

            if info["crashed"]:
                fail = 1
            
            # Save step data to the log file
            logger.info(f"{states[1].tolist()}#{adv_action}#{rewards[1]}")
            episode_reward += rewards[1]
            step_count += 1
            states = next_states

        ep_rewards.append(episode_reward / step_count if step_count > 0 else 0)
        fail_count.append(fail)

    # --- Calculate and Save Final Results for this model ---
    overall_avg_reward = np.mean(ep_rewards)
    overall_fail_rate = np.mean(fail_count)

    print(f'Model: {model_name}, Avg Reward: {overall_avg_reward:.3f}, Fail Rate: {overall_fail_rate:.3f}')

    # Append results to the main CSV file
    with open(RESULTS_FILE, "a", newline="") as f:
        f.write(f"{model_name},{overall_avg_reward},{overall_fail_rate}\n")

    env.close()

print("\n--- ALL TESTS COMPLETE ---")