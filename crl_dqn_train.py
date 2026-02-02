import os
import sys
import random
from collections import deque
from datetime import datetime
from statistics import mean
import numpy as np
import gymnasium as gym
import highway_env
import pandas as pd
import csv
from sklearn.cluster import KMeans

# Import from your library
from crl_lib.qtable import QTable
from crl_lib.dqn import DQN
# --- CAUSAL PART START ---
from crl_lib.util import generate_dot, build_img
from crl_lib.query import init_model, counterfactual
# --- CAUSAL PART END ---


# --- CONFIGURATION ---
REPETITIONS = 10
EPISODES = 500
UPDATE_EVERY = 10

ENVIRONMENT = "intersectionMA-v0"


if ENVIRONMENT == "highwayMA-v0":
    FEATURES = 5
    NUM_VEHICLES = 2
    ACTION_VARIABLES = 5
    ENV_LABEL = "highway"
    EGO_MODEL_PATH = os.path.join("models", "EGO", "highway", "table_EGO_TRAIN_2000.txt")
    
elif ENVIRONMENT == "intersectionMA-v0":
    FEATURES = 7
    NUM_VEHICLES = 2
    ACTION_VARIABLES = 3
    ENV_LABEL = "intersection"
    EGO_MODEL_PATH = os.path.join("models", "EGO", "intersection", "table_EGO_TRAIN_8000.txt")

OBS_VARIABLES = NUM_VEHICLES * FEATURES

config = { "reward": { "type": "TTCIReward", "target_vehicles": 1 }, "collision_reward": 5.0 }

# --- DQN HYPERPARAMETERS ---
LEARNING_RATE = 0.001
DISCOUNT = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 10000
GRADIENT_STEPS = 1
TARGET_UPDATE_FREQUENCY = 50 

epsilon = 1.0
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 3
EPSILON_MIN = 0.1
epsilon_decay_value = (1.0 - EPSILON_MIN) / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


# --- CAUSAL PART START ---
########################    CAUSAL RL PARAMETERS
PLAN_AFTER = 10
ACTIONS_TO_PLAN = 1
ROLLOUT_STEPS = 1
START_PLANNING_AT = 50
UPDATE_MODEL_EVERY = 100
SIMULATIONS = 5
NUM_CLUSTERS = 100
DISCOVERY = False

########################    MODEL VARIABLES
DATASET_FILE = "crl_lib/dataset.csv"
DOT_FILE = "crl_lib/causal_graph.dot"
IMG_FILE = "crl_lib/causal_graph.svg"

columns = []
for timestep in range(ROLLOUT_STEPS+1):
    for observation in range(OBS_VARIABLES):
        columns.append(f"O{observation}_{timestep}")
    if timestep > 0:
        columns.append(f"R_{timestep-1}")
    if timestep < ROLLOUT_STEPS:
        columns.append(f"A_{timestep}")

if not DISCOVERY:
    print("Generating Causal Graph...")
    generate_dot(OBS_VARIABLES, ROLLOUT_STEPS + 1, DOT_FILE)

########################    HELPER FUNCTIONS
ACTIONS_ALL = { 0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER' }
ACTIONS_REVERSE = {v: k for k, v in ACTIONS_ALL.items()}

def number_to_action(number):
    return ACTIONS_ALL.get(number, "UNKNOWN_ACTION")

def action_to_number(action):
    return ACTIONS_REVERSE.get(action, -1)

def select_transitions(all_values):
    """Selects transitions for planning using KMeans clustering."""
    if len(all_values) <= SIMULATIONS:
        return all_values

    orig_df = pd.DataFrame(all_values)
    # Select only the initial state and reward for clustering
    df_to_cluster = orig_df.iloc[:, :OBS_VARIABLES]
    
    kmeans = KMeans(n_clusters=min(NUM_CLUSTERS, len(df_to_cluster)), n_init='auto', random_state=0)
    labels = kmeans.fit_predict(df_to_cluster.values)
    
    clusters = [orig_df[labels == i] for i in range(kmeans.n_clusters)]
    
    # Calculate average rewards for weighting (assuming reward is one of the last columns)
    
    avg_rewards = [c.iloc[:, -1].mean() for c in clusters]
    
    # Normalize weights to handle negative rewards
    min_reward = min(avg_rewards)
    weights = [r - min_reward + 1e-6 for r in avg_rewards]
    
    to_simulate_transitions = []
    
    # Flatten the list of clusters into a list of indices and their corresponding cluster id
    indexed_rows = []
    for i, cluster in enumerate(clusters):
        for idx in cluster.index:
            indexed_rows.append({'original_index': idx, 'cluster_id': i})

    df_indexed = pd.DataFrame(indexed_rows).set_index('original_index')
    
    # Weighted sampling of clusters
    cluster_ids = [i for i, c in enumerate(clusters) if not c.empty]
    if not cluster_ids: return []

    # Ensure weights correspond to available clusters
    valid_weights = [weights[i] for i in cluster_ids]
    
    try:
        chosen_cluster_ids = random.choices(cluster_ids, weights=valid_weights, k=SIMULATIONS)
        
        for cluster_id in chosen_cluster_ids:
            # Sample a random row from the chosen cluster
            random_index = clusters[cluster_id].sample(n=1).index[0]
            to_simulate_transitions.append(orig_df.loc[random_index].values.tolist())

    except ValueError as e:
        print(f"Warning: Could not perform weighted choice, falling back to random sampling. Error: {e}")
        # Fallback to simple random sampling if weights are problematic
        indices = np.random.choice(len(all_values), SIMULATIONS, replace=False)
        to_simulate_transitions = [all_values[i] for i in indices]

    return to_simulate_transitions
# --- CAUSAL PART END ---


# ==============================================================================
# ===== MAIN PROGRAM ===========================================================
# ==============================================================================
global_steps = 0

for i in range(REPETITIONS):
    env = gym.make(ENVIRONMENT, config=config, render_mode='rgb_array')
    epsilon = 1.0

    try:
        ego_agent = QTable(OBS_VARIABLES, ACTION_VARIABLES, env_label=ENV_LABEL)
        ego_agent._load_model(EGO_MODEL_PATH)
        print(f"Repetition {i+1}/{REPETITIONS} | Successfully loaded EGO agent model.")
    except Exception as e:
        print(f"Warning: could not load EGO model ({e}). Using a random EGO agent.")
        sys.exit()

    adv_agent = DQN(OBS_VARIABLES, ACTION_VARIABLES, learning_rate=LEARNING_RATE,
                      discount=DISCOUNT, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE,
                      gradient_steps=GRADIENT_STEPS)

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("output", "DATA", ENV_LABEL, "CRL_DQN")
    os.makedirs(results_dir, exist_ok=True)
    file_name = f"CRL_DQN_r{i+1}_{now_str}.csv"
    RESULTS_FILE = os.path.join(results_dir, file_name)
    print(f"Results will be saved in: {RESULTS_FILE}")

    with open(RESULTS_FILE, "w", newline="") as f:
        f.write("run,avg,max,min,fr,eps\n")

    previous_rewards, fail_count = [], []

    for episode in range(EPISODES):
        obs, info = env.reset()
        states = obs
        done, truncated, total_reward, steps, fail = False, False, 0.0, 0, 0
        episode_losses = []

        # --- CAUSAL PART START ---
        # Initialize/Update Causal Model
        if episode == 0:
            buffer_values = []
            all_values = []
            pd.DataFrame(columns=columns).to_csv(DATASET_FILE, index=False)
        elif (episode == START_PLANNING_AT or (episode > START_PLANNING_AT and episode % UPDATE_MODEL_EVERY == 0)):
            with open(DATASET_FILE, "a", newline="") as f:
                csv.writer(f).writerows(buffer_values)
            buffer_values = []
            df_t = pd.read_csv(DATASET_FILE)
            model = init_model(df_t, DOT_FILE)
            print(f"Episode {episode}: Causal model updated with {len(df_t)} samples.")

        # Main Planning Phase
        if episode >= START_PLANNING_AT and episode % PLAN_AFTER == 0:
            print(f"Episode {episode}: Starting planning phase...")
            transitions_to_plan = select_transitions(all_values)

            for transition in transitions_to_plan:
                if len(transition) < OBS_VARIABLES + 1: continue # Skip malformed transitions
                observed_state = transition[:OBS_VARIABLES]

                for plan_action_num in range(ACTION_VARIABLES):
                    plan_action_str = number_to_action(plan_action_num)
                    
                    try:
                        sim_samples = counterfactual(model, columns, transition, "A_0", plan_action_str)
                        plan_next_state = list(sim_samples[[f"O{i}_1" for i in range(OBS_VARIABLES)]].mean())
                        plan_reward = round(float(sim_samples["R_0"].mean()), 3)
                        plan_done = 1 if plan_reward >= 1 else 0

                        adv_agent.plan_memory.append((tuple(observed_state), plan_action_num, plan_reward, tuple(plan_next_state), plan_done))
                    except Exception as e:
                        # print(f"Counterfactual failed: {e}")
                        pass
        
        action_values = []
        action_count = 0
        # --- CAUSAL PART END ---


        while not (done or truncated):
            global_steps += 1
            steps += 1

            ego_action = ego_agent.choose_action(states[0].flatten())

            if np.random.random() > epsilon:
                adv_action = adv_agent.act(states[1].flatten())
            else:
                adv_action = np.random.randint(0, ACTION_VARIABLES)

            # --- CAUSAL PART START --- (Data Collection)
            if action_count == 0:
                action_values.extend(states[1].flatten())
            action_values.append(number_to_action(adv_action))
            # --- CAUSAL PART END ---

            actions = (ego_action, adv_action)
            next_obs, rewards, done, truncated, info = env.step(actions)

            if info.get("crashed", False):
                fail = 1

            adv_reward = np.clip(float(rewards[1]), -1.0, 1.0)
            adv_agent.memorize(states[1].flatten(), adv_action, adv_reward, next_obs[1].flatten(), done or truncated)
            total_reward += adv_reward
            
            # --- CAUSAL PART START --- (Data Collection)
            action_values.extend(next_obs[1].flatten())
            action_values.append(rewards[1])
            if action_count == ROLLOUT_STEPS - 1 or done or truncated:
                if len(action_values) == len(columns):
                    all_values.append(action_values)
                    buffer_values.append(action_values)
                action_values = []
                action_count = 0
            else:
                action_count += 1
            # --- CAUSAL PART END ---
            
            states = next_obs

            if len(adv_agent.memory) > BATCH_SIZE:
                loss = adv_agent.replay()
                if loss is not None:
                    episode_losses.append(loss)
            
            # This is where the agent learns from imagined experiences
            if global_steps % 5 == 0 and len(adv_agent.plan_memory) > BATCH_SIZE:
                adv_agent.replay_plan()

        if episode % TARGET_UPDATE_FREQUENCY == 0:
            adv_agent.update_target()

        episode_avg_reward = total_reward / steps if steps > 0 else 0.0
        previous_rewards.append(episode_avg_reward)
        fail_count.append(fail)

        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value
            epsilon = max(EPSILON_MIN, epsilon)

        if episode % UPDATE_EVERY == 0:
            latest_runs, latest_fails = previous_rewards[-UPDATE_EVERY:], fail_count[-UPDATE_EVERY:]
            avg_reward = mean(latest_runs) if latest_runs else 0.0
            avg_fails = mean(latest_fails) if latest_fails else 0.0
            avg_loss = mean(episode_losses) if episode_losses else 0.0
            print(f"Rep: {i+1}, Ep: {episode}, Avg Reward: {avg_reward:.4f}, Avg Fails: {avg_fails:.2f}, Epsilon: {epsilon:.3f}, Loss: {avg_loss:.6f}")

            with open(RESULTS_FILE, "a", newline="") as f:
                f.write(f"{episode},{avg_reward},{max(latest_runs) if latest_runs else 0.0},{min(latest_runs) if latest_runs else 0.0},{avg_fails},{epsilon}\n")

    final_model_path = os.path.join(results_dir, f"adv_model_rep{i+1}_final.keras")
    adv_agent.save(final_model_path)
    env.close()

print("\nProcessing complete.")