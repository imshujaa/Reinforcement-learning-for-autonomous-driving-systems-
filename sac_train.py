import gymnasium
import highway_env

from crl_lib.qtable import QTable
from crl_lib.sac_fixed import SAC 

from datetime import datetime
import numpy as np
import time
import logging
import sys
import os
import highway_env

from crl_lib.qtable import QTable
from crl_lib.sac_fixed import SAC

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
    ADV_SAC_MODEL = os.path.join("models", "ADV", "highway", "sac_ADV_pretrained.pth")

elif ENVIRONMENT == "intersectionMA-v0":
    FEATURES = 7
    NUM_VEHICLES = 2
    ACTION_VARIABLES = 3
    ENV_LABEL = "intersection"
    EGO_MODEL = os.path.join("models", "EGO", "intersection", "table_EGO_TRAIN_8000.txt")
    ADV_SAC_MODEL = os.path.join("models", "ADV", "intersection", "sac_ADV_pretrained.pth")

OBS_VARIABLES = NUM_VEHICLES * FEATURES

EPISODES = 500 # episode budget   

# SAC specific hyperparameters
SAC_Q_LR = 1e-3
SAC_POLICY_LR = 1e-3
SAC_TARGET_ENTROPY_SCALE = 0.5 # 0.5 per intersectionMA
SAC_AUTOTUNE_ALPHA = True
SAC_ALPHA = 0.2 #0.05 per intersectionMA
SAC_TAU = 0.01
SAC_DISCOUNT = 0.99
SAC_BATCH_SIZE = 128
SAC_BUFFER_SIZE = 50000
SAC_GRADIENT_STEPS = 2
SAC_LEARNING_STARTS = 500

RENDER = False
SHOW_EVERY = 10  # How oftern the current solution is rendered
UPDATE_EVERY = 10  # How oftern the current progress is recorded

# Exploration settings
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2 
EPSILON_MIN = 0.05
EPSILON_DECAY_VALUE = (1.0 - EPSILON_MIN) / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

########################    FUNCTIONS

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

########################    MAIN PROGRAM

cost_dir = os.path.join("output", "DATA", ENV_LABEL, "_cost")
os.makedirs(cost_dir, exist_ok=True)
with open(os.path.join(cost_dir, "times_SAC.csv"), "w", newline="") as f:
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
    adv_agent = SAC(
        state_size=OBS_VARIABLES, 
        action_size=ACTION_VARIABLES,
        q_lr=SAC_Q_LR,
        policy_lr=SAC_POLICY_LR,
        target_entropy_scale=SAC_TARGET_ENTROPY_SCALE,
        autotune=SAC_AUTOTUNE_ALPHA,
        alpha=SAC_ALPHA,
        tau=SAC_TAU,
        gamma=SAC_DISCOUNT,
        batch_size=SAC_BATCH_SIZE,
        buffer_size=SAC_BUFFER_SIZE,
        gradient_steps=SAC_GRADIENT_STEPS,
        use_prioritized_buffer=True
    )
    
    adv_agent.env_label = ENV_LABEL  
    try:
        if os.path.exists(ADV_SAC_MODEL):
            adv_agent.load_model(ADV_SAC_MODEL)
            print(f"Loaded pre-trained SAC model from {ADV_SAC_MODEL}")
        else:
            print(f"No pre-trained SAC model found at {ADV_SAC_MODEL}, starting with random weights")
    except Exception as e:
        print(f"Could not load pre-trained SAC model: {e}")
        print("Starting with random weights")

    ep_rewards = []
    fail_count = []

    clear_logger()
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("output", "DATA", ENV_LABEL, "SAC")
    os.makedirs(results_dir, exist_ok=True)
    RESULTS_FILE = os.path.join(results_dir, f"SAC_r{i+1}_{now_str}.csv")
    with open(RESULTS_FILE, "w", newline="") as f:
        f.write("run,avg,max,min,fr,eps\n")

    epsilon = 1
    global_step = 0

    # cycle on EPISODES
    for episode in range(EPISODES):
        logger = get_logger(i + 1)
        episode_reward = 0

        # RESET
        obs, info = env.reset()
        if isinstance(obs, (list, tuple)):
            states = [np.array(obs_i).flatten() for obs_i in obs]
        else:
            state = np.array(obs).flatten()
            states = [state, state]

        done = truncated = False
        fail = 0
        cnt = 0

        while not (done or truncated):
            if episode % SHOW_EVERY == 0 and RENDER:
                env.render()

            # ACTIONS
            actions = [ego_agent.choose_action(states[0])]
            
            if np.random.random() > epsilon:
                action = adv_agent.act(states[1].reshape(1, -1))
                actions.append(action)
            else:
                actions.append(np.random.randint(0, ACTION_VARIABLES))

            # STEP 
            if isinstance(obs, (list, tuple)):
                next_states, rewards, done, truncated, info = env.step(tuple(actions))
                next_states = [np.array(obs_i).flatten() for obs_i in next_states]
            else:
                # Single observation case
                next_obs, reward, done, truncated, info = env.step(actions[0])
                next_state = np.array(next_obs).flatten()
                next_states = [next_state, next_state]
                rewards = [reward, reward]

            # Handle done and truncated
            if isinstance(rewards, (list, tuple)):
                reward_adv = rewards[1]
            else:
                reward_adv = rewards
                
            if isinstance(done, (list, tuple)):
                done_flag = done[1]
            else:
                done_flag = done

            # Store experience in SAC replay buffer
            adv_agent.memorize(states[1], actions[1], reward_adv, next_states[1], done_flag)

            # Train SAC if enough experiences collected
            global_step += 1
            if global_step > SAC_LEARNING_STARTS and len(adv_agent.rb) >= SAC_BATCH_SIZE:
                # SAC replay now handles gradient steps internally
                adv_agent.replay()
                
                # Update target networks
                if global_step % 2 == 0: 
                    adv_agent.update_target()

            if isinstance(info, dict) and "crashed" in info:
                fail = info["crashed"]
            elif isinstance(info, (list, tuple)) and len(info) > 1:
                fail = info[1].get("crashed", 0) if isinstance(info[1], dict) else 0
            else:
                fail = 0

            logger.info(f"{states[1].tolist()}#{actions[1]}#{reward_adv}")
            episode_reward += reward_adv
            cnt += 1
            states = next_states

        # EPSILON
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= EPSILON_DECAY_VALUE
            epsilon = max(EPSILON_MIN, epsilon)

        episode_reward = episode_reward/cnt
        # AVG REWARD COMPUTATION
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

        # SAVE MODELS
        if not episode % UPDATE_EVERY:
            try:
                model_filename = f"SAC_{now_str}_r{i+1}_ep{episode}.pth"
                adv_agent.save_model(model_filename)
                print(f"Model saved at episode {episode}")
            except Exception as e:
                print(f"Error saving model at episode {episode}: {e}")

    # Save final model for this repetition
    try:
        final_model_filename = f"SAC_{now_str}_r{i+1}_final.pth"
        adv_agent.save_model(final_model_filename)
        print(f"Final model for repetition {i+1} saved as {final_model_filename}")
    except Exception as e:
        print(f"Error saving final model for repetition {i+1}: {e}")

    end_repetition = time.time()
    elapsed = end_repetition - start_repetition
    with open(os.path.join(cost_dir, "times_SAC.csv"), "a", newline="") as f:
        f.write(f"{i},{elapsed}\n")

    env.close()