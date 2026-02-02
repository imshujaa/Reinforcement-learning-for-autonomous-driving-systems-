import gym
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd
from crl_lib.dqn import DQN
import logging

env = gym.make('CartPole-v1')
times_of_repetitions = 10
RESULTS_FILE = ""
ENV_LABEL = "cartpole"

########################	DQN PARAMETERS

# DQN specific parameters
LEARNING_RATE = 0.001
DISCOUNT = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 10000
GRADIENT_STEPS = 1

EPISODES = 500  # Number of iterations episode
SHOW_EVERY = 10  # How often the current solution is rendered
UPDATE_EVERY = 10  # How often the current progress is recorded
UPDATE_TARGET_EVERY = 10  # How often to update target network

# Exploration settings
epsilon = 1  
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 3
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


########################	FUNCTION DEFINITIONS

# Create DQN agent
def create_dqn_agent():
	obsSpaceSize = len(env.observation_space.high)
	actionSpaceSize = env.action_space.n
	dqn_agent = DQN(
		state_size=obsSpaceSize,
		action_size=actionSpaceSize,
		_learning_rate=LEARNING_RATE,
		_discount=DISCOUNT,
		_batch_size=BATCH_SIZE,
		_buffer_size=BUFFER_SIZE,
		_gradient_steps=GRADIENT_STEPS,
		env_label="cartpole"
	)
	return dqn_agent

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

########################	PROGRAM

for i in range(0, times_of_repetitions):
	env = gym.make('CartPole-v1')
	epsilon = 1
	clear_logger()
	now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
	results_dir = os.path.join("output", "DATA", ENV_LABEL, "DQN")
	os.makedirs(results_dir, exist_ok=True)
	RESULTS_FILE = os.path.join(results_dir, f"DQN_r{i+1}_{now_str}.csv")
	dqn_agent = create_dqn_agent()

	previousCnt = []  
	previousFailures = [] 
	metrics = {'ep': [], 'avg': [], 'min': [], 'max': [], 'failure_rate': []}  
	with open(RESULTS_FILE, "w", newline="") as f:
		f.write("run,avg,min,max,eps,fr" + "\n")
	logger = get_logger(i + 1)
	# cycle on EPISODES
	for episode in range(EPISODES):
		state, _ = env.reset() 
		state = np.array(state, dtype=np.float32)
		done = False 
		cnt = 0  # how many movements cart has made
		fail = 0  # track if this episode is a failure

		# cycle on STEPS till done
		while not done:
			cnt += 1
			# Get action 
			if np.random.random() > epsilon:
				action = dqn_agent.act(state)
			else:
				action = np.random.randint(0, env.action_space.n)
			

			newState, reward, terminated, truncated, _ = env.step(action) 
			done = terminated or truncated
			newState = np.array(newState, dtype=np.float32)

			# Reward
			if not done:
				reward = 1.0  # +1 for each step in the episode in which the pole is balanced
			else:
				# Penalty for falling before the episode length threshold
				if cnt < 195: 
					reward = -10.0
					fail = 1
				else:
					reward = 10.0
					fail = 0

			dqn_agent.memorize(state, action, reward, newState, done)

			state = newState

		# Check if the episode is a failure
		if done and cnt < 500:
			fail = 1
		logger.info(f"{state.tolist()}#{action}#{reward}")
		previousCnt.append(cnt)
		previousFailures.append(fail)

		# Train the DQN if we have enough experiences
		if len(dqn_agent.memory) > BATCH_SIZE:
			dqn_agent.replay()

		# Update target network
		if episode % UPDATE_TARGET_EVERY == 0:
			dqn_agent.update_target()

		# Decaying is being done every episode if episode number is within decaying range
		if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
			epsilon -= epsilon_decay_value

		# Add new metrics for graph
		if episode % UPDATE_EVERY == 0:
			latestRuns = previousCnt[-UPDATE_EVERY:]
			latestFailures = previousFailures[-UPDATE_EVERY:]
			averageCnt = sum(latestRuns) / len(latestRuns)
			failureRate = sum(latestFailures) / len(latestFailures)  # Calculate failure rate
			metrics['ep'].append(episode)
			metrics['avg'].append(averageCnt)
			metrics['min'].append(min(latestRuns))
			metrics['max'].append(max(latestRuns))
			metrics['failure_rate'].append(failureRate)
			print("Run:", episode, "Average:", averageCnt, "Min:", min(latestRuns), "Max:", max(latestRuns), "Eps:", epsilon, "Failure Rate:", failureRate)
			with open(RESULTS_FILE, "a", newline="") as f:
				f.write(str(episode) + "," + str(averageCnt) + "," + str(min(latestRuns)) + "," + str(max(latestRuns))+ ","+ str(epsilon) + "," + str(failureRate) + "\n")

	# Save the trained model
	dqn_agent._save_model("final", f"r{i+1}_{now_str}")

	env.close()
