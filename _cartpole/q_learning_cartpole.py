import gym
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd
import logging

#from crl_lib.util import *
#from crl_lib.query import *
from crl_lib.qtable import QTable

env = gym.make('CartPole-v1')
times_of_repetitions = 10
RESULTS_FILE = ""
ENV_LABEL = "cartpole"

########################	Q-LEARNING PARAMETERS

# How much new info will override old info. 0 means nothing is learned, 1 means only most recent is considered, old knowledge is discarded
LEARNING_RATE = 0.1
# Between 0 and 1, mesue of how much we carre about future reward over immedate reward
DISCOUNT = 0.95
EPISODES = 500  # Number of iterations episode
SHOW_EVERY = 10  # How oftern the current solution is rendered
UPDATE_EVERY = 10  # How oftern the current progress is recorded

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 3
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


########################	FUNCTION DEFINITIONS

# Create Q table using QTable class
def create_q_table():
	obsSpaceSize = len(env.observation_space.high)
	actionSpaceSize = env.action_space.n
	qTable = QTable(
		state_size=obsSpaceSize,
		action_size=actionSpaceSize,
		_learning_rate=LEARNING_RATE,
		_discount=DISCOUNT,
		_num_bins=20,
		env_label="cartpole"
	)
	return qTable

def clear_logger():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

def get_logger(repetition):
    logger = logging.getLogger()
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("output", "LOGS", ENV_LABEL, "Q")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"Q_logs_r{repetition}_{now_str}.log")
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
	results_dir = os.path.join("output", "DATA", ENV_LABEL, "Q")
	os.makedirs(results_dir, exist_ok=True)
	RESULTS_FILE = os.path.join(results_dir, f"Q_r{i+1}_{now_str}.csv")
	qTable = create_q_table()

	previousCnt = []  # array of all scores over runs
	previousFailures = []  # array to track failures (True if failed, False if successful)
	metrics = {'ep': [], 'avg': [], 'min': [], 'max': [], 'failure_rate': []}  # metrics recorded for graph
	with open(RESULTS_FILE, "w", newline="") as f:
		f.write("run,avg,min,max,eps,failure_rate" + "\n")
	logger = get_logger(i + 1)
	# cycle on EPISODES
	for episode in range(EPISODES):
		state, _ = env.reset()  # env.reset() returns (observation, info)
		done = False  # has the enviroment finished?
		cnt = 0  # how may movements cart has made
		fail = 0  # track if this episode is a failure

		# cycle on STEPS till done
		while not done:
			# if episode % SHOW_EVERY == 0:
			# 	env.render()  # if running RL comment this out
			cnt += 1
			# Get action 
			if np.random.random() > epsilon:
				action = qTable.choose_action(state)
			else:
				action = np.random.randint(0, env.action_space.n)
			

			newState, reward, terminated, truncated, _ = env.step(action)  # env.step() returns 5 values
			done = terminated or truncated

			# Reward logic similar to PPO
			if not done:
				reward = 1.0  # +1 for each step in the episode in which the pole is balanced
			else:
				# Penalty for falling before the episode length threshold
				if cnt < 195:  # CartPole-v1 max is 500, consider <195 as failure
					reward = -10.0
					fail = 1
				else:
					reward = 10.0
					fail = 0

			# Update Q-table using the learn method
			qTable.learn(state, action, reward, newState, done)

			state = newState

		# Check if the episode is a failure (same logic as PPO)
		if done and cnt < 500:
			fail = 1
		logger.info(f"{state.tolist()}#{action}#{reward}")
		previousCnt.append(cnt)
		# Track if this episode was a failure
		previousFailures.append(fail)

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


	env.close()