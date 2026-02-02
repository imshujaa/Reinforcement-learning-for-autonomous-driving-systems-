import gymnasium as gyms
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from crl_lib.qtable import QTable
import highway_env
from datetime import datetime
import time

times_of_repetitions = 1
ENVIRONMENT = "intersection-v0"

if ENVIRONMENT == "highway-v0":
	FEATURES = 5
	NUM_VEHICLES = 2
	ENV_LABEL = "highway"
elif ENVIRONMENT == "intersection-v0":
	FEATURES = 7
	NUM_VEHICLES = 2
	ENV_LABEL = "intersection"

OBS_VARIABLES = NUM_VEHICLES*FEATURES
RESULTS_FILE = ""

########################	Q-LEARNING PARAMETERS

EPISODES = 8000   # Number of iterations episode
RENDER = False
SHOW_EVERY = 1  # How oftern the current solution is rendered
UPDATE_EVERY = 100  # How oftern the current progress is recorded

# # Exploration settings
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 1
epsilon_decay_value = START_EPSILON_DECAYING / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
EPSILON_MIN = 0.1

########################	PROGRAM

for i in range(0, times_of_repetitions):
	env = gyms.make(ENVIRONMENT,render_mode="rgb_array")
	now = datetime.now()


	epsilon = 1
	RESULTS_FILE = "output/DATA/_ego_train/"+ENV_LABEL+"/ego_train_"+str(now)+".csv"
	
	# bins, obsSpaceSize, qTable = create_bins_and_q_table()
	qTable = QTable(OBS_VARIABLES,env.action_space.n)

	ep_rewards = []
	fail_count = []

	with open(RESULTS_FILE, "w", newline="") as f:
		f.write("run,avg,max,min,fr,eps" + "\n")


	# cycle on EPISODES
	for episode in range(EPISODES):
		episode_reward = 0

		obs = env.reset()
		state = np.array(obs[0]).reshape(1,OBS_VARIABLES).tolist()[0]

		done = truncated = False  # has the enviroment finished?
		fail = 0 
		cnt = 0

		# cycle on STEPS till done
		while not (done or truncated):
			if episode % SHOW_EVERY == 0 and RENDER:
				env.render() 
			
			# Get action 
			if np.random.random() > epsilon:
				action = qTable.choose_action(state)
			else:
				action = np.random.randint(0, env.action_space.n)
			
			next_state, reward, done, truncated, info = env.step(action)  # perform action on enviroment
			
			next_state = np.array(next_state).reshape(1,OBS_VARIABLES).tolist()[0]
			qTable.learn(state, action, reward, next_state, done)

			if info["crashed"]:
				fail = 1

			episode_reward += reward
			state = next_state
			cnt += 1

		# Decaying is being done every episode if episode number is within decaying range
		if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
			epsilon -= epsilon_decay_value
			if epsilon < EPSILON_MIN:
				epsilon = EPSILON_MIN

		# Add new metrics for graph
		ep_rewards.append(episode_reward)
		fail_count.append(fail)
		if not episode % UPDATE_EVERY:
			average_reward = sum(ep_rewards[-UPDATE_EVERY:]) / UPDATE_EVERY
			average_fails = sum(fail_count[-UPDATE_EVERY:]) / UPDATE_EVERY
			print(f'Episode: {episode:>5d}, average fails: {average_fails:>4.1f}, average reward: {average_reward:>4.1f}, max reward: {max(ep_rewards[-UPDATE_EVERY:]):>4.1f}, min reward: {min(ep_rewards[-UPDATE_EVERY:]):>4.1f}, current epsilon: {epsilon:>1.2f}')
			with open(RESULTS_FILE, "a", newline="") as f:
				f.write(str(episode) + "," + str(average_reward) + "," + str(max(ep_rewards[-UPDATE_EVERY:])) + "," + str(min(ep_rewards[-UPDATE_EVERY:]))+ ","+ str(average_fails) + ","+ str(epsilon) + "\n")

		qTable._save_model("EGO_TRAIN_"+str(EPISODES)+"_"+str(now))

	env.close()