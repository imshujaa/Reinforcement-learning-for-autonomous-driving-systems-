
import gymnasium
import highway_env

from crl_lib.util import *
from crl_lib.query import *
from crl_lib.qtable import QTable


import csv
import pandas as pd
from sklearn.cluster import KMeans
from statistics import mean
import statistics

from datetime import datetime
import numpy as np
import time
import logging
import sys

# NUMBER OF REPETITIONS
REPETITIONS = 10

ENVIRONMENT = "highwayMA-v0" # highwayMA, intersectionMA

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
RESULTS_FILE = ""

########################	CAUSAL RL PARAMETERS

# NRS - plan after n real episodes
PLAN_AFTER = 1
# PA-B - breadth of the simulation (how many actions to plan on)
ACTIONS_TO_PLAN = 1
# PA-D - depth of the simulation (rollout length)
ROLLOUT_STEPS = 1

START_PLANNING_AT = 50
UPDATE_MODEL_EVERY = 100
SIMULATIONS = 20
NUM_CLUSTERS = 100

DISCOVERY = False
MEASURE_TIME = False

STOP_WITH_EPSILON = True

# number of episodes for smoothing rewards
SMOOTH_WINDOW = 50

# AUXILIARY (DO NOT CHANGE)
STOP_PLANNING = False
STOP_PLANNING_DERIVATIVE = False
GROWING = False
DECREASING = False


########################	Q-LEARNING PARAMETERS

# VARIABLES
EPISODES = 500   # Number of iterations episode
RENDER = True
SHOW_EVERY = 1  # How oftern the current solution is rendered
UPDATE_EVERY = 10  # How oftern the current progress is recorded

# Exploration settings
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 3
EPSILON_MIN = 0.1
EPSILON_DECAY_VALUE = START_EPSILON_DECAYING / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


########################	MODEL VARIABLES

DATASET_FILE = "crl_lib/dataset.csv"
DOT_FILE = "crl_lib/causal_graph.dot"
IMG_FILE = "crl_lib/causal_graph.svg"
GEN_IMG_ONCE = True
UPDATE_MODEL_SAMPLE = 1000000

columns = []
time_slices = []
required_arrows = []
forbidden_arrows = []

for timestep in range(ROLLOUT_STEPS+1):
	t_temp = []
	
	for observation in range(OBS_VARIABLES):
		columns.append("O"+str(observation)+"_"+str(timestep))
		t_temp.append("O"+str(observation)+"_"+str(timestep))
		for observation_ in range(OBS_VARIABLES):
			if observation != observation_:
				forbidden_arrows.append(["O"+str(observation)+"_"+str(timestep), "O"+str(observation_)+"_"+str(timestep)])

	if timestep > 0:
		columns.append("R_"+str(timestep-1))
		t_temp.append("R_"+str(timestep-1))

		for observation in range(OBS_VARIABLES):
			required_arrows.append(["O"+str(observation)+"_"+str(timestep-1), "O"+str(observation)+"_"+str(timestep)])
			required_arrows.append(["O"+str(observation)+"_"+str(timestep-1), "R_"+str(timestep-1)])
			required_arrows.append(["O"+str(observation)+"_"+str(timestep), "R_"+str(timestep-1)])
			required_arrows.append(["A_"+str(timestep-1), "O"+str(observation)+"_"+str(timestep)])
		required_arrows.append(["A_"+str(timestep-1), "R_"+str(timestep-1)])

	if timestep < ROLLOUT_STEPS:
		columns.append("A_"+str(timestep))
		t_temp.append("A_"+str(timestep))

	time_slices.append(t_temp)


if not DISCOVERY:
    print("I Generating graph")
    generate_dot(OBS_VARIABLES, ROLLOUT_STEPS+1, DOT_FILE)
    build_img(DOT_FILE,IMG_FILE)
	

########################	FUNCTIONS

def clear_logger():
	# Remove all existing handlers
	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)

def get_logger(repetition):
	logger = logging.getLogger("CRL")

	now = datetime.now()
	safe_timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
	log_file = 'output/LOGS/'+str(ENV_LABEL)+'/CRL/CRL_logs_r' + str(repetition) + '_' + safe_timestamp + '.log'
	logging.basicConfig(filename=log_file,format='%(asctime)s %(message)s')
	
	logger.setLevel(logging.DEBUG)
	logger.info("Started")
	return logger

if ENV_LABEL == "highway":
	ACTIONS_ALL = {
		0: 'LANE_LEFT',
		1: 'IDLE',
		2: 'LANE_RIGHT',
		3: 'FASTER',
		4: 'SLOWER'
	}
elif ENV_LABEL == "intersection":
	ACTIONS_ALL = {
		0: "SLOWER",
		1: "IDLE",
		2: "FASTER"
	}

# Create reverse lookup dictionary (action string to number)
ACTIONS_REVERSE = {v: k for k, v in ACTIONS_ALL.items()}

def number_to_action(number):
    """Convert a number to its corresponding action string using ACTIONS_ALL."""
    return ACTIONS_ALL.get(number, "UNKNOWN_ACTION")

def action_to_number(action):
    """Convert an action string to its corresponding number using ACTIONS_REVERSE."""
    return ACTIONS_REVERSE.get(action, -1)  # returns -1 if action not found


def select_transitions(all_values):
    to_simulate_transitions = []

    if len(all_values) > SIMULATIONS:
        orig_df = pd.DataFrame(all_values)
        df = orig_df.iloc[:, :OBS_VARIABLES].join(orig_df.iloc[:, -1:])
        df_to_cluster = df.iloc[:, :OBS_VARIABLES]

        kmeans = KMeans(n_clusters=NUM_CLUSTERS, n_init='auto')
        labels = kmeans.fit_predict(df_to_cluster.values)

        clusters = [df[labels == i].copy() for i in range(NUM_CLUSTERS)]
        avg_rewards = [cluster.iloc[:, OBS_VARIABLES].mean() for cluster in clusters]

        selected_indices = []

        for sim in range(SIMULATIONS):
            available_clusters = [
                (i, clusters[i], avg_rewards[i]) 
                for i in range(NUM_CLUSTERS) if not clusters[i].empty
            ]

            if not available_clusters:
                print("No more clusters available to sample from.")
                break

            cluster_indices, _, weights = zip(*available_clusters)
            chosen_cluster_idx = random.choices(cluster_indices, weights=weights, k=1)[0]

            random_index_in_cluster = clusters[chosen_cluster_idx].sample(n=1).index[0]
            random_row_in_cluster = orig_df.loc[random_index_in_cluster].values.tolist()

            clusters[chosen_cluster_idx].drop(index=random_index_in_cluster, inplace=True)

            to_simulate_transitions.append(random_row_in_cluster)
            selected_indices.append(random_index_in_cluster)

    else:
        to_simulate_transitions = all_values.copy()

    return to_simulate_transitions


########################	MAIN PROGRAM
with open("output/DATA/"+str(ENV_LABEL)+"/_cost/times_CRL.csv", "w", newline="") as f:
	f.write("rep,time\n")

for i in range(0, REPETITIONS):
	
	start_repetition = time.time()


	#ENVIRONMENT
	env = gymnasium.make(ENVIRONMENT,render_mode="rgb_array")

	# EGO AGENT
	ego_agent = QTable(OBS_VARIABLES,ACTION_VARIABLES)
	try:
		ego_agent._load_model(EGO_MODEL)
	except Exception as e:
		print(e)
		sys.exit()

	# ADVERSARIAL AGENT
	adv_agent = QTable(OBS_VARIABLES,ACTION_VARIABLES)

	ep_rewards = []
	ep_derivatives = []
	fail_count = []

	clear_logger()
	now = datetime.now()
	# Format the timestamp into a string safe for filenames
	safe_timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
	RESULTS_FILE = "output/DATA/"+str(ENV_LABEL)+"/CRL/CRL_r"+str(i+1)+"_"+safe_timestamp+".csv"
	with open(RESULTS_FILE, "w", newline="") as f:
		f.write("run,avg,max,min,fr,der,eps" + "\n")

	if MEASURE_TIME:
		planning_times = []
		fitting_times = []

	epsilon = 1
	STOP_PLANNING = False
	STOP_PLANNING_DERIVATIVE = False
	GROWING = False
	count_growing = 0
	DECREASING = False

	# cycle on EPISODES
	for episode in range(EPISODES):

		logger = get_logger(i+1)
		episode_reward = 0

		# RESET ENVIRONMENT
		obs = env.reset()
		states = tuple(np.array(obs_i.reshape(1,OBS_VARIABLES).tolist()[0]) for obs_i in obs[0])
		done = truncated = False
		fail = 0
		cnt = 0

		# epsilon-based plan termination logic
		if episode > END_EPSILON_DECAYING and STOP_WITH_EPSILON:
			STOP_PLANNING = True

		# CAUSAL MODEL INIT/UPDATE
		if episode == 0:
			df = pd.DataFrame(columns=columns)
			buffer_values = []
			all_values = []
			df.to_csv(DATASET_FILE, index=False)
		# Model update for the first time or every n episodes
		elif (episode == START_PLANNING_AT or episode % UPDATE_MODEL_EVERY == 0) and not STOP_PLANNING:
			with open(DATASET_FILE, "a", newline="") as f:
				writer = csv.writer(f)
				writer.writerows(buffer_values)
			buffer_values = []
			df_t = pd.read_csv(DATASET_FILE)

			if MEASURE_TIME:
				start = time.time()

			if DISCOVERY:
				model = init_model_cd(df_t)
			else:
				model = init_model(df_t, DOT_FILE)

			# print(evaluate_model(model,df_t))

			if MEASURE_TIME:
				end = time.time()
				fitting_times.append(end-start)

			if GEN_IMG_ONCE and DISCOVERY:
				build_img(DOT_FILE,IMG_FILE)
				GEN_IMG_ONCE = False

		# MAIN PLANNING PHASE
		if episode >= START_PLANNING_AT and episode % PLAN_AFTER == 0 and not STOP_PLANNING:

			if MEASURE_TIME:
				start = time.time()

			episode_values = select_transitions(all_values)

			for transition in episode_values:
				# Start state
				plan_state = transition
				plan_done = False
				
				for step in range(ROLLOUT_STEPS):
					all_actions_offset = 0
					
					observed_action = plan_state[OBS_VARIABLES]
					planned_actions = [observed_action]

					for breadth_lvl in range(ACTIONS_TO_PLAN):
						if step <= 1:
							state_var = columns[step*(OBS_VARIABLES+1):step*(OBS_VARIABLES+1)+OBS_VARIABLES]
							state_var.append(columns[step*(OBS_VARIABLES+1)+OBS_VARIABLES+step])
							next_state_var = columns[step*(OBS_VARIABLES+1)+OBS_VARIABLES+1+step:step*(OBS_VARIABLES+1)+2*OBS_VARIABLES+1+step]
							reward_var = columns[step*(OBS_VARIABLES+1)+2*OBS_VARIABLES+1+step]
						else:
							start_index = columns.index(next_state_var[0])
							state_var = columns[start_index:start_index+OBS_VARIABLES]
							state_var.append(columns[start_index+OBS_VARIABLES+1])
							start_index = start_index+OBS_VARIABLES+2
							next_state_var = columns[start_index:start_index+OBS_VARIABLES]
							reward_var = columns[start_index+OBS_VARIABLES]

						observed_state = []
						for variable in state_var:
							if 'O' in variable:
								observed_state.append(plan_state[columns.index(variable)])
						
						# RANDOM ACTION SELECTION
						plan_action = number_to_action(np.random.randint(0, env.action_space[0].n))
						while plan_action in planned_actions:
							plan_action = number_to_action(np.random.randint(0, env.action_space[0].n))
						planned_actions.append(plan_action)

						# COUNTERFACTUAL
						sim_samples = counterfactual(model, columns, plan_state, state_var[len(state_var)-1], plan_action)
						sim_samples = sim_samples[columns]
						plan_next_state = list(sim_samples[next_state_var].mean())
						plan_done = 0
						plan_reward = round(float(sim_samples[reward_var].mean()), 3)
						
						if plan_reward >= 1:
							plan_reward = 1
							plan_done = 1
						elif plan_reward < 0:
							plan_reward = 0
						
						plan_action = action_to_number(plan_action)
						adv_agent.learn(observed_state, plan_action, plan_reward, plan_next_state, plan_done)


				if MEASURE_TIME:
					end = time.time()
					planning_times.append(end-start)

		# MAIN EPISODE CYCLE
		action_values = []
		episode_values = []
		action_count = 0
		while not (done or truncated):
			if episode % SHOW_EVERY == 0 and RENDER:
				env.render() 

			# ACTIONS
			actions = []
			actions.append(ego_agent.choose_action(states[0]))
			if np.random.random() > epsilon:
				actions.append(adv_agent.choose_action(states[1]))
			else:
				actions.append(np.random.randint(0, env.action_space[0].n))

			
			if action_count == 0:
				for value in states[1]:  
					action_values.append(value)
			action_values.append(number_to_action(actions[1]))

			# STEP
			next_states, rewards, done, truncated, info = env.step((actions[0],actions[1]))  # perform action on enviroment
			next_states = tuple(np.array(obs_i.reshape(1,OBS_VARIABLES).tolist()[0]) for obs_i in next_states)			
			
			# LEARN
			adv_agent.learn(states[1], actions[1], rewards[1], next_states[1], done)

			if info["crashed"]:
				fail = 1

			for value in next_states[1]:  
				action_values.append(value)
			action_values.append(rewards[1])

			if action_count == ROLLOUT_STEPS-1 or done:
				# In case we end up before the end of the model length, we copy the last observation for the rest of the row
				if action_count < ROLLOUT_STEPS-1 and done:
					temp_values = action_values[(len(action_values))-(OBS_VARIABLES+2):(len(action_values))]
					while action_count < ROLLOUT_STEPS-1:
						for obs_var in range(OBS_VARIABLES+2):
							action_values.append(temp_values[obs_var])
						action_count += 1
				
				buffer_values.append(action_values)
				all_values.append(action_values)
				episode_values.append(action_values)
				action_values = []
				action_count = 0
			else:
				action_count += 1

			logger.info(str(states[1].tolist()) + "#" + str(actions[1]) + "#" + str(rewards[1]))
			episode_reward += rewards[1]
			cnt += 1
			states = next_states


		# EPSILON
		if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
			epsilon -= EPSILON_DECAY_VALUE
			if epsilon < EPSILON_MIN:
				epsilon = EPSILON_MIN

		# AVG REWARD COMPUTATION
		episode_reward = episode_reward/cnt
		ep_rewards.append(episode_reward)
		fail_count.append(fail)

		# Check if sufficient data to compute smoothed derivative
		if len(ep_rewards) >= SMOOTH_WINDOW:
			dy_dx = np.gradient(ep_rewards[-SMOOTH_WINDOW:])			
			avg_recent_derivative = np.mean(dy_dx)
			ep_derivatives.append(avg_recent_derivative)

		else:
			ep_derivatives.append(0)

		if not episode % UPDATE_EVERY:
			average_reward = sum(ep_rewards[-UPDATE_EVERY:]) / UPDATE_EVERY
			average_fails = sum(fail_count[-UPDATE_EVERY:]) / UPDATE_EVERY
			average_derivatives = sum(ep_derivatives[-UPDATE_EVERY:]) / UPDATE_EVERY
			print(f'Episode: {episode:>5d}, average fails: {average_fails:>4.1f}, average reward: {average_reward:>4.2f}, average derivative: {average_derivatives:>4.4f}, max reward: {max(ep_rewards[-UPDATE_EVERY:]):>4.2f}, min reward: {min(ep_rewards[-UPDATE_EVERY:]):>4.2f}, current epsilon: {epsilon:>1.1f}')
			with open(RESULTS_FILE, "a", newline="") as f:
				f.write(str(episode) + "," + str(average_reward) + "," + str(max(ep_rewards[-UPDATE_EVERY:])) + "," + str(min(ep_rewards[-UPDATE_EVERY:]))+ ","+ str(average_fails) + ","+ str(average_derivatives) + ","+ str(epsilon) + "\n")

		# SAVE MODELS
		adv_agent._save_model("ADV_CRL_"+str(now))

		if MEASURE_TIME:
			if len(fitting_times) > 0:
				print("Fitting times: " + str(mean(fitting_times)))
			if len(planning_times) > 0:
				print("Planning times: " + str(mean(planning_times)))

	#save recorded times to file
	if MEASURE_TIME:
		with open("DATA/times.csv", "a", newline="") as f:
			f.write(str(mean(fitting_times)) + "," + str(mean(planning_times)) + "\n")
	

	end_repetition = time.time()
	elapsed_time = end_repetition-start_repetition
	with open("output/DATA/"+str(ENV_LABEL)+"/_cost/times_CRL.csv", "a", newline="") as f:
		f.write(str(i) + "," + str(elapsed_time) + "\n")

	env.close()