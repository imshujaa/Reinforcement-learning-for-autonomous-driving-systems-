from dowhy import gcm
import numpy as np, pandas as pd
from dowhy.gcm.uncertainty import estimate_variance
from dowhy.gcm.falsify import falsify_graph
import dowhy.gcm as gcm
from causallearn.search.FCMBased import lingam


import networkx as nx
import pandas as pd
import pydot
from IPython.display import SVG
import sys
import random

# from pycausal.pycausal import pycausal as pc
# from pycausal import search as s
# from pycausal import prior as p
# pc = pc()

# intervention_var = 'ts-security-service_f'

def init_model_pk(df_in, dot_file, prior_knowledge, prior_knowledge_f, prior_knowledge_t):
    
    # pc.start_vm(java_max_heap_size = "8192m")

    # # prior = p.knowledge(requiredirect=prior_knowledge)
    # prior = p.knowledge(forbiddirect = prior_knowledge_f, requiredirect=prior_knowledge, addtemporal = prior_knowledge_t)
    # # prior = p.knowledge(requiredirect=prior_knowledge, addtemporal = prior_knowledge_t)

    # tetrad = s.tetradrunner()
    # # tetrad.run(algoId = 'fci', dfs = df_in, testId = 'fisher-z-test', 
    # #     depth = -1, maxPathLength = -1, 
    # #     completeRuleSetUsed = False, verbose = False, priorKnowledge=prior)
    # tetrad.run(algoId = 'fges', dfs = df_in, scoreId = 'cg-bic-score', 
    #     dataType = 'discrete', numCategoriesToDiscretize = 20,
    #     maxDegree = 3, faithfulnessAssumed = True, 
    #     numberResampling = 50, resamplingEnsemble = 2, addOriginalDataset = True, verbose = False, priorKnowledge=prior)
    # # tetrad.run(algoId = 'gfci', dfs = df_in, testId = 'disc-bic-test', scoreId = 'bdeu-score', 
    # #        priorKnowledge = prior, dataType = 'discrete',
    # #        maxDegree = 3, maxPathLength = -1, 
    # #        completeRuleSetUsed = False, faithfulnessAssumed = True, verbose = True,
    # #        numberResampling = 5, resamplingEnsemble = 1, addOriginalDataset = True)
    
    # # tetrad.run(algoId = 'pc-all', dfs = df_in, testId = 'fisher-z-test', priorKnowledge = prior, 
    # #        fasRule = 1, depth = -1, conflictRule = 1, concurrentFAS = True,
    # #        useMaxPOrientationHeuristic = False, verbose = True)

    # #salvo modello dot
    # dot = pc.tetradGraphToDot(tetrad.getTetradGraph())
    # original_stdout = sys.stdout
    # with open(dot_file, 'w') as res:
    #     sys.stdout = res
    #     print(dot)
    #     sys.stdout = original_stdout 


    # causal_model = nx.DiGraph(nx.nx_pydot.read_dot(dot_file)) 
    # nodes = causal_model.nodes
    # graph = causal_model

    # causal_model = gcm.InvertibleStructuralCausalModel(causal_model)
    # # causal_model = gcm.StructuralCausalModel(causal_model)
    # data = df_in

    # gcm.auto.assign_causal_mechanisms(causal_model, data)
    # gcm.fit(causal_model, data)

    # # pc.stop_vm()
    # # return causal_model, nodes, graph
    # return causal_model
    return 0


def init_model_cd(df_in):
    causal_model = lingam.DirectLiNGAM()
    causal_model.fit(df_in)

    print(causal_model.causal_order_)
    print(causal_model.adjacency_matrix_)

    return causal_model

def init_model(df_in,dot_file):

    causal_model = nx.DiGraph(nx.nx_pydot.read_dot(dot_file)) 
    nodes = causal_model.nodes
    graph = causal_model
    causal_model = gcm.InvertibleStructuralCausalModel(causal_model)

    data = df_in

    gcm.auto.assign_causal_mechanisms(causal_model, data)

    gcm.fit(causal_model, data)

    # return causal_model, nodes, graph
    return causal_model


def query_model(causal_model, dictionary):    

    samples = gcm.interventional_samples(causal_model,
                                        dictionary,
                                        num_samples_to_draw=1000)
    return samples

def query_model_counterfactual(causal_model, dictionary_act, dictionary_obs):    
    
    samples = gcm.counterfactual_samples(causal_model,
    dictionary_act,
    observed_data=pd.DataFrame(data=dictionary_obs))

    return samples

def select_action_rollout(model, initial_step, actual_step, observation, action_space, max_steps, resolution):
    if actual_step < max_steps and actual_step < (initial_step + resolution):
        # print("Actual step: " + str(actual_step))
        variables  = ['cart_position', 'cart_velocity', 'pole_angle', 'pole_angular_velocity', 'action']
        angles = []
        velocities = []


        for action in range(action_space):
            # print("Rollout action: " + str(action))
            temp_obs = observation
            temp_obs.append(action)

            samples = multi_intervention(model, variables, temp_obs)
            # samples = counterfactual(model, variables, temp_obs)
            new_observation = [samples["cart_position_n"].mean(), samples["cart_velocity_n"].mean(), samples["pole_angle_n"].mean(), samples["pole_angular_velocity_n"].mean()]
            reward = abs(samples["pole_angle_n"].mean()) + (((initial_step + resolution) - actual_step)/resolution)*abs(select_action_rollout(model, initial_step, actual_step+1, new_observation, action_space, max_steps, resolution))
            # print("Reward action: " + str(reward))
            angles.append(reward)
            # velocities.append(abs(samples["pole_angular_velocity_n"].mean()))

        if actual_step == initial_step:
            print("Initial step decision")

            if all(elem == angles[0] for elem in angles):
                print("I Selecting random action")
                action = random.randint(0,action_space-1)
            else:
                action_a = 0
                # action_v = 0
                
                min_a = abs(angles[0])
                # min_v = abs(velocities[0])
                for index in range(len(angles)):
                    if abs(angles[index]) < min_a:
                        min_a = angles[index]
                        action_a = index
                    # if abs(velocities[index]) < min_v:
                    #     min_v = velocities[index]
                    #     action_v = index
            
            # if initial_step%3 == 0:
            #     action = action_v
            # else:
                action = action_a

            print("Selected Action: " + str(action))
            return action
        else:
            print("Actual step: " + str(actual_step) + " angles: " + str(angles))
            return min(angles)
    else:
        return 0


def select_action(model, step, observation, action_space, max_steps):
    rewards = []
    angles = []

    var = ['cart_position', 'cart_velocity', 'pole_angle', 'pole_angular_velocity', 'action']
    print("------------------------------------------")
    for action in range(action_space):

        temp_obs = observation
        temp_obs.append(action)

        samples = multi_intervention(model, var, temp_obs)
        
        # value = rollout(samples, 15, step, max_steps)
        value = samples["pole_angle_n"].mean()
        angles.append(value)
        print(value)
    
    if all(elem == angles[0] for elem in angles):
        print("I Selecting random action")
        action = random.randint(0,action_space-1)
    else:
        action = 0
        
        min_a = abs(angles[0])
        for index in range(len(angles)):
            if abs(angles[index]) < min_a:
                min_a = angles[index]
                action = index

    print("Action: " + str(action))
    return action

def rollout(samples, resolution, step, max_steps):
    index = step
    value = 0
    weight = resolution
    # denom = 10

    while index < max_steps and index < (step + resolution):
        value += weight/resolution * samples["R_"+str(index)].mean()
        weight -= 1
        index += 1

    return value

def multi_intervention(causal_model,variables,values):
    dictionary = {}

    # var_index = 0
    value_0 = values[0]
    value_1 = values[1]
    value_2 = values[2]
    value_3 = values[3]
    value_4 = values[4]

    temp_dict = dict({variables[0]: lambda y: value_0})
    dictionary.update(temp_dict)
    temp_dict = dict({variables[1]: lambda y: value_1})
    dictionary.update(temp_dict)
    temp_dict = dict({variables[2]: lambda y: value_2})
    dictionary.update(temp_dict)
    temp_dict = dict({variables[3]: lambda y: value_3})
    dictionary.update(temp_dict)
    temp_dict = dict({variables[4]: lambda y: value_4})
    dictionary.update(temp_dict)

    # for variable in variables:
    #     # value = values[var_index]
    #     temp_dict = dict({variable: lambda y: var_index})
    #     dictionary.update(temp_dict)
    #     # print(dictionary.items())
    #     var_index += 1

    samples = query_model(causal_model, dictionary)

    return samples

def counterfactual(causal_model,variables,values, var_cf, val_cf):
    dictionary_act = {}
    dictionary_obs = {}

    for index in range(len(values)):
        temp_dict = dict({variables[index]: [values[index]]})
        dictionary_obs.update(temp_dict)

    temp_dict = dict({var_cf: lambda y: val_cf})
    dictionary_act.update(temp_dict)

    samples = query_model_counterfactual(causal_model, dictionary_act, dictionary_obs)

    return samples


def evaluate_model(causal_model,data):
    return gcm.evaluate_causal_model(causal_model, data, evaluate_causal_mechanisms=True, evaluate_invertibility_assumptions=True)


def intervention(causal_model,variable,value):
    dictionary = {}
    temp_dict = dict({variable: lambda y: value})
    dictionary.update(temp_dict) 
    samples = query_model(causal_model, dictionary)

    return samples