import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os 

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

RUN_ON_CPU = True


class DQN:
    def __init__(self, state_size, action_size, _learning_rate=0.01, _discount=0.99, _batch_size=32, _buffer_size=2000, _gradient_steps=1, env_label="highway"):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=_buffer_size)
        self.gamma = _discount  
        self.learning_rate = _learning_rate
        self.batch_size = _batch_size
        self.gradient_steps = _gradient_steps
        self.env_label = env_label  # Store environment label for dynamic paths
        self.model = self._build_model()
        self.model_t = self._build_target_model()
        

    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.state_size*2, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.state_size*2, activation='relu'))
        model.add(Dense(self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        if RUN_ON_CPU:
            with tf.device("cpu:0"):
                model.compile(loss='mse',optimizer=Adam(learning_rate=self.learning_rate))
        else:
            model.compile(loss='mse',optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def _build_target_model(self):
        model = self._build_model()
        if RUN_ON_CPU:
            with tf.device("cpu:0"):
                model.set_weights(self.model.get_weights()) 
        else:
            model.set_weights(self.model.get_weights())
        return model

    def _save_model(self, obj, descr):
        models_dir = os.path.join("models", "ADV", self.env_label, "DQN")
        os.makedirs(models_dir, exist_ok=True)
        filename = f"model_o{obj}_{descr}.weights.h5"
        filepath = os.path.join(models_dir, filename)
        self.model.save_weights(filepath)

    def _load_model(self, obj, descr):
        models_dir = os.path.join("models", "ADV", self.env_label, "DQN")
        filename = f"model_o{obj}_{descr}.weights.h5"
        filepath = os.path.join(models_dir, filename)
        self.model.load_weights(filepath)
        self._build_target_model()

    def memorize(self, state, action, reward, next_state, done):
        if len(state.shape) > 1:
            state = state.flatten()
        if len(next_state.shape) > 1:
            next_state = next_state.flatten()
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        if RUN_ON_CPU:
            with tf.device("cpu:0"):
                act_values = self.model.predict(state, verbose=0)
        else:
            act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  
    
    def replay(self):
        for gradient_step in range(self.gradient_steps):
            minibatch = random.sample(self.memory, self.batch_size)
            state, action, reward, next_state, done = zip(*minibatch)

            state = np.vstack(state)
            next_state = np.vstack(next_state)
            action = np.array(action)
            reward = np.array(reward)[:, None]
            done = np.array(done)[:, None]

            if RUN_ON_CPU:
                with tf.device("cpu:0"):
                    current_state_q_values = self.model.predict(state, verbose=0)
                    next_state_q_values = self.model_t.predict(next_state, verbose=0)
            else:
                current_state_q_values = self.model.predict(state, verbose=0)
                next_state_q_values = self.model_t.predict(next_state, verbose=0)                

            q_learning_targets = reward + self.gamma * np.max(next_state_q_values, axis=1, keepdims=True) * (1-done)

            batch_indices = np.arange(self.batch_size)
 
            current_state_q_values[batch_indices, action] = q_learning_targets.flatten()

            if RUN_ON_CPU:
                with tf.device("cpu:0"):
                    self.model.fit(state, current_state_q_values, verbose=0)
            else:
                self.model.fit(state, current_state_q_values, verbose=0)

    def update_target(self):
        if RUN_ON_CPU:
            with tf.device("cpu:0"):
                self.model_t.set_weights(self.model.get_weights())
        else:
            self.model_t.set_weights(self.model.get_weights())

    def replay_single(self, state, action, reward, next_state):
        target = reward
        if reward < 1000000:
            target = (reward + self.gamma *
                        np.amax(self.model_t.predict(next_state)[0]))
            
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

    
