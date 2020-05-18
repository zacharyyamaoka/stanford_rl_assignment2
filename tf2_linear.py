import tensorflow as tf
from tensorflow import keras
import numpy as np
import datetime
from collections import deque
from utils.replay_buffer import ReplayBuffer


class Linear():

    def __init__(self, replay_buffer_len=5000, learning_rate=0.001, espilon=1, verbose=0):
        self.gamma = 1
        self.epsilon = espilon
        self.min_epsilon = 0.1
        self.delta_epsilon = 0.025
        self.epsilon_update_freq = 5000

        self.lr = learning_rate
        self.target_update_freq = 250
        self.Optimizer = keras.optimizers.RMSprop(
            learning_rate=self.lr)

        self.model = keras.Sequential(
            [
                keras.layers.Flatten(input_shape=[5, 5, 1]),
                keras.layers.Dense(
                    units=5, name="layer1"),
            ]
        )

        self.replay_buffer = ReplayBuffer(
            size=replay_buffer_len, frame_history_len=1)
        self.mini_batch_size = 32

        if verbose > 0:
            self.model.summary()

        self.target_model = keras.models.clone_model(self.model)
        self.update_target_weights()

        self.update_steps = 0
        self.total_reward = 0

    def greedy_policy(self, curr_state):  # Espilon Greedy
        random_val = np.random.uniform()

        if random_val < 0:
            action = np.random.randint(5)
            action_values = np.zeros(5)

        else:
            curr_state = self.preprocess(curr_state)
            action_values = self.model(curr_state).numpy()
            action = np.argmax(action_values)

        return action, action_values

    def policy(self, curr_state):  # Espilon Greedy

        random_val = np.random.uniform()

        if random_val < self.epsilon:
            action = np.random.randint(5)
            action_values = np.zeros(5)

        else:
            curr_state = self.preprocess(curr_state)
            action_values = self.model(curr_state).numpy()
            action = np.argmax(action_values)

        return action, action_values

    def update_epsilon(self):
        self.epsilon -= self.delta_epsilon
        self.epsilon = max(self.epsilon, self.min_epsilon)

    def update_target_weights(self):
        self.target_model.set_weights(self.model.get_weights())

    def schedule(self, step):

        if step % self.epsilon_update_freq == 0:
            self.update_epsilon()
            print("UPDATING EPSILON: ", self.epsilon, "step: ", step)

        if step % self.target_update_freq == 0:
            self.update_target_weights()
            print("UPDATING TARGET WEIGHTS step: ", step)

    def preprocess(self, state):
        """
        Change image values for better numerical properties:
        1. Mean center
        2. Normalize between for max 1
        """

        if len(state.shape) == 3:
            state = np.expand_dims(state, axis=0)

        B = state.shape[0]
        W = state.shape[1]
        H = state.shape[2]
        D = state.shape[3]

        state = state - np.mean(state, axis=(1, 2)
                                ).reshape(B, 1, 1, D)

        state = state / np.amax(state, axis=(1, 2)
                                ).reshape(B, 1, 1, D)

        state = state.astype("float32")
        return state

    def replay_fill(self, curr_state, action, reward, next_state, done):
        # store next_state, on the next_update to avoid dupilcation
        # curr_state_processed = self.preprocess(curr_state)
        idx = self.replay_buffer.store_frame(curr_state)
        self.replay_buffer.store_effect(idx, action, reward, done)

    def replay_update(self):

        self.update_steps += 1

        # Sample mini-batch of experince from buffer
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = \
            self.replay_buffer.sample(self.mini_batch_size)

        obs_batch = self.preprocess(obs_batch)
        next_obs_batch = self.preprocess(next_obs_batch)

        with tf.GradientTape() as tape:

            action_values = self.model(obs_batch)  # (batch_size, 5)
            pred_value = tf.gather(action_values, act_batch, axis=1)

            target_value = rew_batch
            future_value = self.gamma * \
                np.amax(self.target_model(next_obs_batch).numpy(), axis=1)
            target_value[np.logical_not(
                done_mask)] += future_value[np.logical_not(done_mask)]

            loss = tf.math.reduce_mean(0.5 * (target_value - pred_value)**2)

        gradients = tape.gradient(loss, self.model.trainable_weights)

        self.Optimizer.apply_gradients(
            zip(gradients, self.model.trainable_weights))

        self.schedule(self.update_steps)

        return loss.numpy()

    def update(self, curr_state, action, reward, next_state, done):

        self.update_steps += 1

        with tf.GradientTape() as tape:
            curr_state = self.preprocess(curr_state)
            next_state = self.preprocess(next_state)

            if len(curr_state.shape) == 3:
                curr_state = np.expand_dims(curr_state, axis=0)
                next_state = np.expand_dims(next_state, axis=0)

            curr_state = np.expand_dims(curr_state, axis=0)
            action_values = self.model(curr_state)
            pred_value = action_values[:, action]

            target_value = reward
            if not done:
                target_value += self.gamma * \
                    tf.math.reduce_max(self.target_model(next_state)).numpy()

            loss = tf.math.reduce_sum(0.5 * (target_value - pred_value)**2)

        gradients = tape.gradient(loss, self.model.trainable_weights)
        # processed_graxwds = [tf.clip_by_norm(g, 1) for g in gradients]

        self.Optimizer.apply_gradients(
            zip(gradients, self.model.trainable_weights))

        self.schedule(self.update_steps)

        return loss.numpy()
