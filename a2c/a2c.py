import gym
import logging
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import requests

print("TensorFlow Ver: ", tf.__version__)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.preprocessing.sequence as sequence
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Concatenate
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from tensorflow.keras import backend as K

from gumball_requester import *

TARGET_SELF = 0

class A2CAgent:
  def __init__(self, model, lr=7e-3, gamma=0.99, value_c=0.5, entropy_c=1e-4):
    # `gamma` is the discount factor; coefficients are used for the loss terms.
    self.gamma = gamma
    self.value_c = value_c
    self.entropy_c = entropy_c
    self.lr = lr

    self.model = model
    self.model.compile(
      optimizer=ko.RMSprop(lr=lr),
      # Define separate losses for policy logits and value estimate.
      loss=[self._logits_loss, self._value_loss])

  def train(self, env, batch_sz=64, updates=250, use_network=False):
    # Storage helpers for a single batch of data.
    actions = np.empty((batch_sz,), dtype=np.int32)
    rewards, dones, values = np.empty((3, batch_sz))
    observations = np.empty((batch_sz,) + env.observation_space.shape)
    # Training loop: collect samples, send to optimizer, repeat updates times.
    ep_rewards = [0.0]
    winners = []
    next_obs, _, _, info = env.reset()
    for update in range(updates):
      for step in range(batch_sz):
        observations[step] = next_obs.copy()
        actions[step], values[step] = self.model.action_value(next_obs[None, :], info['valid_actions'])
        if info['valid_actions'][actions[step]] == 0:
            print('invalid choice made')
            print('valid_moves', info['valid_actions'])
            b=1/0


        try:
          next_obs, rewards[step], dones[step], info = env.step(actions[step])
        except requests.Timeout:
          # back off and retry
          print('timeout step request force ending session')
          next_obs, _, _, info = env.reset()
          dones[step] = True
          rewards[step] = 0.0

        ep_rewards[-1] += rewards[step]
        if dones[step]:
          ep_rewards.append(0.0)
          next_obs, _, _, info = env.reset()
          logging.info("Episode: %0.3f, Reward: %0.3f " % (len(ep_rewards) - 1, ep_rewards[-2]))

      _, next_value = self.model.action_value(next_obs[None, :], info['valid_actions'])
      returns, advs = self._returns_advantages(rewards, dones, values, next_value)
      # A trick to input actions and advantages through same API.
      acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
      # Performs a full training step on the collected batch.
      # Note: no need to mess around with gradients, Keras API handles it.
      print("training on batches")
      losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
      logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))

    return ep_rewards, winners

  def test(self, env, render=False, use_network=False):
    obs, ep_reward, done, info = env.reset()
    while not done:

        action, _ = self.model.action_value(obs[None, :], info['valid_actions'])
        try:
            obs, reward, done, info = env.step(action)
        except requests.Timeout:
            # back off and retry
            print('timeout step request force ending session')
            done = True
        ep_reward += reward
        if render:
            env.render()
    return ep_reward

  def _returns_advantages(self, rewards, dones, values, next_value):
    # `next_value` is the bootstrap value estimate of the future state (critic).
    returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
    # Returns are calculated as discounted sum of future rewards.
    for t in reversed(range(rewards.shape[0])):
      returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
    returns = returns[:-1]
    # Advantages are equal to returns - baseline (value estimates in our case).
    advantages = returns - values
    return returns, advantages

  def _value_loss(self, returns, value):
    # Value loss is typically MSE between value estimates and returns.
    return self.value_c * kls.mean_squared_error(returns, value)

  def _logits_loss(self, actions_and_advantages, logits):
    # A trick to input actions and advantages through the same API.
    actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
    # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
    # `from_logits` argument ensures transformation into normalized probabilities.
    weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
    # Policy loss is defined by policy gradients, weighted by advantages.
    # Note: we only calculate the loss on the actions we've actually taken.
    actions = tf.cast(actions, tf.int32)
    policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
    # Entropy loss can be calculated as cross-entropy over itself.
    probs = tf.nn.softmax(logits)
    entropy_loss = kls.categorical_crossentropy(probs, probs)
    # We want to minimize policy and maximize entropy losses.
    # Here signs are flipped because the optimizer minimizes.
    return policy_loss - self.entropy_c * entropy_loss
