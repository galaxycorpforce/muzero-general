# %load pokemon_shadow_env/poke_env.py
import gym
import gym.spaces
from gym.utils import seeding
import numpy as np
import json
import requests
import os
from random import randint
import random
from time import sleep

#SERVER_URL = os.environ.get('SERVER_URL', 'http://localhost:12231/api/')
SERVER_URL = os.environ.get('SERVER_URL', 'http://galaxytesty.pythonanywhere.com/api/')
class GumballRequester(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, server_url=SERVER_URL):
        self.server_url = server_url
        self.action_space = gym.spaces.Discrete(n=5)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(10,), dtype=np.float32)
        self.bot_id=None
        sleep(random.random() * randint(1,10))
        print('Grabbing Bot Id to use')
        self.get_available_bot()

    def get_params(self):
        return {
            'bot_id': self.bot_id
        }

    def fire_request(self, url, params):
        url = self.server_url + url
        data_json = json.dumps(params)
#        print('url:', url)
        headers = {'content-type': 'application/json'}

        response = requests.post(url, data=data_json, headers=headers, timeout=15)
#        print('response', response)
        return response.json()

    # Checks for least recently used bot to avoid clashes
    def get_available_bot(self):
        params = self.get_params()
        url = 'get_next_bot_id'
        raw_resp = self.fire_request(url, params)
        self.bot_id = raw_resp['bot_id']


    def step(self, action):
        action = int(action)
        params = self.get_params()
        params['action'] = action
        url = 'step'
        raw_resp = self.fire_request(url, params)
        obs, reward, done, info = np.asarray(raw_resp['obs']), raw_resp['reward'], raw_resp['done'], raw_resp['info']

        return obs, reward, done, info

    def reset(self):
        params = self.get_params()

        url = 'reset'
        raw_resp = self.fire_request(url, params)
        obs, reward, done, info = np.asarray(raw_resp['obs']), raw_resp['reward'], raw_resp['done'], raw_resp['info']
        return obs, reward, done, info


    def sample_actions(self):
        params = self.get_params()
        url = 'get_sample_action'
        resp = self.fire_request(url, params)
        return resp['action']

    def reward(self, reward):
        return reward * 0.01
