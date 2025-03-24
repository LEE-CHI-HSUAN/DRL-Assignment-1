# Remember to adjust your student ID in meta.xml

import torch
from TaxiMemory import TaxiMemory
from utils import get_state_tensor
from dqn_net import DQN

device = "cpu"


def interactive():
    action = int(input("action(0-5): "))
    return action


policy_net = DQN(23, 6).to(device)
policy_net.load_state_dict(torch.load("DQN.pt", map_location=device))


def dqn_play(state):
    with torch.no_grad():
        action = policy_net(state).max(1).indices.item()
        return action


taxi_memory = TaxiMemory()
last_action = None


def get_action(obs):
    global last_action

    if last_action is None:
        taxi_memory.reset(obs)
        state = get_state_tensor(obs, taxi_memory, device)
    else:
        state = get_state_tensor(obs, taxi_memory, device, last_action)
    print(state)

    # return interactive()
    # last_action = gp_play(state)
    last_action = dqn_play(state)
    return last_action
