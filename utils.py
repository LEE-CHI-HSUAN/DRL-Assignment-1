import random
import torch
from TaxiMemory import TaxiMemory


def select_action(
    policy_net: torch.nn.Module,
    state: torch.Tensor,
    device: torch.DeviceObjType,
    eps_threshold: float = 0,
) -> torch.Tensor:
    """Selects an action based on an epsilon-greedy strategy.

    Args:
        policy_net (torch.Module): The DQN agent.
        state (torch.Tensor): The current state of the environment.
        eps_threshold (float, optional): The probability of selecting a random action. Defaults to 0.

    Returns:
        torch.Tensor: The selected action as a tensor of shape (1, 1).
    """
    if random.random() < eps_threshold:
        return torch.randint(0, 6, (1, 1), dtype=torch.long, device=device)
    else:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)


def get_state_tensor(
    state: tuple,
    taxi_memory: TaxiMemory,
    device: torch.DeviceObjType,
    action: int = None,
) -> torch.Tensor:
    if action is not None:
        kioku = taxi_memory.update(state, action)
    else:
        kioku = taxi_memory.get_state(state)

    # vectors from stations
    taxi_x, taxi_y = state[0], state[1]
    station_dirs = []
    for i in range(2, 10, 2):
        station_dirs.append(state[i] - taxi_x)
        station_dirs.append(state[i + 1] - taxi_y)

    state = (*station_dirs, *kioku, *(state[10:14]))
    return torch.tensor([state], dtype=torch.float32, device=device)


def env_jump_before_pick(env, taxi_memory, k=1):
    """
    k=0-3
    """
    if k == 0:
        return env.get_state()
    dest_idx = env.stations.index(env.destination)
    locations_idx = list(range(4))  # all
    locations_idx.remove(env.stations.index(env.passenger_loc))  # remove goal
    locations_idx = random.sample(locations_idx, k)  # shrink to k

    env.taxi_pos = env.stations[random.choice(locations_idx)]

    for idx in locations_idx:
        taxi_memory.visit_mask[idx] = 0
    if dest_idx in locations_idx:
        taxi_memory.destination_mask[dest_idx] = 1
    return env.get_state()


def env_jump_after_pick(env, taxi_memory, k=1):
    """
    k=0-3
    """
    if k == 0:
        env.get_state()
    dest_idx = env.stations.index(env.destination)
    passenger_idx = env.stations.index(env.passenger_loc)
    locations_idx = list(range(4))  # all
    locations_idx.remove(passenger_idx)  # remove passenger and add it back later
    locations_idx = random.sample(locations_idx, k - 1)  # shrink to k-1
    locations_idx.append(passenger_idx)

    env.taxi_pos = env.passenger_loc
    env.step(4)

    taxi_memory.passenger_picked_up = True
    for idx in locations_idx:
        taxi_memory.visit_mask[idx] = 0
    if dest_idx in locations_idx:
        taxi_memory.destination_mask[dest_idx] = 1
    return env.get_state()
