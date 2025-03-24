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
