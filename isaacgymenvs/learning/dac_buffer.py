# Stuff added by Dhruv Sreenivas for Discriminator Actor-Critic in the IsaacGym ILFO suite
import torch

def double_shape(shape):
    '''double everything in the shape tuple, return a tuple'''
    return tuple([2 * elt for elt in shape])

class AMPVectorizedReplayBuffer:
    '''AMP vectorized replay buffer to handle AMP observations as well.'''
    
    def __init__(self, obs_shape, action_shape, capacity, device):
        """Create Vectorized Replay buffer (off policy AMP)
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        See Also
        --------
        ReplayBuffer.__init__
        """

        self.device = device

        self.obses = torch.empty((capacity, *obs_shape), dtype=torch.float32, device=self.device)
        self.next_obses = torch.empty((capacity, *obs_shape), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((capacity, *action_shape), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        self.dones = torch.empty((capacity, 1), dtype=torch.bool, device=self.device)
        self.amp_obses = torch.empty((capacity, *double_shape(obs_shape)), dtype=torch.float32, device=self.device)

        self.capacity = capacity
        self.idx = 0
        self.full = False
        

    def add(self, obs, action, reward, next_obs, done, amp_obs):

        num_observations = obs.shape[0]
        remaining_capacity = min(self.capacity - self.idx, num_observations)
        overflow = num_observations - remaining_capacity
        if remaining_capacity < num_observations:
            self.obses[0: overflow] = obs[-overflow:]
            self.actions[0: overflow] = action[-overflow:]
            self.rewards[0: overflow] = reward[-overflow:]
            self.next_obses[0: overflow] = next_obs[-overflow:]
            self.dones[0: overflow] = done[-overflow:]
            self.amp_obses[0: overflow] = amp_obs[-overflow:]
            self.full = True
        self.obses[self.idx: self.idx + remaining_capacity] = obs[:remaining_capacity]
        self.actions[self.idx: self.idx + remaining_capacity] = action[:remaining_capacity]
        self.rewards[self.idx: self.idx + remaining_capacity] = reward[:remaining_capacity]
        self.next_obses[self.idx: self.idx + remaining_capacity] = next_obs[:remaining_capacity]
        self.dones[self.idx: self.idx + remaining_capacity] = done[:remaining_capacity]
        self.amp_obses[self.idx: self.idx + remaining_capacity] = amp_obs[:remaining_capacity]

        self.idx = (self.idx + num_observations) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obses: torch tensor
            batch of observations
        actions: torch tensor
            batch of actions executed given obs
        rewards: torch tensor
            rewards received as results of executing act_batch
        next_obses: torch tensor
            next set of observations seen after executing act_batch
        not_dones: torch tensor
            inverse of whether the episode ended at this tuple of (observation, action) or not
        not_dones_no_max: torch tensor
            inverse of whether the episode ended at this tuple of (observation, action) or not, specifically exlcuding maximum episode steps
        amp_obs: torch Tensor
            batch of associated amp observations
        """

        idxs = torch.randint(0,
                            self.capacity if self.full else self.idx, 
                            (batch_size,), device=self.device)
        obses = self.obses[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_obses = self.next_obses[idxs]
        dones = self.dones[idxs]
        amp_obses = self.amp_obses[idxs]

        return obses, actions, rewards, next_obses, dones, amp_obses