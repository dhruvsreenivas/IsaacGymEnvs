# Stuff added by Dhruv Sreenivas for testing on-policy algorithms in the IsaacGym ILFO suite

from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.algos_torch.a2c_continuous import A2CAgent

import time
from datetime import datetime
import numpy as np
from torch import optim
import torch 
from torch import nn

from isaacgymenvs.learning.amp_network_builder import AMPBuilder
from isaacgymenvs.learning.amp_models import ModelAMPContinuous

def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

class PPOFixedDiscriminatorAgent(A2CAgent):
    '''PPO agent with reward coming from a fixed discriminator.'''
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self._load_config_params(self.config)
        
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std
    
        # load AMP agent checkpoint (aka rebuild everything bc making an agent is heinous af in this env)
        amp_chkpt_path = self.config['amp_chkpt_path']
        amp_trained_state = self._load_amp_params(amp_chkpt_path)
        
        amp_network = AMPBuilder()
        amp_network.load(params['amp_network'])
        amp_model = ModelAMPContinuous(amp_network)
        net_config = self._build_amp_net_config()
        self._amp_agent = amp_model.build(net_config).to(self.ppo_device)
        self._amp_agent.load_state_dict(amp_trained_state['model'])
        
        # scaling amp input mean/std
        self._amp_input_mean_std = RunningMeanStd(self._amp_observation_space.shape).to(self.ppo_device)
        self._amp_input_mean_std.load_state_dict(amp_trained_state['amp_input_mean_std'])
        
        # for testing after init!
        print('======== TESTING ========')
        print(f'model type: {type(self.model)}')
        print(f'network type: {type(self.model.a2c_network)}')
        print('======== FINISHED TESTING ========')
        
    def _build_amp_net_config(self):
        obs_shape = self.obs_shape
        build_config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        build_config['amp_input_shape'] = self._amp_observation_space.shape
        return build_config
    
    def _load_config_params(self, config):
        # load amp specific config params
        
        self._task_reward_w = config['task_reward_w']
        self._disc_reward_w = config['disc_reward_w']
        self._disc_reward_scale = config['disc_reward_scale']
        self._amp_observation_space = self.env_info['amp_observation_space']
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        
        return
        
    # ========= amp params =========
    def _load_amp_params(self, chkpt_path):
        state = torch_ext.load_checkpoint(chkpt_path)
        return state
    
    def _eval_disc(self, amp_obs):
        # no need to preprocess I think
        return self._amp_agent.a2c_network.eval_disc(amp_obs)
    
    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs
    
    # amp rewards
    def _calc_amp_rewards(self, amp_obs: torch.Tensor) -> torch.Tensor:
        amp_obs = self._preproc_amp_obs(amp_obs)
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits)) 
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))
            disc_r *= self._disc_reward_scale
        return disc_r
    
    def _combine_rewards(self, task_rewards, amp_rewards):
        return self._task_reward_w * task_rewards + self._disc_reward_w * amp_rewards
    
    # ========= modified training =========
    def play_steps(self):
        self.set_eval()
        
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)
            
            # get discriminator rewards and combine like AMP
            assert 'amp_obs' in infos, "not an AMP env -- bad!"
            amp_rewards = self._calc_amp_rewards(infos['amp_obs'])
            print(f'task reward mean: {rewards.mean()}')
            print(f'amp reward mean: {amp_rewards.mean()}')
            combined_rewards = self._combine_rewards(rewards, amp_rewards)

            shaped_rewards = self.rewards_shaper(combined_rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards # task rewards only! -- important for evaluation
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = self.dones.view(self.num_actors, self.num_agents).all(dim=1).nonzero(as_tuple=False)

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        return batch_dict