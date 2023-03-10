# Stuff added by Dhruv Sreenivas for testing on-policy algorithms in the IsaacGym ILFO suite

from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.algos_torch.model_builder import ModelBuilder

import time
from datetime import datetime
import numpy as np
from torch import optim
import torch 
from torch import nn

import isaacgymenvs.learning.common_agent as common_agent
from isaacgymenvs.learning.amp_network_builder import AMPBuilder
from isaacgymenvs.learning.amp_models import ModelAMPContinuous

class PPOFixedDiscriminatorAgent(common_agent.CommonAgent):
    '''PPO agent with reward coming from a fixed discriminator.'''
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        
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
        
    def _build_amp_net_config(self):
        config = self._build_net_config()
        config['amp_input_shape'] = self._amp_observation_space.shape
        return config
    
    def _load_config_params(self, config):
        super()._load_config_params(config)
        
        self._task_reward_w = config['task_reward_w']
        self._disc_reward_w = config['disc_reward_w']
        self._disc_reward_scale = config['disc_reward_scale']
        self._amp_observation_space = self.env_info['amp_observation_space']
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        
        return
    
    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs
        
    # ========= amp params =========
    def _load_amp_params(self, chkpt_path):
        state = torch_ext.load_checkpoint(chkpt_path)
        return state
    
    def _eval_disc(self, amp_obs):
        # no need to preprocess I think
        return self._amp_agent.a2c_network.eval_disc(amp_obs)
    
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
        
        epinfos = []
        update_list = self.update_list

        for n in range(self.horizon_length):
            self.obs, done_env_ids = self._env_reset_done()
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            
            # change the rewards to the AMP agent-based discriminator ones
            amp_rewards = self._calc_amp_rewards(infos['amp_obs'])
            assert amp_rewards.size() == rewards.size()
            rewards = self._combine_rewards(rewards, amp_rewards)
            
            shaped_rewards = self.rewards_shaper(rewards) # just scaling rewards by something arbitrary (1.0 here)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
  
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        
        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        return batch_dict