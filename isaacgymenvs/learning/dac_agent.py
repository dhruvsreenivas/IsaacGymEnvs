# Stuff added by Dhruv Sreenivas for Discriminator Actor-Critic in the IsaacGym ILFO suite

import rl_games.algos_torch.sac_agent as sac_agent
from rl_games.interfaces.base_algorithm import BaseAlgorithm
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
import torch
import torch.nn as nn
import numpy as np
import time
import os

import isaacgymenvs.learning.replay_buffer as replay_buffer
import isaacgymenvs.learning.dac_buffer as dac_buffer
from typing import Mapping

def dict_detach(info_dict: Mapping[str, torch.Tensor]):
    '''Returns the same dict, but with all values detached.'''
    return {
        k: v.detach()
        for k, v in info_dict.items()
    }

class DACAgent(sac_agent.SACAgent):
    '''Discriminator Actor-Critic agent.'''
    def __init__(self, base_name, params):
        # config setup
        self.config = config = params['config']
        print(config)

        # =================== SAC INIT ===================
        
        # TODO: Get obs shape and self.network
        super().load_networks(params)
        super().base_init(base_name, config)
        
        # load AMP configs
        self._load_amp_config_params(self.config)
        
        self.num_warmup_steps = config["num_warmup_steps"]
        self.gamma = config["gamma"]
        self.critic_tau = config["critic_tau"]
        self.batch_size = config["batch_size"]
        self.init_alpha = config["init_alpha"]
        self.learnable_temperature = config["learnable_temperature"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.num_steps_per_episode = config.get("num_steps_per_episode", 1)
        self.normalize_input = config.get("normalize_input", False)

        self.max_env_steps = config.get("max_env_steps", 1000) # temporary, in future we will use other approach

        print(self.batch_size, self.num_actors, self.num_agents)

        self.num_frames_per_epoch = self.num_actors * self.num_steps_per_episode

        self.log_alpha = torch.tensor(np.log(self.init_alpha)).float().to(self._device)
        self.log_alpha.requires_grad = True
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]

        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max())
        ]

        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        net_config = {
            'obs_dim': self.env_info["observation_space"].shape[0],
            'action_dim': self.env_info["action_space"].shape[0],
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'normalize_input' : self.normalize_input,
            'normalize_input': self.normalize_input,
            'amp_input_shape': self._amp_observation_space.shape
        }
        self.model = self.network.build(net_config)
        self.model.to(self._device)

        print("Number of Agents", self.num_actors, "Batch Size", self.batch_size)

        self.actor_optimizer = torch.optim.Adam(self.model.sac_network.actor.parameters(),
                                                lr=float(self.config['actor_lr']),
                                                betas=self.config.get("actor_betas", [0.9, 0.999]))

        self.critic_optimizer = torch.optim.Adam(self.model.sac_network.critic.parameters(),
                                                 lr=float(self.config["critic_lr"]),
                                                 betas=self.config.get("critic_betas", [0.9, 0.999]))

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=float(self.config["alpha_lr"]),
                                                    betas=self.config.get("alphas_betas", [0.9, 0.999]))

        self.target_entropy_coef = config.get("target_entropy_coef", 1.0)
        self.target_entropy = self.target_entropy_coef * -self.env_info['action_space'].shape[0]
        print("Target entropy", self.target_entropy)

        self.step = 0
        self.algo_observer = config['features']['observer']

        # TODO: Is there a better way to get the maximum number of episodes?
        self.max_episodes = torch.ones(self.num_actors, device=self._device)*self.num_steps_per_episode
        # self.episode_lengths = np.zeros(self.num_actors, dtype=int)
        
        # ==================== FINALLY WE CAN ADD AMP STUFF IN ====================
        
        # add AMP input mean std
        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(self._amp_observation_space.shape).to(self.ppo_device)
        
        # add discriminator optimizer (over both disc mlp and the disc logit final layer)
        assert isinstance(self.model.sac_network._disc_mlp, nn.Module) and isinstance(self.model.sac_network._disc_logits, nn.Module), "something went wrong in discriminator init"
        disc_params = list(self.model.sac_network._disc_mlp.parameters()) + list(self.model.sac_network._disc_logits.parameters())
        self.disc_optimizer = torch.optim.Adam(
            disc_params,
            lr=float(self.config['discriminator_lr']),
            betas=self.config.get('discriminator_betas', [0.9, 0.999])
        )
        
        # replace online replay buffer with the defined AMP one
        self.replay_buffer = dac_buffer.AMPVectorizedReplayBuffer(
            self.env_info['observation_space'].shape,
            self.env_info['action_space'].shape,
            self.replay_buffer_size,
            self._device
        )
    
    # init fns
    def init_tensors(self):
        super().init_tensors()
        self._build_amp_buffers()
        return
    
    def _init_amp_demo_buf(self):
        '''Fills expert buffer to be full at the start of training.'''
        buffer_size = self._amp_obs_demo_buffer.get_buffer_size()
        num_batches = int(np.ceil(buffer_size / self.batch_size))

        for i in range(num_batches):
            curr_samples = self._fetch_amp_obs_demo(self.batch_size)
            self._amp_obs_demo_buffer.store({'amp_obs': curr_samples})

        return
    
    def set_eval(self):
        super().set_eval()
        if self._normalize_amp_input:
            self._amp_input_mean_std.eval()
        return

    def set_train(self):
        super().set_train()
        if self._normalize_amp_input:
            self._amp_input_mean_std.train()
        return
    
    def _load_amp_config_params(self, config):
        # reward weighting
        self._task_reward_w = config['task_reward_w']
        self._disc_reward_w = config['disc_reward_w']
        self._reward_type = config['disc_reward_type']

        # additional amp parameters
        self._amp_observation_space = self.env_info['amp_observation_space']
        self._disc_coef = config['disc_coef']
        self._disc_logit_reg = config['disc_logit_reg']
        self._disc_grad_penalty = config['disc_grad_penalty']
        self._disc_weight_decay = config['disc_weight_decay']
        self._disc_reward_scale = config['disc_reward_scale']
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        self._num_disc_updates_per_ac_update = config.get('num_disc_updates_per_ac_update', 1)
        self._update_disc_during_expl_period = config.get('update_disc_during_expl_period', False)
        return
    
    # amp based methods
    def _init_amp_demo_buf(self):
        buffer_size = self._amp_obs_demo_buffer.get_buffer_size()
        num_batches = int(np.ceil(buffer_size / self.batch_size))

        for i in range(num_batches):
            curr_samples = self._fetch_amp_obs_demo(self.batch_size)
            self._amp_obs_demo_buffer.store({'amp_obs': curr_samples})

        return
    
    def _update_amp_demos(self):
        new_amp_obs_demo = self._fetch_amp_obs_demo(self.batch_size)
        self._amp_obs_demo_buffer.store({'amp_obs': new_amp_obs_demo})
        return
    
    def _fetch_amp_obs_demo(self, num_samples):
        amp_obs_demo = self.vec_env.env.fetch_amp_obs_demo(num_samples)
        return amp_obs_demo

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _combine_rewards(self, task_rewards, amp_rewards):
        disc_r = amp_rewards['disc_rewards']
        combined_rewards = self._task_reward_w * task_rewards + \
                         + self._disc_reward_w * disc_r
        return combined_rewards
        
    def _build_amp_buffers(self):
        amp_obs_demo_buffer_size = int(self.config['amp_obs_demo_buffer_size'])
        self._amp_obs_demo_buffer = replay_buffer.ReplayBuffer(amp_obs_demo_buffer_size, self.ppo_device)
        # no need for amp replay buffer -- store amp batch in the replay buffer itself
        return
        
    # weight fns
    def get_full_state_weights(self):
        state = self.get_weights()

        state['steps'] = self.step
        state['actor_optimizer'] = self.actor_optimizer.state_dict()
        state['critic_optimizer'] = self.critic_optimizer.state_dict()
        state['log_alpha_optimizer'] = self.log_alpha_optimizer.state_dict()
        state['discriminator_optimizer'] = self.disc_optimizer.state_dict()

        return state

    def get_weights(self):
        state = {
            'actor': self.model.sac_network.actor.state_dict(),
            'critic': self.model.sac_network.critic.state_dict(), 
            'critic_target': self.model.sac_network.critic_target.state_dict(),
            'discriminator_mlp': self.model.sac_network._disc_mlp.state_dict(),
            'discriminator_logits': self.model.sac_network._disc_logits.state_dict()
        }
        if self._normalize_amp_input:
            state.update({'amp_input_mean_std': self._amp_input_mean_std.state_dict()})
        return state
    
    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def set_weights(self, weights):
        self.model.sac_network.actor.load_state_dict(weights['actor'])
        self.model.sac_network.critic.load_state_dict(weights['critic'])
        self.model.sac_network.critic_target.load_state_dict(weights['critic_target'])
        self.model.sac_network._disc_mlp.load_state_dict(weights['discriminator_mlp'])
        self.model.sac_network._disc_logits.load_state_dict(weights['discriminator_logits'])

        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(weights['running_mean_std'])
        
        if self._normalize_amp_input and 'amp_input_mean_std' in weights:
            self._amp_input_mean_std.load_state_dict(weights['amp_input_mean_std'])

    def set_full_state_weights(self, weights):
        self.set_weights(weights)

        self.step = weights['step']
        self.actor_optimizer.load_state_dict(weights['actor_optimizer'])
        self.critic_optimizer.load_state_dict(weights['critic_optimizer'])
        self.log_alpha_optimizer.load_state_dict(weights['log_alpha_optimizer'])
        self.disc_optimizer.load_state_dict(weights['discriminator_optimizer'])
        
    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint)
    
    # loss functions for discriminator
    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.zeros_like(disc_logits))
        return loss
    
    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits))
        return loss
    
    def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo):
        # prediction loss
        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # logit reg
        logit_weights = self.model.sac_network.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty
        disc_demo_grad = torch.autograd.grad(disc_demo_logit, obs_demo, grad_outputs=torch.ones_like(disc_demo_logit),
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        # weight decay
        if (self._disc_weight_decay != 0):
            disc_weights = self.model.sac_network.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)

        disc_info = {
            'disc_loss': disc_loss,
            'disc_grad_penalty': disc_grad_penalty,
            'disc_logit_loss': disc_logit_loss,
            'disc_agent_acc': disc_agent_acc,
            'disc_demo_acc': disc_demo_acc,
            'disc_agent_logit': disc_agent_logit,
            'disc_demo_logit': disc_demo_logit
        }
        return disc_info
    
    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        agent_acc = disc_agent_logit < 0
        agent_acc = torch.mean(agent_acc.float())
        demo_acc = disc_demo_logit > 0
        demo_acc = torch.mean(demo_acc.float())
        return agent_acc, demo_acc

    @torch.no_grad()
    def _calc_disc_rewards(self, amp_obs, reward_type='amp'):
        disc_logits = self.model.disc(amp_obs)
        if reward_type == 'amp':
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))
        else:
            disc_r = disc_logits # AIRL reward

        disc_r *= self._disc_reward_scale
        return disc_r

    def _combine_rewards(self, task_rewards, amp_rewards):
        combined_rewards = self._task_reward_w * task_rewards + self._disc_reward_w * amp_rewards
        return combined_rewards
        
    # new update fn
    def update_disc(self, replay_obs, expert_obs, step):
        del step
        expert_obs.requires_grad_(True)
        
        replay_logits = self.model.disc(replay_obs)
        expert_logits = self.model.disc(expert_obs)
        disc_loss_info = self._disc_loss(replay_logits, expert_logits, expert_obs)
        disc_loss = disc_loss_info['disc_loss']
        
        self.disc_optimizer.zero_grad(set_to_none=True)
        disc_loss.backward()
        self.disc_optimizer.step()
        
        return dict_detach(disc_loss_info)

    def update(self, step):
        obs, action, reward, next_obs, done, rl_amp_replay_obs = self.replay_buffer.sample(self.batch_size)
        not_done = ~done
        obs = self.preproc_obs(obs)
        next_obs = self.preproc_obs(next_obs)
        rl_amp_replay_obs = self._preproc_amp_obs(rl_amp_replay_obs)
        
        # update discriminator
        amp_expert_obs = self._amp_obs_demo_buffer.sample(self.batch_size)
        amp_expert_obs = amp_expert_obs['amp_obs']
        amp_expert_obs = self._preproc_amp_obs(amp_expert_obs)
        disc_info = self.update_disc(rl_amp_replay_obs, amp_expert_obs, step)

        for _ in range(self._num_disc_updates_per_ac_update - 1):
            _, _, _, _, _, amp_replay_obs = self.replay_buffer.sample(self.batch_size)
            amp_expert_obs = self._amp_obs_demo_buffer.sample(self.batch_size)
            amp_expert_obs = amp_expert_obs['amp_obs']
            
            amp_replay_obs = self._preproc_amp_obs(amp_replay_obs)
            amp_expert_obs = self._preproc_amp_obs(amp_expert_obs)
            
            disc_info = self.update_disc(amp_replay_obs, amp_expert_obs, step)

        # update reward to account for discriminator stuff
        disc_reward = self._calc_disc_rewards(rl_amp_replay_obs, reward_type=self._reward_type)
        combined_reward = self._combine_rewards(reward, disc_reward)

        # update actor/critic
        critic_loss, critic1_loss, critic2_loss = self.update_critic(obs, action, combined_reward, next_obs, not_done, step)

        actor_loss, entropy, alpha, alpha_loss = self.update_actor_and_alpha(obs, step)

        actor_loss_info = actor_loss, entropy, alpha, alpha_loss
        self.soft_update_params(self.model.sac_network.critic, self.model.sac_network.critic_target,
                                     self.critic_tau)
        
        return actor_loss_info, disc_info, critic1_loss, critic2_loss

    # eval + train rollouts
    
    def play_steps(self, random_exploration = False):
        total_time_start = time.time()
        total_update_time = 0
        total_time = 0
        step_time = 0.0
        actor_losses = []
        entropies = []
        alphas = []
        alpha_losses = []
        critic1_losses = []
        critic2_losses = []
        disc_losses_during_rl = []
        disc_losses_pre_rl = []

        obs = self.obs
        for s in range(self.num_steps_per_episode):
            self.set_eval()
            if random_exploration:
                action = torch.rand((self.num_actors, *self.env_info["action_space"].shape), device=self._device) * 2.0 - 1.0
            else:
                with torch.no_grad():
                    action = self.act(obs.float(), self.env_info["action_space"].shape, sample=True)

            step_start = time.time()

            with torch.no_grad():
                next_obs, rewards, dones, infos = self.env_step(action)
            step_end = time.time()

            self.current_rewards += rewards
            self.current_lengths += 1

            total_time += (step_end - step_start)
            step_time += (step_end - step_start)

            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])

            not_dones = 1.0 - dones.float()

            self.algo_observer.process_infos(infos, done_indices)

            no_timeouts = self.current_lengths != self.max_env_steps
            dones = dones * no_timeouts

            self.current_rewards = self.current_rewards * not_dones
            self.current_lengths = self.current_lengths * not_dones

            if isinstance(obs, dict):
                obs = obs['obs']
            if isinstance(next_obs, dict):    
                next_obs = next_obs['obs']
            amp_obs = infos['amp_obs']

            rewards = self.rewards_shaper(rewards)
            self.replay_buffer.add(obs, action, torch.unsqueeze(rewards, 1), next_obs, torch.unsqueeze(dones, 1), amp_obs)

            self.obs = obs = next_obs.clone()

            if not random_exploration:
                self.set_train()
                update_time_start = time.time()
                actor_loss_info, disc_info, critic1_loss, critic2_loss = self.update(self.epoch_num)
                update_time_end = time.time()
                update_time = update_time_end - update_time_start

                self.extract_actor_stats(actor_losses, entropies, alphas, alpha_losses, actor_loss_info)
                critic1_losses.append(critic1_loss)
                critic2_losses.append(critic2_loss)
                disc_losses_during_rl.append(disc_info['disc_loss'])
            elif self._update_disc_during_expl_period:
                # update discriminator during explore period
                _, _, _, _, _, amp_replay_obs = self.replay_buffer.sample(self.batch_size)
                amp_expert_obs = self._amp_obs_demo_buffer.sample(self.batch_size)
                disc_info = self.update_disc(amp_replay_obs, amp_expert_obs, 0)
                disc_losses_pre_rl.append(disc_info['disc_loss'])
            else:
                update_time = 0

            total_update_time += update_time

        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        play_time = total_time - total_update_time

        return step_time, play_time, total_update_time, total_time, actor_losses, entropies, alphas, alpha_losses, critic1_losses, critic2_losses, disc_losses_pre_rl, disc_losses_during_rl
    
    def train(self):
        self.init_tensors()
        self.algo_observer.after_init(self)
        self.last_mean_rewards = -100500
        total_time = 0
        # rep_count = 0
        self.frame = 0
        self.obs = self.env_reset()
        
        # initialize expert buffer as well
        self._init_amp_demo_buf()
        print('============ everything initialized and ready to go! ============')

        while True:
            self.epoch_num += 1
            step_time, play_time, update_time, epoch_total_time, actor_losses, entropies, alphas, alpha_losses, critic1_losses, critic2_losses, disc_losses_pre_rl, disc_losses_during_rl = self.train_epoch()

            total_time += epoch_total_time

            curr_frames = self.num_frames_per_epoch
            self.frame += curr_frames

            fps_step = curr_frames / step_time
            fps_step_inference = curr_frames / play_time
            fps_total = curr_frames / epoch_total_time

            self.writer.add_scalar('performance/step_inference_rl_update_fps', fps_total, self.frame)
            self.writer.add_scalar('performance/step_inference_fps', fps_step_inference, self.frame)
            self.writer.add_scalar('performance/step_fps', fps_step, self.frame)
            self.writer.add_scalar('performance/rl_update_time', update_time, self.frame)
            self.writer.add_scalar('performance/step_inference_time', play_time, self.frame)
            self.writer.add_scalar('performance/step_time', step_time, self.frame)

            # add pre rl stuff if it exists
            if len(disc_losses_pre_rl) > 0:
                self.writer.add_scalar('losses/disc_loss_pre_rl', torch_ext.mean_list(disc_losses_pre_rl).item(), self.frame)
            
            # logging training
            if self.epoch_num >= self.num_warmup_steps:
                self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(actor_losses).item(), self.frame)
                self.writer.add_scalar('losses/c1_loss', torch_ext.mean_list(critic1_losses).item(), self.frame)
                self.writer.add_scalar('losses/c2_loss', torch_ext.mean_list(critic2_losses).item(), self.frame)
                self.writer.add_scalar('losses/entropy', torch_ext.mean_list(entropies).item(), self.frame)

                if alpha_losses[0] is not None:
                    self.writer.add_scalar('losses/alpha_loss', torch_ext.mean_list(alpha_losses).item(), self.frame)
                self.writer.add_scalar('info/alpha', torch_ext.mean_list(alphas).item(), self.frame)

                # add discriminator losses
                self.writer.add_scalar('losses/disc_loss_during_rl', torch_ext.mean_list(disc_losses_during_rl).item(), self.frame)

            self.writer.add_scalar('info/epochs', self.epoch_num, self.frame)
            self.algo_observer.after_print_stats(self.frame, self.epoch_num, total_time)

            # logging eval
            if self.game_rewards.current_size > 0:
                mean_rewards = self.game_rewards.get_mean()
                mean_lengths = self.game_lengths.get_mean()

                self.writer.add_scalar('rewards/step', mean_rewards, self.frame)
                self.writer.add_scalar('rewards/time', mean_rewards, total_time)
                self.writer.add_scalar('episode_lengths/step', mean_lengths, self.frame)
                self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)
                checkpoint_name = self.config['name'] + '_ep_' + str(self.epoch_num) + '_rew_' + str(mean_rewards)

                should_exit = False

                if mean_rewards > self.last_mean_rewards and self.epoch_num >= self.save_best_after:
                    print('saving next best rewards: ', mean_rewards)
                    self.last_mean_rewards = mean_rewards
                    self.save(os.path.join(self.nn_dir, self.config['name']))
                    if self.last_mean_rewards > self.config.get('score_to_win', float('inf')):
                        print('Maximum reward achieved. Network won!')
                        self.save(os.path.join(self.nn_dir, checkpoint_name))
                        should_exit = True

                if self.epoch_num >= self.max_epochs and self.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(self.epoch_num) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                update_time = 0

                if should_exit:
                    return self.last_mean_rewards, self.epoch_num
