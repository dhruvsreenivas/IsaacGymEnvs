# Stuff added by Dhruv Sreenivas for Discriminator Actor-Critic in the IsaacGym ILFO suite

from rl_games.algos_torch.models import ModelSACContinuous

class ModelDACContinuous(ModelSACContinuous):
    '''DAC model class.'''
    def __init__(self, network):
        super().__init__(network)
        return
    
    class Network(ModelSACContinuous.Network):
        def __init__(self, net, **kwargs):
            super().__init__(net, **kwargs) # parameter sac_network is the net
            return
        
        def forward(self, input_dict):
            # sac outputs
            is_train = input_dict.get('is_train', True)
            dist = super().forward(input_dict)
            
            # additional discriminator outputs, everything in dict
            result = {}
            if (is_train):
                amp_obs = input_dict['amp_obs']
                disc_agent_logit = self.sac_network.eval_disc(amp_obs)
                result["disc_agent_logit"] = disc_agent_logit

                amp_obs_replay = input_dict['amp_obs_replay']
                disc_agent_replay_logit = self.sac_network.eval_disc(amp_obs_replay)
                result["disc_agent_replay_logit"] = disc_agent_replay_logit

                amp_demo_obs = input_dict['amp_obs_demo']
                disc_demo_logit = self.sac_network.eval_disc(amp_demo_obs)
                result["disc_demo_logit"] = disc_demo_logit
            
            result["action_dist"] = dist
            return result
        
        def disc(self, amp_obs):
            return self.sac_network.eval_disc(amp_obs)
        
    def norm_obs(self, obs):
        return self.Network.norm_obs(obs)
        