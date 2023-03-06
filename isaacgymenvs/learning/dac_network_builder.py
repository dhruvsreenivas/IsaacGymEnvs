# Stuff added by Dhruv Sreenivas for Discriminator Actor-Critic in the IsaacGym ILFO suite.

from rl_games.algos_torch import network_builder
import torch
import torch.nn as nn

DISC_LOGIT_INIT_SCALE = 1.0

class DACBuilder(network_builder.SACBuilder):
    '''Discriminator Actor-Critic network builder.'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return
    
    def build(self, name, **kwargs):
        net = DACBuilder.Network(self.params, **kwargs)
        return net
    
    class Network(network_builder.SACBuilder.Network):
        '''Constructs the underlying PyTorch modules for the DAC agent.'''
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            
            # build discriminator on top of it
            amp_input_shape = kwargs.get('amp_input_shape')
            self._build_disc(amp_input_shape)
            return
            
        def _build_disc(self, input_shape):
            '''Stolen '_build_disc' method from AMP.'''
            self._disc_mlp = nn.Sequential()

            mlp_args = {
                'input_size' : input_shape[0], 
                'units' : self._disc_units, 
                'activation' : self._disc_activation, 
                'dense_func' : torch.nn.Linear
            }
            self._disc_mlp = self._build_mlp(**mlp_args)
            
            mlp_out_size = self._disc_units[-1]
            self._disc_logits = torch.nn.Linear(mlp_out_size, 1)

            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias) 

            torch.nn.init.uniform_(self._disc_logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits.bias)
            
        def load(self, params):
            super().load(params)

            self._disc_units = params['disc']['units']
            self._disc_activation = params['disc']['activation']
            self._disc_initializer = params['disc']['initializer']
            return
            
        def eval_disc(self, amp_obs):
            disc_mlp_out = self._disc_mlp(amp_obs)
            disc_logits = self._disc_logits(disc_mlp_out)
            return disc_logits

        def get_disc_logit_weights(self):
            return torch.flatten(self._disc_logits.weight)

        def get_disc_weights(self):
            weights = []
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))

            weights.append(torch.flatten(self._disc_logits.weight))
            return weights