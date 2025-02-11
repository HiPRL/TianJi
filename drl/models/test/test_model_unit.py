import sys
sys.path.insert(0, './../../../')

import torch
import unittest
import numpy as np
from drl.models import MLP, ActorCritic, AtariModel, GRU, MixerNet, PPOActorCritic, PPOAtari


class TestMLP(unittest.TestCase):
    def setUp(self) -> None:
        self.mlp = MLP(c1=4, c2=32)
        self.mlp.init_params()
        self.mlp.to_device()
        print("MLP initialized")
    
    def test_forward(self):
        data = torch.ones(4)
        self.mlp.forward(data)


class TestActorCritic(unittest.TestCase):
    def setUp(self) -> None:
        self.actor_critic = ActorCritic(32, 4, 36)
        self.actor_critic.init_params()
        self.actor_critic.to_device()
        print("ActorCritic initialized")
    
    def test_forward(self):
        state = torch.ones(32)
        action = torch.ones(4)
        self.actor_critic.actor_forward(state)
        self.actor_critic.critic_forward(state, action)



class TestAtariModel(unittest.TestCase):
    def setUp(self) -> None:
        self.atari_model = AtariModel(
            act_dim=32,
            dueling=True
        )
        self.atari_model.init_params()
        self.atari_model.to_device()
        print("AtariModel initialized")
    
    def test_forward(self):
        state = torch.ones((32, 4, 84, 84))
        self.atari_model.forward(state)


class TestGRU(unittest.TestCase):
    def setUp(self) -> None:
        self.gru = GRU(128, 12)
        self.gru.init_params()
        self.gru.to_device()
        print("GRU initialized")
    
    def test_forward(self):
        state = torch.ones(32, 128)
        hidden_state = torch.zeros(32, 64)
        self.gru.forward(state, hidden_state)


class TestMixerNet(unittest.TestCase):
    def setUp(self) -> None:
        self.mixer_net = MixerNet(12, 54)
        self.mixer_net.init_params()
        self.mixer_net.to_device()
        print("MixerNet initialized")
    
    def test_forward(self):
        state = torch.ones(32, 54)
        a_qs = torch.ones(32, 12, 1)
        self.mixer_net.forward(a_qs, state)


class TestPPOActorCritic(unittest.TestCase):
    def setUp(self) -> None:
        self.ppo_actor_critic = PPOActorCritic(48, 12, 64)
        self.ppo_actor_critic.init_params()
        self.ppo_actor_critic.to_device()
        print("PPOActorCritic initialized")
    
    def test_forward(self):
        state = torch.ones(32, 48)
        self.ppo_actor_critic.forward(state)


class TestPPOAtari(unittest.TestCase):
    def setUp(self) -> None:
        self.ppo_atari = PPOAtari(22, 12)
        self.ppo_atari.init_params()
        self.ppo_atari.to_device()
        print("PPOAtari initialized")
    
    def test_forward(self):
        state = torch.ones((32, 22, 88, 88))
        self.ppo_atari.forward(state)




if __name__ == '__main__':
    t = unittest.TestSuite()
    t.addTest(TestMLP("test_forward"))
    t.addTest(TestActorCritic("test_forward"))
    t.addTest(TestAtariModel("test_forward"))
    t.addTest(TestGRU("test_forward"))
    t.addTest(TestMixerNet("test_forward"))
    t.addTest(TestPPOActorCritic("test_forward"))
    t.addTest(TestPPOAtari("test_forward"))
    run=unittest.TextTestRunner()
    run.run(t)