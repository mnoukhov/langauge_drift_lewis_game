import torch
import torch.nn.functional as F

from drift.core import BaseSpeaker, BaseListener
from drift import GUMBEL_DIST


class Speaker(BaseSpeaker):
    def __init__(self, env_config):
        super(Speaker, self).__init__(env_config)
        self.linear1 = torch.nn.Linear(
            self.env_config["p"] * self.env_config["t"], 200, bias=True
        )
        self.linear2 = torch.nn.Linear(
            200,
            self.env_config["p"] * self.env_config["p"] * self.env_config["t"],
            bias=True,
        )
        self.relu = torch.nn.ReLU()
        self.value_linear = torch.nn.Linear(200, 1)
        self.init_weight()

    def init_weight(self):
        torch.nn.init.normal_(self.linear1.weight, std=0.1)
        torch.nn.init.normal_(self.linear2.weight, std=0.1)
        torch.nn.init.normal_(self.value_linear.weight, std=0.1)
        torch.nn.init.zeros_(self.value_linear.bias)

    def greedy(self, objs):
        logits = self.get_logits(objs)
        return torch.argmax(logits, -1)

    def gumbel(self, objs, temperature=1):
        logits = self.get_logits(objs)
        logprobs = F.log_softmax(logits, dim=-1)
        g = GUMBEL_DIST.sample(logits.shape)
        y = F.softmax((g + logprobs) / temperature, dim=-1)
        msgs = torch.argmax(y, dim=-1)
        entropy = torch.distributions.Categorical(probs=y).entropy()
        return y, msgs, entropy

    def new_gumbel(self, objs, temperature=1.0):
        logits = self.get_logits(objs)
        sample = torch.distributions.RelaxedOneHotCategorical(
            logits=logits, temperature=temperature
        ).rsample()

        entropy = torch.distributions.Categorical(logits=logits).entropy()

        size = sample.size()
        indexes = sample.argmax(dim=-1)
        hard_sample = torch.zeros_like(sample).view(-1, size[-1])
        hard_sample.scatter_(1, indexes.view(-1, 1), 1)
        hard_sample = hard_sample.view(*size)

        sample = sample + (hard_sample - sample).detach()

        return sample, entropy

    def sample(self, objs):
        logits = self.get_logits(objs)
        dist = torch.distributions.Categorical(logits=logits)
        msgs = dist.sample()
        logprobs = dist.log_prob(msgs)
        return logprobs, msgs

    def get_logits(self, objs, msgs=None):
        """return [bsz, nb_prop, vocab_size]"""
        oh_objs = self._one_hot(objs)
        # logits = self.linear2(self.relu(self.linear1(oh_objs)))
        logits = self.linear2(self.linear1(oh_objs))
        return logits.view(objs.shape[0], self.env_config["p"], -1)

    def a2c(self, objs):
        oh_objs = self._one_hot(objs)
        obj_enc = self.linear1(oh_objs)
        logits = self.linear2(obj_enc)
        logits = logits.view(objs.shape[0], self.env_config["p"], -1)
        dist = torch.distributions.Categorical(logits=logits)
        ents = dist.entropy()
        msgs = dist.sample()
        logprobs = dist.log_prob(msgs)
        values = self.value_linear(obj_enc)
        return {"msgs": msgs, "logprobs": logprobs, "ents": ents, "values": values}

    def _one_hot(self, objs):
        """Make input a concatenation of one-hot
        :param objs [bsz, nb_props]
        :param oh_objs [bsz, nb_props * nb_types]
        """
        oh_objs = torch.Tensor(
            size=[objs.shape[0], objs.shape[1], self.env_config["t"]]
        )
        oh_objs = oh_objs.to(device=objs.device)
        oh_objs.zero_()
        oh_objs.scatter_(2, objs.unsqueeze(-1), 1)
        return oh_objs.view([objs.shape[0], -1])


class Listener(BaseListener):
    def __init__(self, env_config):
        super(Listener, self).__init__(env_config)
        self.linear1 = torch.nn.Linear(
            self.env_config["p"] * self.env_config["t"], 200, bias=False
        )
        self.linear2 = torch.nn.Linear(200, self.env_config["t"], bias=False)
        self.relu = torch.nn.ReLU()
        self.init_weight()

    def init_weight(self):
        torch.nn.init.normal_(self.linear1.weight, std=0.1)
        torch.nn.init.normal_(self.linear2.weight, std=0.1)

    def get_logits(self, oh_msgs):
        # return self.linear2(self.relu(self.linear1(oh_msgs)))
        return self.linear2(self.linear1(oh_msgs))
