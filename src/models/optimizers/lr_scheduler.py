import torch.nn as nn
import math
import numpy as np

class WarmUpLRScheduler(nn.Module):
    def __init__(self, optimizer, d_model, factor=1, step_size=1, warmup_step=4000):
        super().__init__()
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_step = warmup_step
        self._steps = 0
        self._rate = 0
        self.factor = factor
        self.step_size = step_size

    def step(self):
        self._steps += 1
        if self._steps % self.step_size == 0:
            self.optimizer.param_groups[0]['lr'] = self.rate()

    def rate(self, steps=None):
        if steps is None:
            steps = self._steps
        if steps % self.step_size == 0:
            # steps = steps // self.step_size
            self._rate = 1 * self.d_model ** (-0.5) * min(steps * self.warmup_step ** (-1.5), self.factor*steps ** (-0.5))
        return self._rate


class CosDecayLRScheduler(nn.Module):
    def __init__(self, optimizer, step_size=100, epoches=50, num_examples=1e+6, batch_size=32, init_lr=0.001, mini_lr=1e-7, warmup_size=0):
        super().__init__()
        assert step_size >= 1 and batch_size >= 1
        assert warmup_size >= 0 and warmup_size < 1
        self.optimizer = optimizer
        self.step_size = step_size
        self.epoches = epoches
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.mini_lr = mini_lr
        self.warmup_size = warmup_size

        self.run_total_steps = epoches * (num_examples // batch_size // self.step_size)
        self.run_warmup_steps = self.run_total_steps * self.warmup_size
        self.run_decay_steps = self.run_total_steps - self.run_warmup_steps
        if self.run_warmup_steps > 0:
            self.warmup_rate = self.init_lr / self.run_warmup_steps
        else:
            self.warmup_rate = 0
        self.a = (self.init_lr - self.mini_lr) / 2
        self.b = (self.init_lr + self.mini_lr) / 2

        self._steps = 0
        self._rate = self.init_lr

    def step(self):
        self._steps += 1
        if self._steps % self.step_size == 0:
            self.optimizer.param_groups[0]['lr'] = self.rate()

    def rate(self, steps=None):
        if steps is None:
           steps = self._steps
        run_steps = steps // self.step_size
        if run_steps <= self.run_warmup_steps:
            self._rate = run_steps * self.warmup_rate
        elif run_steps <= self.run_total_steps:
            self._rate = self.a * math.cos(np.pi * (run_steps - self.run_warmup_steps) / self.run_decay_steps) + self.b
        else:
            self._rate = self.mini_lr
        return self._rate


class ExpDecayLRScheduler(nn.Module):
    def __init__(self, optimizer, step_size=100, epoches=50, num_examples=1e+6, batch_size=32, init_lr=0.001, mini_lr=1e-7, warmup_size=0):
        super().__init__()
        assert step_size >= 1 and batch_size >= 1
        assert warmup_size >= 0 and warmup_size < 1
        self.optimizer = optimizer
        self.step_size = step_size
        self.epoches = epoches
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.mini_lr = mini_lr
        self.warmup_size = warmup_size
        self._zero = 1e-10

        self.run_total_steps = epoches * (num_examples // batch_size // self.step_size)
        self.run_warmup_steps = self.run_total_steps * self.warmup_size
        self.run_decay_steps = self.run_total_steps - self.run_warmup_steps
        if self.run_warmup_steps > 0:
            self.warmup_rate = self.init_lr / self.run_warmup_steps
        else:
            self.warmup_rate = 0
        if self.run_decay_steps > 0:
            self.gamma = math.exp(math.log(self._zero) / self.run_decay_steps)
        else:
            self.gamma = 1

        self._steps = 0
        self._rate = self.init_lr

    def step(self):
        self._steps += 1
        if self._steps % self.step_size == 0:
            self.optimizer.param_groups[0]['lr'] = self.rate()

    def rate(self, steps=None):
        if steps is None:
           steps = self._steps
        run_steps = steps // self.step_size
        if run_steps <= self.run_warmup_steps:
            self._rate = run_steps * self.warmup_rate
        elif run_steps <= self.run_total_steps:
            self._rate = (self.init_lr - self.mini_lr) * self.gamma ** (run_steps - self.run_warmup_steps) + self.mini_lr
        else:
            self._rate = self.mini_lr
        return self._rate