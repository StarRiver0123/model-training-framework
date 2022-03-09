import torch
import torch.nn as nn
import math
import numpy as np

class WarmUpLRScheduler(nn.Module):
    def __init__(self, optimizer, d_model, factor=1, step_size=1, warmup_step=4000):
        super().__init__()
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_step = warmup_step
        self._steps = nn.Parameter(torch.tensor(0), requires_grad=False)
        self._rate = 0
        self.factor = factor
        self.step_size = step_size

    def step(self):
        self._steps += 1
        if self._steps % self.step_size == 0:
            _rate = self.rate()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = _rate

    def rate(self, steps=None):
        if steps is None:
            steps = self._steps
        if steps % self.step_size == 0:
            # steps = steps // self.step_size
            self._rate = 1 * self.d_model ** (-0.5) * min(steps * self.warmup_step ** (-1.5), self.factor*steps ** (-0.5))
        return self._rate


class CosDecayLRScheduler(nn.Module):
    def __init__(self, optimizer, step_size=100, epoches=50, num_examples=1e+6, batch_size=32, max_lr=0.001, min_lr=1e-7, warmup_size=0):
        super().__init__()
        assert step_size >= 1 and batch_size >= 1
        assert warmup_size >= 0 and warmup_size < 1
        self.optimizer = optimizer
        # 本函数存在一个bug，step_size>1并且优化器的初始lr不合适，比如对于Ner任务设置偏大比如1e-4时，模型训练会出现奇怪的预测结果。但不知道原因。
        self.step_size = 1
        self.epoches = epoches
        self.num_examples = int(num_examples)
        self.batch_size = batch_size
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_size = warmup_size

        self.run_total_steps = epoches * (num_examples // batch_size // self.step_size)
        self.run_warmup_steps = self.run_total_steps * self.warmup_size
        self.run_decay_steps = self.run_total_steps - self.run_warmup_steps
        if self.run_warmup_steps > 0:
            self.warmup_rate = self.max_lr / self.run_warmup_steps
        else:
            self.warmup_rate = 0
        self.a = (self.max_lr - self.min_lr) / 2
        self.b = (self.max_lr + self.min_lr) / 2
        # 为了支持断点续训，要保存lr_scheduler的状态，需要保存steps的值，需要用Parameter来定义。
        self._steps = nn.Parameter(torch.tensor(0), requires_grad=False)
        self._rate = self.max_lr

    def step(self):
        self._steps += 1
        if self._steps.item() % self.step_size == 0:
            _ = self.rate()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self._rate


    def rate(self, steps=None):
        if steps is None:
           steps = self._steps.item()
        run_steps = steps // self.step_size
        if run_steps <= self.run_warmup_steps:
            self._rate = run_steps * self.warmup_rate
        elif run_steps <= self.run_total_steps:
            self._rate = self.a * math.cos(np.pi * (run_steps - self.run_warmup_steps) / self.run_decay_steps) + self.b
        else:
            self._rate = self.min_lr
        return self._rate

