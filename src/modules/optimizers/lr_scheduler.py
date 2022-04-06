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
    def __init__(self, optimizer, step_size=100, epochs=50, num_examples=1e+6, batch_size=32, min_lr=1e-7, warmup_size=0):
        super().__init__()
        assert step_size >= 1 and batch_size >= 1
        assert warmup_size >= 0 and warmup_size < 1
        self.optimizer = optimizer
        # 本函数存在一个bug，step_size>1并且优化器的初始lr不合适，比如对于Ner任务设置偏大比如1e-4时，模型训练会出现奇怪的预测结果。但不知道原因。
        self.step_size = 1
        self.epochs = epochs
        self.num_examples = int(num_examples)
        self.batch_size = batch_size
        self.max_lr = optimizer.state_dict()['param_groups'][0]['lr']
        self.min_lr = min_lr
        self.warmup_size = warmup_size

        self.run_total_steps = epochs * (num_examples // batch_size // self.step_size)
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

# 这个方法有问题，还不能用
class SelfAdjustingAfterWarmUpLRScheduler(nn.Module):
    def __init__(self, optimizer, mean_loss_window=5, cross_mean=True, adjusting_ratio=0.01, epochs=50, num_examples=1e+6, batch_size=32, warmup_size=0, smoothing_zero=1e-8):
        super().__init__()
        self.optimizer = optimizer
        self.mean_loss_window = mean_loss_window
        self.cross_mean = cross_mean
        self.adjusting_ratio = adjusting_ratio
        self.epochs = epochs
        self.num_examples = int(num_examples)
        self.batch_size = batch_size
        self.warmup_size = warmup_size
        self.smoothing_zero = smoothing_zero
        self.milestone_lr = optimizer.state_dict()['param_groups'][0]['lr']
        self.run_total_steps = epochs * (num_examples // batch_size)
        self.run_warmup_steps = self.run_total_steps * self.warmup_size
        self.run_self_adjusting_steps = self.run_total_steps - self.run_warmup_steps
        if self.run_warmup_steps > 0:
            self.warmup_rate = self.milestone_lr / self.run_warmup_steps
        else:
            self.warmup_rate = 0
        # 为了支持断点续训，要保存lr_scheduler的状态，需要保存steps的值，需要用Parameter来定义。
        # self.steps = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.steps = -1
        self.rate = self.milestone_lr
        self.last_loss = 0
        if self.cross_mean:
            self.loss_variance_ratio = np.zeros((self.mean_loss_window, 3))
        else:
            self.loss_variance_ratio = np.zeros((3, self.mean_loss_window))
        self.lr_candidates = [self.rate, self.rate, self.rate]

    def update_optimizer_lr(self, lr=None):
        if lr is None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.rate
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def step(self, loss=0):
        self.steps += 1
        if self.steps < self.run_warmup_steps:
            self.rate = self.steps * self.warmup_rate
            self.update_optimizer_lr()
        else:
            '''
             注意，接收到的loss，是在上一步的模型参数作用下得到的，而模型参数则是在上一步学习率作用下得到的。
             本函数假定在训练过程中，先更新优化器更新模型参数，再学习率.
             所以需要注意，loss_variance_ratio和update_optimizer_lr是相错一步的。
             学习率、模型参数、loss的作用链为：学习率 -> 模型参数 -> loss
            '''
            if not self.cross_mean:
                # 本步得到的loss是上一步学习率的作用效果，所以把loss变化率的值填入上一步的矩阵位置中
                last_adjusting_steps = int((self.steps - 1 - self.run_warmup_steps) % (3 * self.mean_loss_window))
                last_round_number = int(last_adjusting_steps // self.mean_loss_window)
                last_inner_step = int(last_adjusting_steps % self.mean_loss_window)
                self.loss_variance_ratio[last_round_number, last_inner_step] = (self.last_loss - loss) / (
                        self.last_loss + self.smoothing_zero)
                # 开始进行本步的更新
                adjusting_steps = int((self.steps - self.run_warmup_steps) % (3 * self.mean_loss_window))
                if adjusting_steps == 0:  # 大周期开始（包含3个小周期：学习率不变看loss变化率，学习率增加看loss变化率,学习率减少看loss变化率，哪个loss变化率大，选哪个学习率为下一大周期的初始学习率）
                    if self.mean_loss_window > 3:
                        mean_list = (np.sum(self.loss_variance_ratio, axis=-1) - np.max(self.loss_variance_ratio, axis=-1) - np.min(self.loss_variance_ratio, axis=-1)) / (self.mean_loss_window - 2)
                    else:
                        mean_list = np.mean(self.loss_variance_ratio, axis=-1)
                    max_id = int(np.where(mean_list == max(mean_list))[0][0])
                    basic_rate = self.lr_candidates[max_id]
                    # if max_id == 0:
                    #     self.adjusting_ratio = max(self.smoothing_zero, self.adjusting_ratio - self.adjusting_ratio * 0.01)
                    # else:
                    #     self.adjusting_ratio = min(0.3, self.adjusting_ratio + self.adjusting_ratio * 0.01)
                    # print("steps, adjusting_ratio: ", self.steps, self.adjusting_ratio)
                    self.lr_candidates = [basic_rate, min(0.1, basic_rate * (1 + self.adjusting_ratio)), max(1e-8, basic_rate * (1 - self.adjusting_ratio))]
                    self.loss_variance_ratio = np.zeros((3, self.mean_loss_window))
                round_number = int(adjusting_steps // self.mean_loss_window)
                inner_step = int(adjusting_steps % self.mean_loss_window)
                if inner_step == 0:
                    self.update_optimizer_lr(self.lr_candidates[round_number])
            else: # 用交叉平均
                # 本步得到的loss是上一步学习率的作用效果，所以把loss变化率的值填入上一步的矩阵位置中
                last_adjusting_steps = int((self.steps - 1 - self.run_warmup_steps) % (3 * self.mean_loss_window))
                last_round_number = int(last_adjusting_steps // 3)
                last_inner_step = int(last_adjusting_steps % 3)
                self.loss_variance_ratio[last_round_number, last_inner_step] = (self.last_loss - loss) / (
                        self.last_loss + self.smoothing_zero)
                # 开始进行本步的更新
                adjusting_steps = int((self.steps - self.run_warmup_steps) % (3 * self.mean_loss_window))
                if adjusting_steps == 0:  # 大周期开始（包含3个小周期：学习率不变看loss变化率，学习率增加看loss变化率,学习率减少看loss变化率，哪个loss变化率大，选哪个学习率为下一大周期的初始学习率）
                    if self.mean_loss_window > 3:
                        mean_list = (np.sum(self.loss_variance_ratio, axis=0) - np.max(self.loss_variance_ratio, axis=0) - np.min(self.loss_variance_ratio, axis=0)) / (self.mean_loss_window - 2)
                    else:
                        mean_list = np.mean(self.loss_variance_ratio, axis=0)
                    max_id = int(np.where(mean_list == max(mean_list))[0][0])
                    #max_id 需要往前挪一位，因为模型参数是受上一步的学习率影响的。
                    max_id = int((max_id - 1) % 3)
                    basic_rate = self.lr_candidates[max_id]
                    # if max_id == 0:
                    #     self.adjusting_ratio = max(self.smoothing_zero, self.adjusting_ratio - self.adjusting_ratio * 0.01)
                    # else:
                    #     self.adjusting_ratio = min(0.3, self.adjusting_ratio + self.adjusting_ratio * 0.01)
                    # print("steps, adjusting_ratio: ", self.steps, self.adjusting_ratio)
                    self.lr_candidates = [basic_rate, min(0.1, basic_rate * (1 + self.adjusting_ratio)), max(1e-8, basic_rate * (1 - self.adjusting_ratio))]
                    self.loss_variance_ratio = np.zeros((self.mean_loss_window, 3))
                inner_step = int(adjusting_steps % 3)
                self.update_optimizer_lr(self.lr_candidates[inner_step])
        self.last_loss = loss
