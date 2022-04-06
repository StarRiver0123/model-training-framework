import torch
import torch.nn as nn
import torch.nn.functional as F

class ResponseBasedDistilLoss(nn.Module):
    def __init__(self, soft_loss_type='mse', T=10, alpha=0.1):
        super().__init__()
        assert(T >= 1)
        self.T = T
        self.alpha = alpha
        self.soft_loss_type = soft_loss_type
#         if self.soft_loss_type == 'ce':
#             self.soft_loss_func = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, hard_targets):
        # input size:
        #   teacher_logits, student_logits: N,D 即 N,C; target: N
        # output: scalar
        assert (teacher_logits.dim() == 2) and (student_logits.dim() == 2) and (hard_targets.dim() == 1)
        if self.soft_loss_type == 'mse':
            soft_loss = F.mse_loss(student_logits, teacher_logits)
        elif self.soft_loss_type == 'ce':
#             soft_targets = F.softmax(teacher_logits / self.T, dim=-1)
            #soft_loss = F.cross_entropy(student_logits / self.T, F.softmax(teacher_logits / self.T, dim=-1))
            soft_loss = F.kl_div(input=F.log_softmax(student_logits / self.T, dim=1), target=F.softmax(teacher_logits / self.T, dim=1), reduction="batchmean")
        hard_loss = F.cross_entropy(student_logits, hard_targets)
        loss = (1 - self.alpha) * soft_loss + self.alpha * hard_loss
        return loss

