from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Attacker:
    def __init__(self,
                 steps: int,
                 quantize: bool = True,
                 levels: int = 256,
                 max_norm: Optional[float] = None,
                 device: torch.device = torch.device('cpu')) -> None:
        self.steps = steps

        self.quantize = quantize
        self.levels = levels
        self.max_norm = max_norm
        
        self.device = device

    def _iter_attack(self, 
                     model1: nn.Module, 
                     model2: nn.Module,
                     model3: nn.Module,
                     inputs: torch.Tensor, 
                     labels_true: torch.Tensor,
                     labels_target: torch.Tensor,
                     epsilon: Optional[float] = None)-> torch.Tensor:

        batch_size = inputs.shape[0]
        delta = torch.zeros_like(inputs, requires_grad=True)

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=1, momentum=0.9)

        # for choosing best results
        best_loss = 1e4 * torch.ones(inputs.size(0), dtype=torch.float, device=self.device)
        best_delta = torch.zeros_like(inputs)

        for _ in range(self.steps):
            if epsilon:
                delta.data.clamp_(-epsilon, epsilon)
                if self.quantize:
                    delta.data.mul_(self.levels - 1).round_().div_(self.levels - 1)

            adv = inputs + delta

            logits1 = model1(adv)
            logits2 = model2(adv)
            logits3 = model3(adv)

            # logits fuse
            logits_e = (logits1 + logits2 + logits3) / 3
            ce_loss_true = F.cross_entropy(logits_e, labels_true, reduction='none')     
            ce_loss_target = F.cross_entropy(logits_e, labels_target, reduction='none')

            loss = ce_loss_target - ce_loss_true
            
            is_better = loss < best_loss

            best_loss[is_better] = loss[is_better]
            best_delta[is_better] = delta.data[is_better]
            
            loss = torch.mean(loss)
            optimizer.zero_grad()
            loss.backward()

            # renorming gradient to [-1, 1]
            grad_norms = delta.grad.view(batch_size, -1).norm(p=float('inf'), dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))

            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            optimizer.step()

            # avoid out of bound
            delta.data.add_(inputs)
            delta.data.clamp_(0, 1).sub_(inputs)

        return best_delta, best_loss

    def attack(self, 
               model1: nn.Module, 
               model2: nn.Module,
               model3: nn.Module,
               inputs: torch.Tensor, 
               labels_true: torch.Tensor,
               labels_target: torch.Tensor)-> torch.Tensor:

        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')

        best_delta, best_loss = self._iter_attack(model1, model2, model3, inputs, labels_true, labels_target, self.max_norm)

        return inputs + best_delta
