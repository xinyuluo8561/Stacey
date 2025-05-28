import torch
from torch.optim import Optimizer

class Stacey_base(Optimizer):
    def __init__(self, params, lr_tau, lr_eta, lr_alpha, lr_beta1, lr_beta2, momentum, weight_decay, dampening, q, eps, debug):
        if not 0.0 <= lr_tau:
            raise ValueError(f"Invalid learning rate Tau: {lr_tau}")
        if not 0.0 <= lr_eta:
            raise ValueError(f"Invalid learning rate Eta: {lr_eta}")
        if not 0.0 <= lr_alpha:
            raise ValueError(f"Invalid learning rate Alpha: {lr_alpha}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr_tau      = lr_tau,
            lr_eta      = lr_eta,
            lr_alpha    = lr_alpha,
            lr_beta1    = lr_beta1,
            lr_beta2    = lr_beta2,
            momentum    = momentum,
            weight_decay= weight_decay,
            dampening   = dampening,
            q           = q,
            eps         = eps,
            debug       = debug
        )

        params = list(params)
        for p in params:
            p.z = p.data.clone()
            p.m = torch.zeros_like(p.data)

        super(Stacey_base, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Stacey_base, self).__setstate__(state)


