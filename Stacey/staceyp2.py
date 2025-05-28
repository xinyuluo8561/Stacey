import torch
from torch.optim import Optimizer
from staceybase import Stacey_base
 

class Stacey_p2(Stacey_base):

    def __init__(
        self                ,
        params              ,
        lr_tau      = 0.001 ,
        lr_eta      = 0.001 ,
        lr_alpha    = 0.001 , 
        lr_beta1    = 0.9   ,
        lr_beta2    = 0.999  ,
        momentum    = 0     ,
        eps         = 1e-4  ,
        weight_decay= 5e-4  ,
        dampening   = 0     ,
        q           = 3     ,
        debug       = False
    ):
        super(Stacey_p2, self).__init__(params, lr_tau, lr_eta, lr_alpha, lr_beta1, lr_beta2, momentum, weight_decay, dampening, q, eps, debug)
    
    def step(self, closure=None):
        if closure is not None:
            loss = closure()

        params_track = dict(z_dist=0., y_dist=0., update_dist=0., grad_dist=0., momentum_dist=0., pdata_dist=0., debug=False)
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            lr_beta1 = group['lr_beta1']
            lr_beta2 = group['lr_beta2']
            lr_tau = group['lr_tau']
            lr_eta = group['lr_eta']
            lr_alpha = group['lr_alpha']
            q = group['q']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None: continue

                # gradient clipping
                # torch.nn.utils.clip_grad_norm_(p.grad.data, max_norm=5.0, norm_type=q)

                c = torch.add(lr_beta1 * p.m, p.grad, alpha=1 - lr_beta1)
                p.z.add_(c, alpha=-lr_alpha)
                c_tmp = torch.mul(c, torch.pow(torch.abs(c) + eps, (2 - q) / (q - 1)))
                p.data.mul_(1 - lr_tau - lr_eta * weight_decay).add_(p.z, alpha=lr_tau).add_(c_tmp, alpha=lr_eta * (lr_tau - 1))

                p.m.mul_(lr_beta2).add_(p.grad, alpha=1 - lr_beta2)

                assert torch.isnan(p.grad.data).count_nonzero() == 0, "grad"
                assert torch.isnan(p.m).count_nonzero() == 0, "m"
                assert torch.isnan(c).count_nonzero() == 0, "c"
                assert torch.isnan(p.z).count_nonzero() == 0, "z"
                assert torch.isnan(c_tmp).count_nonzero() == 0, "c_tmp"

                del c_tmp, c

        if params_track['debug'] is True:
            params_track['grad_dist'] = torch.sqrt(params_track['grad_dist'])
            params_track['pdata_dist'] = torch.sqrt(params_track['pdata_dist'])
            params_track['update_dist'] = torch.sqrt(params_track['update_dist'])
            params_track['y_dist'] = torch.sqrt(params_track['y_dist'])
            params_track['z_dist'] = torch.sqrt(params_track['z_dist'])
            params_track['momentum_dist'] = torch.sqrt(params_track['momentum_dist'])

        return params_track
