import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from abc import abstractmethod
import math


class AbstractOptimizer(Optimizer):
    """Abstract base class for custom optimizers."""
    
    def __init__(self, params, defaults):
        super().__init__(params, defaults)

    @abstractmethod
    def step(self, closure=None):
        """
        Performs a single optimization step (parameter update).
        To be implemented by subclasses.
        """
        pass

class RNNP_sOptimizer(AbstractOptimizer):
    """
    RNNP-S Optimizer: RNNP with Muon-style scaling
    
    This optimizer applies Row-Normalized Nesterov with Polynomial decay (RNNP)
    to 2D+ parameters and Adam to 1D/scalar parameters. For 2D+ parameters,
    it includes Muon-style scaling: max(1, sqrt(dim[-2]/dim[-1])).
    
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr (float): Learning rate (default: 0.01)
        momentum (float): Momentum factor for Nesterov acceleration (default: 0.9)
        beta (float): Exponential moving average factor for momentum buffer (default: 0.95)
        weight_decay (float): Weight decay (L2 penalty) (default: 0.0)
        betas (tuple): Coefficients for Adam on 1D parameters (default: (0.9, 0.999))
        eps (float): Term added to denominator for numerical stability (default: 1e-8)
    """
    
    def __init__(self, params, lr=0.01, momentum=0.9, beta=0.95, weight_decay=0.0, 
                 betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, 
                       betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.beta = beta
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.state = {}

    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            lr = group.get('lr', self.lr)
            momentum = group.get('momentum', self.momentum)
            beta = group.get('beta', self.beta)
            weight_decay = group.get('weight_decay', self.weight_decay)
            betas = group.get('betas', self.betas)
            eps = group.get('eps', self.eps)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                param_state = self.state.setdefault(p, {})
                
                if grad.dim() >= 2:
                    # Apply RNNP with Muon-style scaling for 2D+ parameters
                    if 'momentum_buffer' not in param_state:
                        buf = torch.zeros_like(grad)
                    else:
                        buf = param_state['momentum_buffer']
                    
                    # Update momentum buffer with exponential moving average
                    buf.lerp_(grad, 1 - beta)
                    
                    # Compute Nesterov momentum update
                    nesterov_buf = grad.lerp(buf, momentum)
                    
                    # Row normalization (normalize along last dimension)
                    normed = F.normalize(nesterov_buf, p=2, dim=-1)
                    
                    # Apply Muon-style scaling: max(1, sqrt(dim[-2]/dim[-1]))
                    scale = max(1, math.sqrt(grad.size(-2) / grad.size(-1)))
                    normed = normed * scale
                    
                    # Apply weight decay if specified
                    if weight_decay > 0:
                        p.data.mul_(1 - lr * weight_decay)
                    
                    # Update parameters
                    p.data.add_(normed, alpha=-lr)
                    param_state['momentum_buffer'] = buf
                    
                elif grad.dim() == 1 or grad.dim() == 0:
                    # Apply Adam for 1D/scalar parameters
                    if 'exp_avg' not in param_state:
                        param_state['exp_avg'] = torch.zeros_like(grad)
                        param_state['exp_avg_sq'] = torch.zeros_like(grad)
                        param_state['step'] = 0
                    
                    exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                    param_state['step'] += 1
                    
                    # Update biased first and second moment estimates
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1-betas[0])
                    exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1-betas[1])
                    
                    # Compute bias-corrected first and second moment estimates
                    bias_correction1 = 1 - betas[0] ** param_state['step']
                    bias_correction2 = 1 - betas[1] ** param_state['step']
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                    
                    # Compute update
                    denom = exp_avg_sq.sqrt().add_(eps)
                    adam_update = exp_avg / denom
                    
                    # Apply weight decay if specified
                    if weight_decay > 0:
                        p.data.mul_(1 - step_size * weight_decay)
                    
                    # Update parameters
                    p.data.add_(adam_update, alpha=-step_size)
                    
        return loss

    def distributed_step(self, closure=None):
        """
        Distributed version of step function (placeholder implementation).
        
        This method provides a basic framework for distributed training support.
        For full distributed training, additional synchronization logic would be needed.
        """
        import torch.distributed as dist
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        # For now, just call the regular step function
        # In a full implementation, this would include gradient synchronization
        return self.step(closure)

    def zero_grad(self, set_to_none=False):
        """
        Sets the gradients of all optimized parameters to zero.
        
        Args:
            set_to_none (bool): Instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly 
                improve performance.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()

    def state_dict(self):
        """
        Returns the state of the optimizer as a dict.
        
        It contains two entries:
        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a dict containing all parameter groups
        """
        # Pack state
        packed_state = {(id(k) if isinstance(k, torch.Tensor) else k): v 
                       for k, v in self.state.items()}
        return {
            'state': packed_state,
            'param_groups': self.param_groups,
        }

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.
        
        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to state_dict().
        """
        # Deep copy, to avoid modifying the user's state_dict
        state_dict = dict(state_dict)
        
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']
        
        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                           "parameter groups")
        
        # Update the state dict
        self.param_groups = saved_groups
        
        # Unpack state
        state = state_dict['state']
        from itertools import chain
        id_map = {old_id: p for old_id, p in 
                 zip(chain.from_iterable((g['params'] for g in saved_groups)),
                     chain.from_iterable((g['params'] for g in groups)))}
        
        def cast(param, value):
            """Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Tensors stored in optimizer state dict are moved to device if needed
                if param.is_cuda and not value.is_cuda:
                    value = value.cuda(param.device)
                if value.dtype != param.dtype:
                    value = value.to(param.dtype)
                return value.clone()
            elif isinstance(value, dict):
                return {k: cast(param, v) for k, v in value.items()}
            elif isinstance(value, list):
                return [cast(param, v) for v in value]
            else:
                return value
        
        # Copy state dict
        self.state = {}
        for k, v in state.items():
            if k in id_map:
                param = id_map[k]
                self.state[param] = cast(param, v)