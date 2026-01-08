"""
Muon optimizer from Keller et al.
Also a lot of borrowing of ideas from modded-nanogpt.

RNNPS optimizer: Row-Normalized Nesterov with Polynomial Scaling.
Similar to Muon but uses row normalization instead of Newton-Schulz orthogonalization.
"""
import torch
import math
import torch.nn.functional as F
from torch import Tensor
import torch.distributed as dist
try:
    import wandb
except ImportError:
    wandb = None

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
        weight_decay: L2 weight decay. (default: 0.0)
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            group = dict(params=[p for p in params if p.numel() == size])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for p in params:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf: Tensor = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                # Apply weight decay (L2 regularization)
                if group["weight_decay"] > 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(g, alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)


class DistMuon(torch.optim.Optimizer):
    """
    Muon: SGD-momentum + (optional) Nesterov, then orthogonalize the 2D update via Newton–Schulz,
    finally apply aspect-ratio scaled step. Performs its own distributed synchronization:
      - reduce_scatter(AVG) for gradient averaging
      - all_gather to replicate updated weights

    Notes:
      * Designed for 2D parameters (e.g., linear/conv kernels reshaped to 2D). Do not use for 0D/1D
        params like embeddings or scalars.
      * Momentum buffers are maintained only on the 'owner' rank for each parameter (rank chosen
        by block-cyclic assignment below). If you checkpoint optimizer state on a single rank,
        consolidate states beforehand.

    Args:
        params: iterable of Tensors
        lr: learning rate
        momentum: momentum coefficient in [0,1)
        nesterov: if True, Nesterov-style update (g <- lerp(g, buf, momentum)); else use buf
        ns_steps: number of Newton–Schulz iterations for the orthogonalization
        weight_decay: L2 weight decay. (default: 0.0)
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        params = list(params)
        assert all(p.ndim == 2 for p in params), "Muon expects 2D parameters only"
        rank = dist.get_rank()
        # Group all parameters by their shape
        shapes = sorted({p.shape for p in params}) # sort to ensure consistent / deterministic ordering
        param_groups = []
        for shape in shapes:
            group_params = [p for p in params if p.shape == shape]
            device, dtype = group_params[0].device, group_params[0].dtype
            assert all(p.device == device for p in group_params)
            assert all(p.dtype == dtype for p in group_params)
            if rank == 0:
                print(f"Muon: Grouping {len(group_params)} params of shape {shape}, device {device}, dtype {dtype}")
            param_groups.append(dict(params=group_params, zero_buffer=torch.zeros_like(group_params[0])))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Ensure all grads exist
        assert all(p.grad is not None for group in self.param_groups for p in group["params"]), "All params must have grads"

        # Kick off all the reduce scatter operations to average up the gradients across all ranks
        all_reduce_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            # Go through params in groups of world_size.
            for base_i in range(0, len(params), world_size):
                # The compute owner of each param is rank i % world_size
                owner_idx = base_i + rank
                # each rank stacks up its chunk of world_size params into a list
                rs_input = [p.grad for p in params[base_i:base_i + world_size]]
                # pad rs_input with the zero buffer to complete the group
                rs_input.extend([zero_buffer] * (world_size - len(rs_input)))
                # the output buffer gets strided across the group based on the rank
                rs_output = params[owner_idx].grad if owner_idx < len(params) else torch.empty_like(zero_buffer)
                # reduce scatter the gradients within this group of world_size params
                work = dist.reduce_scatter(rs_output, rs_input, op=dist.ReduceOp.AVG, async_op=True).get_future()
                all_reduce_futures.append(work)

        # Now each rank computes the update and gathers
        future_idx = 0
        all_gather_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            # Go through params in groups of world_size.
            for base_i in range(0, len(params), world_size):
                # The compute owner of each param is rank i % world_size
                owner_idx = base_i + rank # calculate the index of the param that this rank owns
                # Wait for the reduce scatter to complete
                all_reduce_futures[future_idx].wait() # possibly later we could use wait_any polling instead
                future_idx += 1
                # Owner computes the Muon update, result is in its param
                if owner_idx < len(params):
                    p = params[owner_idx]
                    g = p.grad  # now averaged across ranks
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1.0 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    # Apply weight decay (L2 regularization)
                    if group["weight_decay"] > 0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    scale = (max(1.0, p.size(-2) / p.size(-1)) ** 0.5)
                    p.add_(g, alpha=-group["lr"] * scale)
                # Replicate updated parameters to all ranks
                ag_input = params[owner_idx] if owner_idx < len(params) else zero_buffer
                ag_output = params[base_i:base_i + world_size]
                ag_output.extend([torch.empty_like(zero_buffer) for _ in range(world_size - len(ag_output))]) # pad
                work = dist.all_gather(ag_output, ag_input, async_op=True).get_future()
                all_gather_futures.append(work)

        # Wait for all work to finish
        torch.futures.collect_all(all_gather_futures).wait()


@torch.compile
def row_normalize(G: Tensor, tau: float = 0.0) -> Tensor:
    """
    Row normalization with threshold: normalize each row to have unit L2 norm,
    but only if the row's norm is >= tau.

    Args:
        G: Input tensor to normalize
        tau: Threshold for normalization. Rows with norm < tau are not normalized.
             tau=0 (default) normalizes all rows (original behavior).
    """
    if tau <= 0:
        # Original behavior: normalize all rows
        return F.normalize(G, p=2, dim=-1)
    else:
        # Threshold behavior: only normalize rows with norm >= tau
        row_norms = G.norm(p=2, dim=-1, keepdim=True)  # shape [R, 1]
        # Create mask for rows to normalize
        mask = (row_norms >= tau).float()  # [R, 1]
        # Normalize rows and apply mask
        normalized = F.normalize(G, p=2, dim=-1)
        # Keep original rows where norm < tau, use normalized rows where norm >= tau
        return normalized * mask + G * (1 - mask)

class RNNPS(torch.optim.Optimizer):
    """
    RNNPS - Row-Normalized Nesterov with Polynomial Scaling

    Similar to Muon but uses row normalization instead of Newton-Schulz orthogonalization.
    Each row of the gradient update is normalized to unit L2 norm, subject to a norm threshold.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        beta: The exponential moving average (EMA) coefficient for momentum buffer. (default: 0.95)
        momentum: The momentum coefficient used for Nesterov-style updates. (default: 0.9)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        weight_decay: L2 weight decay. (default: 0.0)
        row_norm_threshold: Threshold for row normalization (tau). Rows with norm < tau are not normalized.
                           tau=0 (default) normalizes all rows (original behavior). (default: 0.0)
        norm_scale_variant: Maximum row norm scaling variant (0-2). (default: 0)
            0: Standard RNNPS (no max row norm scaling)
            1: Linear compensation: scale = default_scale * max_row_norm
            2: Square root compensation: scale = default_scale * sqrt(max_row_norm)
        log_row_norm_stats: Whether to log row norm statistics to console and wandb. (default: False)
        log_row_norm_freq: Frequency (in steps) to log row norm statistics. (default: 100)
    """
    def __init__(self, params, lr=0.02, beta=0.95, momentum=0.9, nesterov=True, weight_decay=0.0, row_norm_threshold=0.0, norm_scale_variant=0, log_row_norm_stats=False, log_row_norm_freq=100):
        defaults = dict(lr=lr, beta=beta, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay, row_norm_threshold=row_norm_threshold, norm_scale_variant=norm_scale_variant)
        params: list[Tensor] = [*params]
        assert all(p.ndim == 2 for p in params), "RNNPS expects 2D parameters only"
        assert norm_scale_variant in [0, 1, 2], f"norm_scale_variant must be 0-2, got {norm_scale_variant}"
        param_groups = []
        for size in {p.numel() for p in params}:
            group = dict(params=[p for p in params if p.numel() == size])
            param_groups.append(group)
        super().__init__(param_groups, defaults)
        self.log_row_norm_stats = log_row_norm_stats
        self.log_row_norm_freq = log_row_norm_freq
        self.step_count = 0

    def _log_row_norm_stats(self, row_norms_list, tau):
        """
        Compute and log row norm statistics.

        Args:
            row_norms_list: List of row norm tensors from all param groups
            tau: Row norm threshold
        """
        if not row_norms_list:
            return

        # Concatenate all row norms
        all_norms = torch.cat([rn.flatten() for rn in row_norms_list])

        # Compute statistics
        mean_norm = all_norms.mean().item()
        std_norm = all_norms.std().item()
        min_norm = all_norms.min().item()
        max_norm = all_norms.max().item()
        median_norm = all_norms.median().item()
        p25_norm = torch.quantile(all_norms, 0.25).item()
        p75_norm = torch.quantile(all_norms, 0.75).item()
        p95_norm = torch.quantile(all_norms, 0.95).item()
        p99_norm = torch.quantile(all_norms, 0.99).item()

        # Count rows affected by threshold
        if tau > 0:
            num_below_tau = (all_norms < tau).sum().item()
            pct_below_tau = 100.0 * num_below_tau / len(all_norms)
        else:
            num_below_tau = 0
            pct_below_tau = 0.0

        # Prepare log message
        log_msg = (
            f"[RNNPS Stats] Step {self.step_count} | "
            f"τ={tau:.6f} | "
            f"mean={mean_norm:.4f} | std={std_norm:.4f} | "
            f"min={min_norm:.4f} | max={max_norm:.4f} | "
            f"median={median_norm:.4f} | "
            f"p25={p25_norm:.4f} | p75={p75_norm:.4f} | "
            f"p95={p95_norm:.4f} | p99={p99_norm:.4f}"
        )
        if tau > 0:
            log_msg += f" | rows<τ={num_below_tau}/{len(all_norms)} ({pct_below_tau:.1f}%)"

        print(log_msg)

        # Log to wandb if available
        if wandb is not None and wandb.run is not None:
            wandb.log({
                "row_norm/mean": mean_norm,
                "row_norm/std": std_norm,
                "row_norm/min": min_norm,
                "row_norm/max": max_norm,
                "row_norm/median": median_norm,
                "row_norm/p25": p25_norm,
                "row_norm/p75": p75_norm,
                "row_norm/p95": p95_norm,
                "row_norm/p99": p99_norm,
            }, step=self.step_count, commit=False)
            if tau > 0:
                wandb.log({
                    "row_norm/threshold": tau,
                    "row_norm/pct_below_threshold": pct_below_tau,
                }, step=self.step_count, commit=False)

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        row_norms_list = [] if self.log_row_norm_stats else None

        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            norm_scale_variant = group["norm_scale_variant"]
            row_norm_threshold = group["row_norm_threshold"]
            for p in params:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf: Tensor = state["momentum_buffer"]
                # EMA update with beta
                buf.lerp_(g, 1 - group["beta"])
                # Nesterov update with momentum (same as Muon)
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                # Compute maximum row norm before normalization
                row_norms = g.norm(p=2, dim=-1)  # shape [R]
                max_row_norm = row_norms.max()  # scalar ν

                # Collect row norms for statistics (before row normalization)
                if self.log_row_norm_stats and self.step_count % self.log_row_norm_freq == 0:
                    row_norms_list.append(row_norms)

                g = row_normalize(g, tau=row_norm_threshold)  # Row normalization with threshold
                # Compute scale based on variant
                default_scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                if norm_scale_variant == 0:
                    scale = default_scale
                elif norm_scale_variant == 1:
                    # Linear compensation: multiply by max_row_norm
                    compensation = 1.0 + 0.5 * math.log(max(1.01, max_row_norm))
                    scale = default_scale * compensation
                elif norm_scale_variant == 2:
                    # Square root compensation: multiply by sqrt(max_row_norm)
                    compensation = 1.0 + 0.6 * math.log(max(1.01, max_row_norm))
                    scale = default_scale * compensation
                else:
                    raise ValueError(f"Invalid norm_scale_variant: {norm_scale_variant}")
                # Apply weight decay (L2 regularization)
                if group["weight_decay"] > 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(g, alpha=-group["lr"] * scale)

        # Log statistics if enabled and at the right frequency
        if self.log_row_norm_stats and self.step_count % self.log_row_norm_freq == 0:
            # Use the row_norm_threshold from the first group (assumes all groups have the same threshold)
            tau = self.param_groups[0]["row_norm_threshold"] if self.param_groups else 0.0
            self._log_row_norm_stats(row_norms_list, tau) 


class DistRNNPS(torch.optim.Optimizer):
    """
    RNNPS: SGD-momentum + (optional) Nesterov, then row-normalize the 2D update,
    finally apply aspect-ratio scaled step. Performs its own distributed synchronization:
      - reduce_scatter(AVG) for gradient averaging
      - all_gather to replicate updated weights

    Notes:
      * Designed for 2D parameters (e.g., linear/conv kernels reshaped to 2D). Do not use for 0D/1D
        params like embeddings or scalars.
      * Momentum buffers are maintained only on the 'owner' rank for each parameter (rank chosen
        by block-cyclic assignment below). If you checkpoint optimizer state on a single rank,
        consolidate states beforehand.

    Args:
        params: iterable of Tensors
        lr: learning rate
        beta: the exponential moving average (EMA) coefficient for momentum buffer. (default: 0.95)
        momentum: momentum coefficient used for Nesterov-style updates in [0,1). (default: 0.9)
        nesterov: if True, Nesterov-style update (g <- lerp(g, buf, momentum)); else use buf
        weight_decay: L2 weight decay. (default: 0.0)
        row_norm_threshold: Threshold for row normalization (tau). Rows with norm < tau are not normalized.
                           tau=0 (default) normalizes all rows (original behavior). (default: 0.0)
        norm_scale_variant: Maximum row norm scaling variant (0-2). (default: 0)
            0: Standard RNNPS (no max row norm scaling)
            1: Linear compensation: scale = default_scale * max_row_norm
            2: Square root compensation: scale = default_scale * sqrt(max_row_norm)
        log_row_norm_stats: Whether to log row norm statistics to console and wandb. (default: False)
        log_row_norm_freq: Frequency (in steps) to log row norm statistics. (default: 100)
    """
    def __init__(self, params, lr: float = 0.02, beta: float = 0.95, momentum: float = 0.9,
                 nesterov: bool = True, weight_decay: float = 0.0, row_norm_threshold: float = 0.0, norm_scale_variant: int = 0, log_row_norm_stats: bool = False, log_row_norm_freq: int = 100):
        defaults = dict(lr=lr, beta=beta, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay, row_norm_threshold=row_norm_threshold, norm_scale_variant=norm_scale_variant)
        params = list(params)
        assert all(p.ndim == 2 for p in params), "RNNPS expects 2D parameters only"
        assert norm_scale_variant in [0, 1, 2], f"norm_scale_variant must be 0-2, got {norm_scale_variant}"
        rank = dist.get_rank()
        # Group all parameters by their shape
        shapes = sorted({p.shape for p in params}) # sort to ensure consistent / deterministic ordering
        param_groups = []
        for shape in shapes:
            group_params = [p for p in params if p.shape == shape]
            device, dtype = group_params[0].device, group_params[0].dtype
            assert all(p.device == device for p in group_params)
            assert all(p.dtype == dtype for p in group_params)
            if rank == 0:
                print(f"RNNPS: Grouping {len(group_params)} params of shape {shape}, device {device}, dtype {dtype}")
            param_groups.append(dict(params=group_params, zero_buffer=torch.zeros_like(group_params[0])))
        super().__init__(param_groups, defaults)
        self.log_row_norm_stats = log_row_norm_stats
        self.log_row_norm_freq = log_row_norm_freq
        self.step_count = 0

    def _log_row_norm_stats(self, row_norms_list, tau):
        """
        Compute and log row norm statistics (DistRNNPS version).

        Args:
            row_norms_list: List of row norm tensors from all param groups
            tau: Row norm threshold
        """
        if not row_norms_list:
            return

        # Concatenate all row norms
        all_norms = torch.cat([rn.flatten() for rn in row_norms_list])

        # Compute statistics
        mean_norm = all_norms.mean().item()
        std_norm = all_norms.std().item()
        min_norm = all_norms.min().item()
        max_norm = all_norms.max().item()
        median_norm = all_norms.median().item()
        p25_norm = torch.quantile(all_norms, 0.25).item()
        p75_norm = torch.quantile(all_norms, 0.75).item()
        p95_norm = torch.quantile(all_norms, 0.95).item()
        p99_norm = torch.quantile(all_norms, 0.99).item()

        # Count rows affected by threshold
        if tau > 0:
            num_below_tau = (all_norms < tau).sum().item()
            pct_below_tau = 100.0 * num_below_tau / len(all_norms)
        else:
            num_below_tau = 0
            pct_below_tau = 0.0

        # Prepare log message
        log_msg = (
            f"[DistRNNPS Stats] Step {self.step_count} | "
            f"τ={tau:.6f} | "
            f"mean={mean_norm:.4f} | std={std_norm:.4f} | "
            f"min={min_norm:.4f} | max={max_norm:.4f} | "
            f"median={median_norm:.4f} | "
            f"p25={p25_norm:.4f} | p75={p75_norm:.4f} | "
            f"p95={p95_norm:.4f} | p99={p99_norm:.4f}"
        )
        if tau > 0:
            log_msg += f" | rows<τ={num_below_tau}/{len(all_norms)} ({pct_below_tau:.1f}%)"

        # Only print from rank 0
        rank = dist.get_rank()
        if rank == 0:
            print(log_msg)

        # Log to wandb if available (only from rank 0)
        if rank == 0 and wandb is not None and wandb.run is not None:
            wandb.log({
                "row_norm/mean": mean_norm,
                "row_norm/std": std_norm,
                "row_norm/min": min_norm,
                "row_norm/max": max_norm,
                "row_norm/median": median_norm,
                "row_norm/p25": p25_norm,
                "row_norm/p75": p75_norm,
                "row_norm/p95": p95_norm,
                "row_norm/p99": p99_norm,
            }, step=self.step_count, commit=False)
            if tau > 0:
                wandb.log({
                    "row_norm/threshold": tau,
                    "row_norm/pct_below_threshold": pct_below_tau,
                }, step=self.step_count, commit=False)

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        row_norms_list = [] if self.log_row_norm_stats else None

        # Ensure all grads exist
        assert all(p.grad is not None for group in self.param_groups for p in group["params"]), "All params must have grads"

        # Kick off all the reduce scatter operations to average up the gradients across all ranks
        all_reduce_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            # Go through params in groups of world_size.
            for base_i in range(0, len(params), world_size):
                # The compute owner of each param is rank i % world_size
                owner_idx = base_i + rank
                # each rank stacks up its chunk of world_size params into a list
                rs_input = [p.grad for p in params[base_i:base_i + world_size]]
                # pad rs_input with the zero buffer to complete the group
                rs_input.extend([zero_buffer] * (world_size - len(rs_input)))
                # the output buffer gets strided across the group based on the rank
                rs_output = params[owner_idx].grad if owner_idx < len(params) else torch.empty_like(zero_buffer)
                # reduce scatter the gradients within this group of world_size params
                work = dist.reduce_scatter(rs_output, rs_input, op=dist.ReduceOp.AVG, async_op=True).get_future()
                all_reduce_futures.append(work)

        # Now each rank computes the update and gathers
        future_idx = 0
        all_gather_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            # Go through params in groups of world_size.
            for base_i in range(0, len(params), world_size):
                # The compute owner of each param is rank i % world_size
                owner_idx = base_i + rank # calculate the index of the param that this rank owns
                # Wait for the reduce scatter to complete
                all_reduce_futures[future_idx].wait() # possibly later we could use wait_any polling instead
                future_idx += 1
                # Owner computes the RNNPS update, result is in its param
                if owner_idx < len(params):
                    p = params[owner_idx]
                    g = p.grad  # now averaged across ranks
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    # EMA update with beta
                    buf.lerp_(g, 1.0 - group["beta"])
                    # Nesterov update with momentum (same as Muon)
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    # Compute maximum row norm before normalization
                    row_norms = g.norm(p=2, dim=-1)  # shape [R]
                    max_row_norm = row_norms.max()  # scalar ν

                    # Collect row norms for statistics (before row normalization)
                    if self.log_row_norm_stats and self.step_count % self.log_row_norm_freq == 0:
                        row_norms_list.append(row_norms)

                    g = row_normalize(g, tau=group["row_norm_threshold"])  # Row normalization with threshold
                    # Compute scale based on variant
                    default_scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                    norm_scale_variant = group["norm_scale_variant"]
                    if norm_scale_variant == 0:
                        scale = default_scale
                    elif norm_scale_variant == 1:
                        # Linear compensation: multiply by max_row_norm
                        compensation = 1.0 + 0.5 * math.log(max(1.01, max_row_norm))
                        scale = default_scale * compensation
                    elif norm_scale_variant == 2:
                        # Square root compensation: multiply by sqrt(max_row_norm)
                        compensation = 1.0 + 0.6 * math.log(max(1.01, max_row_norm))
                        scale = default_scale * compensation
                    else:
                        raise ValueError(f"Invalid norm_scale_variant: {norm_scale_variant}")
                    # Apply weight decay (L2 regularization)
                    if group["weight_decay"] > 0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(g, alpha=-group["lr"] * scale)
                # Replicate updated parameters to all ranks
                ag_input = params[owner_idx] if owner_idx < len(params) else zero_buffer
                ag_output = params[base_i:base_i + world_size]
                ag_output.extend([torch.empty_like(zero_buffer) for _ in range(world_size - len(ag_output))]) # pad
                work = dist.all_gather(ag_output, ag_input, async_op=True).get_future()
                all_gather_futures.append(work)

        # Wait for all work to finish
        torch.futures.collect_all(all_gather_futures).wait()

        # Log statistics if enabled and at the right frequency
        if self.log_row_norm_stats and self.step_count % self.log_row_norm_freq == 0:
            # Use the row_norm_threshold from the first group (assumes all groups have the same threshold)
            tau = self.param_groups[0]["row_norm_threshold"] if self.param_groups else 0.0
            self._log_row_norm_stats(row_norms_list, tau)
