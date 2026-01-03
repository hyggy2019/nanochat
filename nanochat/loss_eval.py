"""
A number of functions that help with evaluating a base model.
"""
import math
import torch
import torch.distributed as dist

@torch.no_grad()
def evaluate_loss_and_metrics(model, batches, steps, token_bytes):
    """
    Evaluate the model on a batch of data and return loss, perplexity, and bpb metrics.

    Args:
        model: The model to evaluate
        batches: Iterator over (x, y) batches
        steps: Number of steps to evaluate
        token_bytes: Token byte lengths tensor (1D, shape (vocab_size,))

    Returns:
        A dict containing:
            - 'loss': Mean loss (same as training loss metric)
            - 'perplexity': exp(loss)
            - 'bpb': Bits per byte (tokenization-independent metric)
    """
    # record the losses
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_tokens = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())

    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none') # (B, T)
        loss2d = loss2d.view(-1) # flatten
        y = y.view(-1) # flatten

        # Count valid tokens (those with y >= 0)
        valid = y >= 0

        # Compute total loss
        total_loss += loss2d[valid].sum()
        total_tokens += valid.sum()

        if (y.int() < 0).any(): # mps does not currently have kernel for < 0 for int64, only int32
            # slightly more complex code path if some target tokens are ignore_index (e.g. -1)
            # any target token < 0 is to be ignored: do NOT index token_bytes with negatives
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            # map valid targets to their byte length; ignored targets contribute 0 bytes
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            # fast path: no ignored targets, safe to index directly
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()

    # sum reduce across all ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)

    # move to cpu and calculate metrics
    total_loss = total_loss.item()
    total_tokens = total_tokens.item()
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()

    # Calculate loss and perplexity
    if total_tokens == 0:
        loss = float('inf')
        perplexity = float('inf')
    else:
        loss = total_loss / total_tokens
        perplexity = math.exp(loss)

    # Calculate bpb
    if total_bytes == 0:
        bpb = float('inf')
    else:
        bpb = total_nats / (math.log(2) * total_bytes)

    return {
        'loss': loss,
        'perplexity': perplexity,
        'bpb': bpb,
    }


@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Deprecated: Use evaluate_loss_and_metrics instead.
    Returns only the bits per byte (bpb) metric.
    """
    metrics = evaluate_loss_and_metrics(model, batches, steps, token_bytes)
    return metrics['bpb']
