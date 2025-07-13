import enum
import traceback
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor

from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.utils.kl_controller import AdaptiveKLController
from roll.utils.logging import get_logger

import logging


logger = get_logger()
logger.setLevel(logging.DEBUG)


def tensor_to_cpu_visitor(obj, path):
    if torch.is_tensor(obj):
        if not obj.is_cpu:
            obj.data = obj.data.detach().cpu()
        return True
    return False


def tensor_to_cuda_visitor(obj, path):
    if torch.is_tensor(obj):
        if not obj.is_cuda:
            obj.data = obj.data.detach().to(device=torch.device("cuda"))
        return True
    return False


def delete_tensor_grad_visitor(obj, path):
    if torch.is_tensor(obj):
        obj.grad = None
        return True
    return False


def traverse_obj(value, visitor, path=()):
    """
    遍历对象的所有属性，包括属性的属性，找到所有的 Tensor。
    :param value: 任意 Python 对象
    :visitor
    :path
    """
    if visitor(value, path):
        return
    elif isinstance(value, dict):
        for key, value in value.items():
            traverse_obj(value, visitor, path + (str(key),))
    elif isinstance(value, list) or isinstance(value, tuple):
        for index, item in enumerate(value):
            traverse_obj(item, visitor, path + (index,))
    elif hasattr(value, "__dict__"):
        for attr_name in dir(value):
            if not attr_name.startswith("__"):
                try:
                    attr_value = getattr(value, attr_name)
                    traverse_obj(attr_value, visitor, path + (f"attr:{attr_name}",))
                except Exception as e:
                    logger.error(e)
                    continue


def union_two_dict(dict1: Dict, dict2: Dict):
    """Union two dict. Will throw an error if there is an item not the same object with the same key.

    Args:
        dict1:
        dict2:

    Returns:

    """
    for key, val in dict2.items():
        if key in dict1:
            assert dict2[key] == dict1[key], f"{key} in meta_dict1 and meta_dict2 are not the same object"
        dict1[key] = val

    return dict1


def divide_by_chunk_size(
    data: Union[np.ndarray, TensorDict], chunk_sizes: List[int]
) -> List[Union[np.ndarray, TensorDict]]:
    """
    将numpy数组按照chunks的大小切分
    """
    if not isinstance(data, (np.ndarray, TensorDict)):
        raise TypeError("Input 'array' must be a numpy ndarray or a TensorDict.")

    if not all(isinstance(size, int) and size > 0 for size in chunk_sizes):
        raise ValueError("All chunk sizes must be positive integers.")

    total_size = sum(chunk_sizes)
    if total_size != len(data):
        raise ValueError(f"The sum of chunk_sizes ({total_size}) does not match the size of the array ({len(data)}).")

    split_data = []
    start_index = 0
    for size in chunk_sizes:
        end_index = start_index + size
        split_data.append(data[start_index:end_index])
        start_index = end_index
    return split_data


def append_to_dict(data: Dict, new_data: Dict):
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)


class RunningMoments:
    def __init__(self):
        """
        Calculates the running mean and standard deviation of a data stream. Modified version of
        https://github.com/DLR-RM/stable-baselines3/blob/a6f5049a99a4c21a6f0bcce458ca3306cef310e0/stable_baselines3/common/running_mean_std.py
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

    @torch.no_grad()
    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """
        Updates running moments from batch's moments computed across ranks
        """
        xs_count = xs.numel()
        xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_mean, xs_var = xs_mean.float(), xs_var.float()

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).float().sqrt()
        self.count = tot_count

        return xs_mean.item(), (xs_var * xs_count / (xs_count - 1)).float().sqrt().item()


def compute_clip_fraction(values: torch.Tensor, clip_max: float, clip_min: float):
    numel = values.numel()
    num_clipped = (values > clip_max).sum().item() + (values < clip_min).sum().item()
    clipfrac = num_clipped / numel if numel > 0 else 0.0
    return clipfrac


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    kl_penalty: str = "kl",
) -> torch.Tensor:
    """
    ref: https://github.com/OpenRLHF/OpenRLHF/blob/494850f50342ed38d5ae76ef45a3207f3523b582/openrlhf/models/utils.py#L7
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html
    """
    if kl_penalty == "kl":
        log_ratio = log_probs - log_probs_base
    elif kl_penalty == "abs":
        log_ratio = (log_probs - log_probs_base).abs()
    elif kl_penalty == "mse":
        log_ratio = 0.5 * (log_probs - log_probs_base).square()
    elif kl_penalty == "k3":
        kl = log_probs_base - log_probs
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        log_ratio = torch.clamp(kld, min=-10, max=10)
    elif kl_penalty == "full":
        log_ratio = F.kl_div(log_probs_base, log_probs, log_target=True, reduction="none").sum(-1)
    else:
        raise NotImplementedError

    if action_mask is not None:
        return log_ratio * action_mask

    return log_ratio


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    logits = logits.float()
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str,
             weights: Optional[torch.Tensor] = None):
    """
    ref: https://github.com/volcengine/verl/blob/78532923368aeb058f62201489546d013df47710/verl/trainer/ppo/core_algos.py#L370
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "seq-mean-token-sum" is the default behavior
        weights: `torch.Tensor`
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        if weights is None:
            weights = torch.ones(loss_mask.shape[0], device=loss_mask.device)
        loss = masked_mean(loss_mat * weights.unsqueeze(-1), loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = masked_mean(loss_mat, loss_mask, dim=-1) # token-sum
        valid_samples = torch.any(loss_mask > 0, dim=-1).float()
        if weights is None:
            weights = torch.ones(loss_mask.shape[0], device=loss_mask.device)
        loss = (seq_losses * weights * valid_samples).sum() / (valid_samples.sum() + 1e-8) # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = masked_mean(loss_mat, loss_mask, dim=-1)
        seq_losses = seq_losses / (torch.sum(loss_mask, dim=-1) + 1e-8)  # token-mean
        valid_samples = torch.any(loss_mask > 0, dim=-1).float()
        if weights is None:
            weights = torch.ones(loss_mask.shape[0], device=loss_mask.device)
        loss = (seq_losses * weights * valid_samples).sum() / (valid_samples.sum() + 1e-8)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = masked_mean(loss_mat, loss_mask, dim=-1)
        valid_samples = torch.any(loss_mask > 0, dim=-1).float()
        if weights is None:
            weights = torch.ones(loss_mask.shape[0], device=loss_mask.device)
        loss = (seq_losses * weights * valid_samples).sum() / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = None) -> torch.Tensor:
    if dim is not None:
        mask_sum = mask.sum(axis=dim)
        return torch.where(mask_sum > 0, (tensor * mask).sum(axis=dim) / (mask_sum + 1e-8), torch.zeros_like(mask_sum))
    else:
        return (
            (tensor * mask).sum() / (mask.sum() + 1e-8) if mask.sum() > 0 else torch.tensor(0.0, device=tensor.device)
        )


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def get_eos_mask(response_id: torch.Tensor, eos_token: int = 2, dtype=torch.int64):
    """
    e.g. end of sentence token=1
    response_id: [0, 0, 2, 42, 3, 5, 1, 0, 0]
    eos_mask:     [1, 1, 1, 1,  1, 1, 1, 0, 0]
    """
    eos_mask = response_id.eq(eos_token).long()
    eos_mask = (torch.cumsum(eos_mask, dim=1) - eos_mask).bool()
    eos_mask = torch.logical_not(eos_mask).to(dtype)
    return eos_mask


def get_pad_mask(response_id: torch.Tensor, pad_token: int = 0, dtype=torch.int64):
    """
    e.g. pad token=0
    response_id: [1, 2, 2, 42, 3, 5, 1, 0, 0]
    pad_mask:     [1, 1, 1, 1,  1, 1, 1, 0, 0]
    """
    pad_mask = response_id.not_equal(pad_token).to(dtype)
    assert (
        not (pad_mask[:, 0] == 0).logical_and(pad_mask.sum(-1) != 0).any()
    ), f"response_id is not valid: {response_id}, pad_token is {pad_token}"
    return pad_mask


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim).unsqueeze(-1)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim).unsqueeze(-1)
    return mean_centered * var.clamp(min=eps).rsqrt()


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def response_level_masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True):
    """Whiten values with masked values."""
    # 考虑response的影响？
    mean = masked_mean(values, mask, dim=-1)
    var = masked_var(mean, mask)
    mean = mean.mean()
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def reduce_metrics(metrics: dict, reduce_func=np.mean) -> dict:
    for key, val in metrics.items():
        metrics[key] = reduce_func(val)
    return metrics


def pad_to_length(tensor: torch.Tensor, length, pad_value, dim=-1):
    if tensor.size(dim) >= length:
        indices = [slice(None)] * tensor.ndim
        indices[dim] = slice(0, length)
        return tensor[indices]
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim
        )


def concatenate_input_and_output(input_ids, output_ids, num_return_sequences):
    batch_size, input_seq_len = input_ids.size()
    _, output_seq_len = output_ids.size()
    repeated_input_ids = (
        input_ids.unsqueeze(1)
        .repeat(1, num_return_sequences, 1)
        .view(batch_size * num_return_sequences, input_seq_len)
    )
    sequences = torch.cat((repeated_input_ids, output_ids), dim=1)
    return sequences


def compute_reinforce_return(token_level_rewards: torch.Tensor, gamma: torch.Tensor, lambd: torch.Tensor):
    with torch.no_grad():
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]
        cumulative_reward = 0
        for t in reversed(range(gen_len)):
            local_reward = token_level_rewards[:, t] if t < gen_len else 0.0
            cumulative_reward = local_reward + gamma * cumulative_reward
            advantages_reversed.append(cumulative_reward)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages
    return advantages, returns


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor, values: torch.Tensor, gamma: torch.Tensor, lambd: torch.Tensor
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        gamma: `(float)`
            discounted factor used in RL
        lambd: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values

    return advantages, returns


def expand_to_token_level(data: "DataProto"):
    response_level_rewards = data.batch["response_level_rewards"].clone().detach()
    batch_size = data.batch.batch_size[0]
    # expand as token_level_rewards
    attention_mask = data.batch["attention_mask"]
    position_ids = data.batch["position_ids"]
    if position_ids.dim() == 3:
        # qwen2vl, (bsz, 3, seqlen), 0/1/2 is same for text, while values of
        # position_ids for text cannot stand for index of tokens, thus use the
        # right padding attention_mask to calculate eos index or `argmax` rather
        # than `max` of position_ids to calculate eos index
        position_ids = position_ids[:, 0]
    eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
    token_level_rewards = torch.zeros_like(attention_mask, dtype=response_level_rewards.dtype)  # (bsz, seqlen)

    token_level_rewards[torch.arange(batch_size), eos_mask_idx] = response_level_rewards

    # select the response part
    token_level_rewards = token_level_rewards[:, 1:]

    return token_level_rewards


def batch_reward_norm(response_level_rewards: torch.Tensor, div_std=True):
    batch_mean = response_level_rewards.mean()
    if div_std:
        normalized_rewards = (response_level_rewards - batch_mean) / (response_level_rewards.std() + 1e-6)
    else:
        normalized_rewards = response_level_rewards - batch_mean
    return normalized_rewards


def group_reward_norm(data: "DataProto", n_sample=-1, div_std=True, div_std_global=False):
    assert n_sample > 1, "n_sample must > 1"
    response_level_rewards = data.batch["response_level_rewards"].clone().detach()
    reshape_reward = response_level_rewards.reshape(*response_level_rewards.size()[:-1], -1, n_sample)
    reshape_reward = reshape_reward - reshape_reward.mean(dim=-1, keepdim=True)
    if div_std:
        if not div_std_global:
            reshape_reward = reshape_reward / (torch.std(reshape_reward, dim=-1, keepdim=True) + 1e-6)
        else:
            reshape_reward = reshape_reward / (torch.std(reshape_reward) + 1e-6)
    data.batch["response_level_rewards"] = reshape_reward.reshape(*response_level_rewards.size())
    return data


def difficulty_mask(data: "DataProto", n_sample=-1, low_threshold=0.1, high_threshold=0.95):
    if n_sample > 1:
        scores = data.batch["scores"].clone().detach()
        reshape_score = scores.reshape(*scores.size()[:-1], -1, n_sample)
        reshape_score_mean = reshape_score.mean(dim=-1, keepdim=True).expand_as(reshape_score).reshape(*scores.size())
        data.batch["difficulty_mask"] = (reshape_score_mean > low_threshold) * (reshape_score_mean < high_threshold)
    else:
        data.batch["difficulty_mask"] = torch.ones_like(data.batch["scores"])
    return data


@torch.no_grad()
def compute_token_reward(data: "DataProto", pipeline_config: RLVRConfig, kl_ctrl: AdaptiveKLController):
    token_level_rewards = expand_to_token_level(data)
    beta = 0
    kld = compute_approx_kl(
        log_probs=data.batch["old_log_probs"],
        log_probs_base=data.batch["ref_log_probs"],
        action_mask=data.batch["response_mask"][:, 1:],
        kl_penalty=pipeline_config.kl_penalty,
    )
    # 是否添加token level kl
    if pipeline_config.add_token_level_kl and "ref_log_probs" in data.batch.keys():
        beta = kl_ctrl.value
        token_level_rewards = token_level_rewards - beta * kld

    current_kl = masked_mean(kld, mask=data.batch["response_mask"][:, 1:], dim=-1)
    current_kl = torch.mean(current_kl, dim=0).item()

    kl_ctrl.update(current=current_kl, n_steps=data.batch.batch_size[0])
    if "token_level_rewards" in data.batch.keys():
        data.rename(old_keys="token_level_rewards", new_keys="token_level_scores")
    metrics = {"critic/kl": current_kl, "critic/kl_coef": beta}

    if pipeline_config.reward_clip:
        reward_clip_frac = compute_clip_fraction(
            values=token_level_rewards, clip_max=pipeline_config.reward_clip, clip_min=-pipeline_config.reward_clip
        )
        metrics["critic/token_reward_clip_frac"] = reward_clip_frac
        token_level_rewards = torch.clamp(
            token_level_rewards, min=-pipeline_config.reward_clip, max=pipeline_config.reward_clip
        )

    data.batch["token_level_rewards"] = token_level_rewards
    return data, metrics


@torch.no_grad()
def reward_postprocess(data: "DataProto", pipeline_config: RLVRConfig, running_ctrl):
    response_level_rewards = data.batch["response_level_rewards"].clone().detach()
    response_level_metrics = {"critic/reward_clip_frac": 0.0}
    # 对reward进行处理: 可以选择不同的normalization方法
    # 使用group-based normalization (按prompt分组)
    if pipeline_config.adv_estimator == "grpo" or pipeline_config.reward_norm == "group":
        if pipeline_config.reward_shift:
            data = group_reward_norm(
                data,
                n_sample=pipeline_config.actor_infer.generating_args.num_return_sequences,
                div_std=False,
            )
        else:
            data = group_reward_norm(
                data,
                n_sample=pipeline_config.actor_infer.generating_args.num_return_sequences,
                div_std=True,
            )
        response_level_rewards = data.batch["response_level_rewards"].clone().detach()

    # 使用batch-based normalization (整个batch)
    elif pipeline_config.reward_norm == "batch":
        if hasattr(pipeline_config, "reward_shift") and pipeline_config.reward_shift:
            response_level_rewards = batch_reward_norm(response_level_rewards, div_std=False)
        else:
            response_level_rewards = batch_reward_norm(response_level_rewards, div_std=True)

    # 使用running statistics进行normalization
    elif pipeline_config.reward_norm == "running":
        running = running_ctrl["domain"]
        running.update(response_level_rewards)
        mean = running.mean
        std = running.std + torch.finfo(response_level_rewards.dtype).eps
        if pipeline_config.reward_shift:
            response_level_rewards = response_level_rewards - mean
        elif pipeline_config.reward_scale:
            response_level_rewards = response_level_rewards / std
        else:
            response_level_rewards = (response_level_rewards - mean) / std

    # 对reward进行clip
    if pipeline_config.reward_clip:
        reward_clip_frac = compute_clip_fraction(
            values=response_level_rewards, clip_max=pipeline_config.reward_clip, clip_min=-pipeline_config.reward_clip
        )
        response_level_rewards = torch.clamp(
            response_level_rewards, min=-pipeline_config.reward_clip, max=pipeline_config.reward_clip
        )

        response_level_metrics = {"critic/reward_clip_frac": reward_clip_frac}

    data.batch["response_level_rewards"] = response_level_rewards
    return data, response_level_metrics


@torch.no_grad()
def get_sample_level_mask(data: "DataProto", pipeline_config: RLVRConfig):
    batch_size = data.batch["response_mask"].size(0)
    mask_metrics = {}

    # mask相关策略
    data.batch["origin_response_mask"] = data.batch["response_mask"].clone()
    response_mask = data.batch["response_mask"][:, 1:].clone()
    true_response_length = response_mask.sum(-1).float()
    max_response_length = data.batch["responses"].shape[-1]

    final_sample_mask = torch.ones(batch_size, device=response_mask.device)

    # 1. max_len_mask: 过滤掉超过最大长度的样本
    if pipeline_config.max_len_mask:
        max_len_mask = (max_response_length != true_response_length).float()
        final_sample_mask = final_sample_mask * max_len_mask
        mask_metrics["actor/max_len_mask_ratio"] = max_len_mask.mean().item()
    else:
        mask_metrics["actor/max_len_mask_ratio"] = 1.0

    # 2. difficulty_mask: 基于难度的过滤
    if pipeline_config.difficulty_mask:
        data = difficulty_mask(
            data,
            n_sample=pipeline_config.actor_infer.generating_args.num_return_sequences,
            low_threshold=pipeline_config.difficulty_low_threshold,
            high_threshold=pipeline_config.difficulty_high_threshold,
        )
        if "difficulty_mask" in data.batch:
            difficulty_mask_tensor = data.batch["difficulty_mask"].float()
            final_sample_mask = final_sample_mask * difficulty_mask_tensor
            mask_metrics["actor/difficulty_mask_ratio"] = difficulty_mask_tensor.mean().item()
        else:
            mask_metrics["actor/difficulty_mask_ratio"] = 1.0
    else:
        mask_metrics["actor/difficulty_mask_ratio"] = 1.0

    # 3. error_max_len_clip: 基于错误和长度的过滤
    if pipeline_config.error_max_len_clip:
        scores = data.batch["scores"]
        error_len_mask = ((scores == 0) & (true_response_length < pipeline_config.error_max_len_threshold)) | (
            scores == 1
        )
        error_len_mask = error_len_mask.float()
        final_sample_mask = final_sample_mask * error_len_mask
        mask_metrics["actor/error_len_mask_ratio"] = error_len_mask.mean().item()
    else:
        mask_metrics["actor/error_len_mask_ratio"] = 1.0

    expanded_sample_mask = final_sample_mask.unsqueeze(-1).expand_as(response_mask)
    final_response_mask = response_mask * expanded_sample_mask
    if final_response_mask.sum() == 0:
        final_response_mask = data.batch["response_mask"][:, 1:].clone()
    mask_metrics["actor/final_mask_ratio"] = final_sample_mask.mean().item()
    mask_metrics["actor/samples_used"] = final_sample_mask.sum().item()
    mask_metrics["actor/samples_total"] = float(batch_size)

    data.batch["final_response_mask"] = final_response_mask
    return data, mask_metrics


@torch.no_grad()
def apply_kl_penalty(data: "DataProto", kl_ctrl: AdaptiveKLController, kl_penalty="kl"):
    response_mask = data.batch["response_mask"][:, 1:]

    token_level_rewards = expand_to_token_level(data)
    if "token_level_rewards" in data.batch.keys():
        data.rename(old_keys="token_level_rewards", new_keys="token_level_scores")

    batch_size = data.batch.batch_size[0]

    if "ref_log_probs" in data.batch.keys():
        kld = compute_approx_kl(
            log_probs=data.batch["old_log_probs"],
            log_probs_base=data.batch["ref_log_probs"],
            action_mask=response_mask,
            kl_penalty=kl_penalty,
        )  # (batch_size, seq_len-1)
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_rewards - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    kl_ctrl.update(current=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"critic/kl": current_kl, "critic/kl_coef": beta}

    return data, metrics


@torch.no_grad()
def compute_advantage(
    data: "DataProto",
    gamma,
    lambd,
    adv_estimator,
    advantage_clip=None,
    whiten_advantages=False,
    whiten_rewards=False,
    response_mask=None,
):
    if response_mask is None:
        response_mask = data.batch["response_mask"][:, 1:]

    token_level_rewards = data.batch["token_level_rewards"].float()
    if whiten_rewards:
        token_level_rewards = masked_whiten(values=token_level_rewards, mask=response_mask)
    # data.batch['token_level_rewards'] = token_level_rewards
    token_level_rewards = token_level_rewards * response_mask
    data.batch["token_level_rewards"] = token_level_rewards
    if adv_estimator == "gae":
        values = data.batch["values"].float()
        data.batch["values"] = values * response_mask
        advantages, returns = compute_gae_advantage_return(
            token_level_rewards=token_level_rewards, values=values, gamma=gamma, lambd=lambd
        )
    elif adv_estimator == "reinforce":
        advantages, returns = compute_reinforce_return(
            token_level_rewards=token_level_rewards, gamma=gamma, lambd=lambd
        )
    elif adv_estimator == "grpo":
        advantages, returns = compute_reinforce_return(
            token_level_rewards=token_level_rewards, gamma=gamma, lambd=lambd
        )
    else:
        raise NotImplementedError

    data.batch["raw_advantages"] = advantages
    if whiten_advantages:
        # TODO whiten过程中是否要考虑response的长度？
        advantages = masked_whiten(values=advantages, mask=response_mask)
    advantages = advantages * response_mask

    if advantage_clip is not None:
        adv_clip_frac = compute_clip_fraction(values=advantages, clip_min=-advantage_clip, clip_max=advantage_clip)
        data.meta_info["metrics"] = {"critic/advantage_clip_frac": adv_clip_frac}
        advantages = torch.clamp(advantages, min=-advantage_clip, max=advantage_clip)

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class GenerateRequestType(enum.Enum):
    ADD = enum.auto()
    ABORT = enum.auto()
    STOP = enum.auto()
    ALIVE_CHECK = enum.auto()
    INTERRUPT = enum.auto()


def postprocess_generate(
    prompts: "DataProto",
    output: torch.Tensor,
    num_return_sequences,
    sequence_length,
    canonical_prompt_length: int,
    eos_token_id,
    pad_token_id,
    fill_eos_token=False,
) -> "DataProto":
    from roll.distributed.scheduler.protocol import DataProto

    output_ids = output
    output_ids = pad_to_length(output_ids, sequence_length, pad_token_id)

    if fill_eos_token:
        # yali: 如果output最后一个token不是pad_token_id，则替换成eos_token_id,
        #  TODO: 需要消融这个变化的影响
        last_token_index = output_ids.size(1) - 1
        need_replace_mask = output_ids[:, last_token_index] != pad_token_id
        output_ids[need_replace_mask, last_token_index] = eos_token_id

    # Extract batch info from the input prompts
    # Note: `prompts` can be a single DataProto for a normal batch, or a list of
    # DataProto for a batch containing interrupted requests. We handle this later.
    is_continued_request = prompts.meta_info.get("is_continued_request", False)
    batch_size = output_ids.shape[0]
    device = output_ids.device

    # --- Determine Prompt Length and Pad to Canonical Shape ---
    # This is the most complex part. For normal requests, all prompts have the
    # same canonical length. For interrupted requests, they have varying lengths
    # and must be padded to the canonical length before being combined.

    # This logic assumes a heterogeneous batch has already been handled upstream
    # and that for this function call, all prompts either are or will be padded
    # to the canonical_prompt_length.
    
    if is_continued_request:
        # For an interrupted request, the 'prompts' object contains the original,
        # unpadded prompt. We must pad or truncate it to the canonical length.
        prompt_tensor = prompts.batch["input_ids"]
        original_prompt_length = prompt_tensor.shape[1]
        original_attention_mask = prompts.batch.get("attention_mask", prompt_tensor.ne(pad_token_id))

        if original_prompt_length > canonical_prompt_length:
            # Prompt is longer than canonical, truncate from the left.
            prompt = prompt_tensor[:, -canonical_prompt_length:]
            attention_mask = original_attention_mask[:, -canonical_prompt_length:]
        else:
            # Prompt is shorter or equal, pad on the left to match canonical length.
            pad_left = canonical_prompt_length - original_prompt_length
            prompt = F.pad(prompt_tensor, (pad_left, 0), value=pad_token_id)
            attention_mask = F.pad(original_attention_mask, (pad_left, 0), value=0)
    else:
        # For a normal request, the prompt is already at the canonical length.
        prompt = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        assert prompt.shape[1] == canonical_prompt_length, \
            f"Normal prompt length {prompt.shape[1]} must equal canonical {canonical_prompt_length}"

    # --- Final Shape Assertions (Pre-Computation) ---
    assert prompt.shape[1] == canonical_prompt_length
    assert attention_mask.shape[1] == canonical_prompt_length
    assert output_ids.shape[1] == sequence_length

    # --- Construct Final Tensors ---
    response = output_ids[:, canonical_prompt_length:]
    canonical_response_length = sequence_length - canonical_prompt_length

    # --- Sanity Checks ---
    assert prompt.shape[0] == response.shape[0], "Batch size mismatch between prompt and response"
    assert response.shape[1] == canonical_response_length, "Response length is incorrect"
    assert prompt.shape[1] + response.shape[1] == sequence_length, "Prompt and response lengths do not sum to sequence length"

    # --- Create Final Masks and Tensors ---
    final_input_ids = torch.cat((prompt, response), dim=1)
    
    response_mask = response.ne(pad_token_id).to(attention_mask.dtype)
    final_attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

    # For continued requests, the attention mask might have a gap of zeros between
    # the end of the prompt and the start of the new response. This fills that gap.
    if is_continued_request:
        for i in range(final_attention_mask.shape[0]):
            valid_length = final_attention_mask[i].sum().int().item()
            final_attention_mask[i][:valid_length] = 1
            final_attention_mask[i][valid_length:] = 0

    position_ids = torch.arange(0, sequence_length, dtype=torch.long, device=device).unsqueeze(0)
    prompt_mask_full = torch.arange(0, sequence_length, device=device)[None, :] < canonical_prompt_length
    response_mask_full = ~prompt_mask_full

    # --- Final Shape Assertions (Post-Computation) ---
    assert final_input_ids.shape == (batch_size, sequence_length)
    assert final_attention_mask.shape == (batch_size, sequence_length)
    assert position_ids.expand(batch_size, -1).shape == (batch_size, sequence_length)
    assert prompt_mask_full.expand(batch_size, -1).shape == (batch_size, sequence_length)
    assert response_mask_full.expand(batch_size, -1).shape == (batch_size, sequence_length)
    
    batch_dict = {
            "prompts": prompt,
            "responses": response,
        "input_ids": final_input_ids,
        "attention_mask": final_attention_mask,
        "position_ids": position_ids.expand(batch_size, -1),
        "prompt_mask": prompt_mask_full.expand(batch_size, -1),
        "response_mask": response_mask_full.expand(batch_size, -1),
        }

    return DataProto(
        meta_info=prompts.meta_info,
        batch=TensorDict(batch_dict, batch_size=[batch_size]),
    )


def get_dist_info_from_comm_plan(comm_plan, rank_in_cluster, rank_in_worker):
    for src_rank, comm_plan_args in comm_plan.items():
        start_rank = 0
        for tgt_device in comm_plan_args["tgt_devices"]:
            start_rank += 1
            if tgt_device["rank"] == rank_in_cluster and tgt_device["device"]["rank"] == rank_in_worker:
                return start_rank, comm_plan_args
    return None, None


def separate_prompt_response(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, response_mask: torch.Tensor, pad_id: int
):
    prompt_mask = attention_mask.bool() & ~response_mask.bool()
    response_mask_valid = attention_mask.bool() & response_mask.bool()
    prompt_ids = torch.where(prompt_mask, input_ids, torch.full_like(input_ids, pad_id))
    response_ids = torch.where(response_mask_valid, input_ids, torch.full_like(input_ids, pad_id))
    return prompt_ids, response_ids
