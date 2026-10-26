"""Explicit reverse-time FM sampler and spatial-mean Gaussian surrogate.

Experimental VP-free schedule sigma(t)=noise*sqrt(t). This follows the
Fokker-Planck drift correction for the linear interpolant, but is NOT a pinned
reproduction of the upstream Flow-GRPO sampler. Last transition is deterministic
and excluded from the policy objective; no stored state is clipped.
"""
from dataclasses import dataclass
import math

import torch


@dataclass
class Transition:
    state: torch.Tensor
    next_state: torch.Tensor
    time: torch.Tensor
    step: float
    std: float
    old_log_prob: torch.Tensor


def transition_parameters(model, state, time, features, step, noise, deterministic=False):
    velocity = model.velocity(state, time, features).float()
    t = time[:, None, None, None].float()
    variance_rate = noise ** 2 * float(time[0]) if not deterministic else 0.0
    # E[z|x_t] = x_t + (1-t) v; score_t = -E[z|x_t]/t.
    score = -(state.float() + (1 - t) * velocity) / t.clamp_min(1e-6)
    mean = state.float() - step * velocity + 0.5 * step * variance_rate * score
    return mean, math.sqrt(variance_rate * step)


def log_probability(next_state, mean, std):
    if std <= 0:
        raise ValueError('Deterministic transitions do not have a Gaussian density')
    values = -0.5 * ((next_state.detach().float() - mean.float()) / std).square()
    values = values - math.log(std) - 0.5 * math.log(2 * math.pi)
    return values.flatten(1).mean(1)


@torch.no_grad()
def sample(model, image, steps=16, noise=0.0, generator=None, record=False):
    if steps < 1 or noise < 0:
        raise ValueError('Steps must be positive and noise nonnegative')
    if record and (steps < 2 or noise == 0):
        raise ValueError('Policy rollouts require >=2 steps and positive SDE noise')
    features = model.encode(image)
    state = torch.randn(image[:, :1].shape, device=image.device, generator=generator)
    trajectory = []
    for i in range(steps):
        time = torch.full((len(image),), 1 - i / steps, device=image.device)
        mean, std = transition_parameters(model, state, time, features, 1 / steps,
                                           noise, deterministic=(i == steps - 1))
        next_state = mean
        if std > 0:
            next_state = mean + std * torch.randn(mean.shape, device=mean.device, generator=generator)
            if record:
                trajectory.append(Transition(state.detach(), next_state.detach(), time,
                                             1 / steps, std,
                                             log_probability(next_state, mean, std)))
        state = next_state
    return ((state + 1) / 2).clamp(0, 1), trajectory


def group_advantages(rewards, epsilon=1e-6, min_std=1e-4):
    rewards = rewards.float()
    std = rewards.std(-1, keepdim=True, unbiased=False)
    normalized = (rewards - rewards.mean(-1, keepdim=True)) / (std + epsilon)
    return torch.where(std >= min_std, normalized, torch.zeros_like(normalized))


def policy_loss(model, reference, features, reference_features, transitions, advantages,
                noise, clip=0.2, beta=0.01):
    if not transitions:
        raise ValueError('No stochastic transitions')
    losses, ratios, kls, clipped = [], [], [], []
    for transition in transitions:
        mean, std = transition_parameters(model, transition.state, transition.time, features,
                                          transition.step, noise)
        if abs(std - transition.std) > 1e-8:
            raise ValueError('Rollout and update variance differ')
        logp = log_probability(transition.next_state, mean, std)
        ratio = (logp - transition.old_log_prob).exp()
        with torch.no_grad():
            ref_mean, _ = transition_parameters(reference, transition.state, transition.time,
                                                reference_features, transition.step, noise)
        kl = ((mean - ref_mean).square() / (2 * std ** 2)).flatten(1).mean(1)
        surrogate = torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages)
        losses.append((-surrogate + beta * kl).mean())
        ratios.append(ratio.detach())
        kls.append(kl.detach())
        clipped.append(((ratio - 1).abs() > clip).float().detach())
    return torch.stack(losses).mean(), dict(
        ratio_mean=torch.cat(ratios).mean().item(),
        ratio_max_error=(torch.cat(ratios) - 1).abs().max().item(),
        kl=torch.cat(kls).mean().item(), clip_fraction=torch.cat(clipped).mean().item())
