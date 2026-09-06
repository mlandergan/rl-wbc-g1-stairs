"""WasabiAMP: skrl's own `AMP` agent, with three formula swaps to match WASABI's exact
discriminator training, ported from
`instinct_rl/algorithms/wasabi.py` (MIT-licensed, copyright Ziwen Zhuang -- see
THIRD_PARTY_NOTICES.md) rather than reimplemented from scratch.

IMPORTANT VERSION NOTE (2026-09-05): this file was originally written against skrl's GitHub
`main` branch API (a dataclass-based `AMP_CFG`, a public `update()` method, `"observations"`-
keyed tensors) -- a smoke test on the actual GPU VM revealed the *installed* skrl version is
1.4.3, which is meaningfully different: a plain-dict `AMP_DEFAULT_CONFIG`, a private `_update()`
method (called internally by `post_interaction()`), `"states"`/`"amp_states"`-keyed tensors, and
`.act()` returning 3-tuples. This file was rewritten from scratch against the REAL installed
v1.4.3 source (pulled directly off the VM: `docker run --rm rl-wbc-g1-stairs cat
.../skrl/agents/torch/amp/amp.py`), not the newer main-branch version. Also corrected: the
original yaml "field name fix" (task_reward_weight -> task_reward_scale etc.) was WRONG for this
installed version -- v1.4.3's real config keys are `task_reward_weight`/`style_reward_weight`/
`amp_state_preprocessor`/`clip_predicted_values`, matching what the yaml already had before that
change. Reverted in skrl_g1_stairs_cfg.yaml.

`_update()` below is v1.4.3's real `AMP._update()` copied essentially verbatim -- the nested
`compute_gae` closure, GAE, PPO clipped surrogate, mixed precision, the replay buffer, the single
combined optimizer step, all UNCHANGED. Only three blocks differ, each marked "WASABI:" in a
comment at the point of change:

1. Style reward (`instinct_rl`'s `"quad"` reward type): bounded
   `clamp(1 - 0.25*(D-1)^2, min=0)`, replacing the base version's unbounded
   `-log(1-sigmoid(D))`. Both versions still apply `self._discriminator_reward_scale` afterward
   (matching the base version's own `style_reward *= self._discriminator_reward_scale` line --
   easy to miss, this project's first draft of this file dropped it by mistake).
2. Discriminator loss (`instinct_rl`'s `"MSELoss"`): `MSE(D(fake),-1) + MSE(D(real),+1)`
   (the original AMP paper's LSGAN-style loss), replacing `BCEWithLogitsLoss`.
3. Gradient penalty: WASABI's own formulation -- computed over
   `concat(rollout_states, reference_states)` with a tolerance margin before squaring,
   replacing the base version's real-data-only (motion side only) formulation.

NOT verified past `py_compile` and a partial-construction smoke test (the previous, wrong-API
version of this file got as far as failing to import before this rewrite) -- confirm a full
smoke test (a couple of `_update()` calls, no NaN/shape errors) before trusting a long run.
"""
from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.agents.torch.amp import AMP
from skrl.resources.schedulers.torch import KLAdaptiveLR


class WasabiAMP(AMP):
    """See module docstring. `cfg` is a plain dict, same as the base `AMP` class -- pass
    `discriminator_gradient_tolerance` as an extra key in it (WASABI's own
    default is 0.0, InstinctLab's parkour config didn't override it either) if you want anything
    other than the 0.0 fallback used below.
    """

    def _update(self, timestep: int, timesteps: int) -> None:
        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]
            for i in reversed(range(memory_size)):
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * (next_values[i] + lambda_coefficient * not_dones[i] * advantage)
                )
                advantages[i] = advantage
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return returns, advantages

        # update dataset of reference motions
        self.motion_dataset.add_samples(states=self.collect_reference_motions(self._amp_batch_size))

        # compute combined rewards
        rewards = self.memory.get_tensor_by_name("rewards")
        amp_states = self.memory.get_tensor_by_name("amp_states")

        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            amp_logits, _, _ = self.discriminator.act(
                {"states": self._amp_state_preprocessor(amp_states)}, role="discriminator"
            )
            # WASABI: bounded "quad" style reward (instinct_rl/algorithms/wasabi.py's
            # compute_auxiliary_reward, discriminator_reward_type="quad"), replacing the base
            # version's unbounded -log(1-sigmoid(D)). amp_logits is the raw discriminator output
            # (a direct linear layer, matching wasabi.py's own "assumes disc is the output of a
            # direct linear layer" comment on this exact formula), not passed through sigmoid
            # first. Still scaled by discriminator_reward_scale afterward, matching the base
            # version's own behavior.
            style_reward = torch.clamp(1.0 - 0.25 * torch.square(amp_logits - 1.0), min=0.0)
            style_reward *= self._discriminator_reward_scale
            style_reward = style_reward.view(rewards.shape)

        combined_rewards = self._task_reward_weight * rewards + self._style_reward_weight * style_reward

        # compute returns and advantages
        values = self.memory.get_tensor_by_name("values")
        next_values = self.memory.get_tensor_by_name("next_values")
        returns, advantages = compute_gae(
            rewards=combined_rewards,
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=next_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        sampled_batches = self.memory.sample_all(names=self.tensors_names, mini_batches=self._mini_batches)
        sampled_motion_batches = self.motion_dataset.sample(
            names=["states"], batch_size=self.memory.memory_size * self.memory.num_envs, mini_batches=self._mini_batches
        )
        if len(self.reply_buffer):
            sampled_replay_batches = self.reply_buffer.sample(
                names=["states"],
                batch_size=self.memory.memory_size * self.memory.num_envs,
                mini_batches=self._mini_batches,
            )
        else:
            sampled_replay_batches = [[batches[self.tensors_names.index("amp_states")]] for batches in sampled_batches]

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_discriminator_loss = 0

        for epoch in range(self._learning_epochs):
            kl_divergences = []

            for batch_index, (
                sampled_states,
                sampled_actions,
                _,
                _,
                _,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
                sampled_amp_states,
                _,
            ) in enumerate(sampled_batches):

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                    sampled_states = self._state_preprocessor(sampled_states, train=True)

                    _, next_log_prob, _ = self.policy.act(
                        {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
                    )

                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    if self._entropy_loss_scale:
                        entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")
                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                    if self._discriminator_batch_size:
                        sampled_amp_states = self._amp_state_preprocessor(
                            sampled_amp_states[0 : self._discriminator_batch_size], train=True
                        )
                        sampled_amp_replay_states = self._amp_state_preprocessor(
                            sampled_replay_batches[batch_index][0][0 : self._discriminator_batch_size], train=True
                        )
                        sampled_amp_motion_states = self._amp_state_preprocessor(
                            sampled_motion_batches[batch_index][0][0 : self._discriminator_batch_size], train=True
                        )
                    else:
                        sampled_amp_states = self._amp_state_preprocessor(sampled_amp_states, train=True)
                        sampled_amp_replay_states = self._amp_state_preprocessor(
                            sampled_replay_batches[batch_index][0], train=True
                        )
                        sampled_amp_motion_states = self._amp_state_preprocessor(
                            sampled_motion_batches[batch_index][0], train=True
                        )

                    # WASABI needs gradients w.r.t. BOTH the rollout and reference states (the
                    # base version only tracks the motion side) -- see the gradient-penalty block
                    # below.
                    sampled_amp_states.requires_grad_(True)
                    sampled_amp_motion_states.requires_grad_(True)
                    amp_logits, _, _ = self.discriminator.act({"states": sampled_amp_states}, role="discriminator")
                    amp_replay_logits, _, _ = self.discriminator.act(
                        {"states": sampled_amp_replay_states}, role="discriminator"
                    )
                    amp_motion_logits, _, _ = self.discriminator.act(
                        {"states": sampled_amp_motion_states}, role="discriminator"
                    )

                    amp_cat_logits = torch.cat([amp_logits, amp_replay_logits], dim=0)

                    # WASABI: MSELoss discriminator loss (instinct_rl's discriminator_loss_func=
                    # "MSELoss" -- the original AMP paper's LSGAN-style loss), replacing
                    # BCEWithLogitsLoss. Rollout+replay pushed toward -1, reference pushed
                    # toward +1 (instinct_rl/algorithms/wasabi.py's compute_amp_losses, applied
                    # here to the base version's existing rollout+replay-vs-motion batching
                    # rather than instinct_rl's own actor-vs-reference-only batching).
                    discriminator_loss = 0.5 * (
                        F.mse_loss(amp_cat_logits, -torch.ones_like(amp_cat_logits))
                        + F.mse_loss(amp_motion_logits, torch.ones_like(amp_motion_logits))
                    )

                    if self._discriminator_logit_regularization_scale:
                        logit_weights = torch.flatten(list(self.discriminator.modules())[-1].weight)
                        discriminator_loss += self._discriminator_logit_regularization_scale * torch.sum(
                            torch.square(logit_weights)
                        )

                    # WASABI: gradient penalty over concat(rollout, reference) with a tolerance
                    # margin before squaring (instinct_rl/algorithms/wasabi.py's
                    # compute_discriminator_gradient), replacing the base version's real-data-
                    # (motion)-only, tolerance-free penalty. Deliberately does NOT include the
                    # replay-buffer states -- WASABI's own formula is a strict 2-way concat, no
                    # replay-buffer concept exists in the reference implementation.
                    if self._discriminator_gradient_penalty_scale:
                        combined_states = torch.cat([sampled_amp_states, sampled_amp_motion_states], dim=0)
                        combined_logits, _, _ = self.discriminator.act({"states": combined_states}, role="discriminator")
                        combined_gradient = torch.autograd.grad(
                            combined_logits,
                            combined_states,
                            grad_outputs=torch.ones_like(combined_logits),
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )[0]
                        tolerance = self.cfg.get("discriminator_gradient_tolerance", 0.0)
                        gradient_penalty = torch.clamp(combined_gradient.norm(2, dim=1) - tolerance, min=0.0).square().mean()
                        discriminator_loss += self._discriminator_gradient_penalty_scale * gradient_penalty

                    if self._discriminator_weight_decay_scale:
                        weights = [
                            torch.flatten(module.weight)
                            for module in self.discriminator.modules()
                            if isinstance(module, torch.nn.Linear)
                        ]
                        weight_decay = torch.sum(torch.square(torch.cat(weights, dim=-1)))
                        discriminator_loss += self._discriminator_weight_decay_scale * weight_decay

                    discriminator_loss *= self._discriminator_loss_scale

                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss + discriminator_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    self.value.reduce_parameters()
                    self.discriminator.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        itertools.chain(
                            self.policy.parameters(), self.value.parameters(), self.discriminator.parameters()
                        ),
                        self._grad_norm_clip,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
                cumulative_discriminator_loss += discriminator_loss.item()

            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        self.reply_buffer.add_samples(states=amp_states.view(-1, amp_states.shape[-1]))

        n = self._learning_epochs * self._mini_batches
        self.track_data("Loss / Policy loss", cumulative_policy_loss / n)
        self.track_data("Loss / Value loss", cumulative_value_loss / n)
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / n)
        self.track_data("Loss / Discriminator loss", cumulative_discriminator_loss / n)
        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())
        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
