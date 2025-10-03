def get_batch_loss_metrics(
    self,
    model,
    batch: Dict[str, Union[List, torch.LongTensor]],
    train_eval: Literal["train", "eval"] = "train",
):
    """Compute the CPO loss and KL divergence (trust region) for a batch."""
    metrics = {}

    # Forward pass on concatenated chosen+rejected
    forward_output = self.concatenated_forward(model, batch)
    (
        num_non_pad_tokens,
        policy_chosen_logps,
        policy_rejected_logps,
        policy_chosen_logits,
        policy_rejected_logits,
        policy_nll_loss,
    ) = forward_output[:6]

    if self.aux_loss_enabled:
        aux_loss = forward_output[6]

    device = policy_chosen_logits.device
    chosen_input_ids = batch["chosen_input_ids"]
    chosen_attention_mask = batch["chosen_attention_mask"]
    chosen_labels = batch["chosen_labels"]

    # --- Forward pass with frozen old model (π_old) ---
    with torch.no_grad():
        old_outputs = self.old_model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask,
            use_cache=False,
        )
        old_chosen_logits = old_outputs.logits

    # --- KL divergence computation (PPO-style approximation) ---
    log_probs_new = F.log_softmax(policy_chosen_logits, dim=-1)
    log_probs_old = F.log_softmax(old_chosen_logits, dim=-1)

    # Gather log-probs at ground-truth labels (as in PPO)
    labels = chosen_labels.clone()
    if labels.shape[1] != log_probs_new.shape[1]:
        labels = labels[:, :log_probs_new.shape[1]]
    labels[labels == self.label_pad_token_id] = 0

    log_probs_new_selected = torch.gather(log_probs_new, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    log_probs_old_selected = torch.gather(log_probs_old, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    # Mask pad tokens
    loss_mask = (chosen_labels != self.label_pad_token_id).float().to(device)
    log_probs_new_selected = log_probs_new_selected * loss_mask
    log_probs_old_selected = log_probs_old_selected * loss_mask

    # PPO-style KL estimate: mean over valid tokens
    kl = (log_probs_new_selected - log_probs_old_selected).mean()

    # --- CPO loss ---
    losses, chosen_rewards, rejected_rewards = self.cpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        num_non_pad_tokens,
        policy_chosen_logits,
        policy_rejected_logits,
    )

    loss = losses.mean() + self.cpo_alpha * kl

    if self.aux_loss_enabled:
        loss += getattr(model.config, "router_aux_loss_coef", 0.0) * aux_loss

    # --- Metrics ---
    reward_accuracies = (chosen_rewards > rejected_rewards).float()
    prefix = "eval_" if train_eval == "eval" else ""

    metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
    metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
    metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
    metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
    metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
    metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
    metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
    metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
    metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean().cpu()
    metrics[f"{prefix}kl_divergence"] = kl.detach().cpu()

    return loss, metrics

# In __init__ of CPOTrainer:
import copy
self.old_model = copy.deepcopy(self.model)
self.old_model.eval()
for p in self.old_model.parameters():
    p.requires_grad = False

# Method inside CPOTrainer to refresh frozen policy:
def update_old_policy(self):
    import copy
    self.old_model = copy.deepcopy(self.model)
    self.old_model.eval()
    for p in self.old_model.parameters():
        p.requires_grad = False

# In compute_loss (top):
if self.state.global_step % 100 == 0:
    self.update_old_policy()



'''log_probs_new = F.log_softmax(policy_chosen_logits, dim=-1)
        log_probs_old = F.log_softmax(old_chosen_logits, dim=-1)



        device1 = policy_chosen_logits.device
        current_chosen_logits = policy_chosen_logits
        if not hasattr(self, "previous_chosen_logits") or self.previous_chosen_logits is None:
            self.previous_chosen_logits = {}
            #kl = torch.tensor(0.0, device=policy_chosen_logits.device)


        prev_logits = self.previous_chosen_logits.get(device1, None)
        if prev_logits is None:
            kl = torch.tensor(0.0, device=device1)
        else:
            
            log_probs_new = F.log_softmax(policy_chosen_logits, dim=-1)
            log_probs_old = F.log_softmax(prev_logits, dim=-1)
            probs_new = log_probs_new.exp()
            kl_per_token = probs_new * (log_probs_new - log_probs_old)
            kl_per_token = kl_per_token.sum(-1) ####


            if "chosen_labels" in batch and hasattr(self, "label_pad_token_id"):
                mask = (batch["chosen_labels"] != self.label_pad_token_id).float().to(device)
                kl = (kl_per_token * mask).sum() / mask.sum()
            else:
                kl = kl_per_token.mean()

        self.previous_chosen_logits[device1] = policy_chosen_logits.detach()'''