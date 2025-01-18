"""Direct Preference Optimization papers' implementation"""
import torch
import torch.nn.functional as F


class DPO:
    """Direct Preference Optimization class"""
    def __init__(self, reference_model, policy_model):
        """
        reference_model: which is typically the original LLM before optimization.
        policy_model: which is the model we want to optimize.
        """
        super().__init__()
        self.reference_model = reference_model
        self.policy_model = policy_model


    def compute_dpo_loss(self, policy_choosen_logprob, policy_rejected_logprob, reference_choosen_logprob, reference_rejected_logprob, beta):
        """
        Compute The DPO loss for a batch of policy and reference model log probabilities.
        Args:
            policy_chosen_logprobs: Log probabilities of the policy model for the chosen responses.
            policy_rejected_logprobs: Log probabilities of the policy model for the rejected responses.
            reference_chosen_logprobs: Log probabilities of the reference model for the chosen responses.
            reference_rejected_logprobs: Log probabilities of the reference model for the rejected responses.
            beta: Temperature parameter for the DPO loss; typically something in the range of 0.1 to 0.5.
            We ignore the reference model as beta -> 0.
        Returns:
            torch.Tensor: The DPO loss for the batch of policy and reference model log probabilities."""

        # The loss in the paper as follow : log(policy_choosen_logprob / reference_choosen_logprob)
        # and this is equal to: log(policy_choosen_logprob) - log(reference_choosen_logprob)
        choosen_logprob = policy_choosen_logprob - reference_choosen_logprob
        rejected_logprob = policy_rejected_logprob - reference_rejected_logprob

        logits = choosen_logprob - rejected_logprob

        losses = -F.logsigmoid(logits * beta)

        return losses.mean()


    def compute_logprobs(self, logits, labels, selection_mask=None):
        """
        Compute log probabilities.
        Args:
            logits: Tensor of shape (batch_size, num_tokens, vocab_size)
            labels: Tensor of shape (batch_size, num_tokens)
            selection_mask: Tensor for shape (batch_size, num_tokens)

        Returns:
            mean_log_prob: Mean log probability excluding padding tokens.
        """
        # Labels are the inputs shifted by one
        labels = labels[:, 1:].clone()

        # Truncate logits to match the labels num_tokens
        logits = logits[:, :-1, :]

        log_probs = F.log_softmax(logits, dim=-1)

        # Gather the log probabilities for the actual labels
        selected_log_probs = torch.gather(
            input=log_probs,
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)

        if selection_mask is not None:
            mask = selection_mask[:, 1:].clone()

            # Apply the mask to filter out padding tokens
            selected_log_probs = selected_log_probs * mask

            # Calculate the average log probability excluding padding tokens
            # This averages over the tokens, so the shape is (batch_size, num_tokens)
            avg_log_prob = selected_log_probs.sum(-1) / mask.sum(-1)

            return avg_log_prob

        return selected_log_probs.mean(-1)


    def compute_dpo_loss_batch(self, batch, beta):
        """Compute the DPO loss on an input batch"""
        policy_choosen_logprob = self.compute_logprobs(
            logits = self.policy_model(batch["choosen"]),
            labels = batch["choosen"],
            selection_mask=batch["chosen_mask"]
        )
        policy_rejected_logprob = self.compute_logprobs(
            logits = self.policy_model(batch["rejected"]),
            labels 	= batch["rejected"],
            selection_mask=batch["rejected_mask"]
        )

        with torch.no_grad():
            reference_choosen_logprob = self.compute_logprobs(
                logits = self.reference_model(batch["choosen"]),
                labels = batch["choosen"],
                selection_mask = batch["chosen_mask"]
            )
            reference_rejected_logprob = self.compute_logprobs(
                logits = self.reference_model(batch["rejected"]),
                labels = batch["rejected"],
                selection_mask = batch["rejected_mask"]
            )
        loss = self.compute_dpo_loss(
            policy_choosen_logprob = policy_choosen_logprob,
            policy_rejected_logprob = policy_rejected_logprob,
            reference_choosen_logprob = reference_choosen_logprob,
            reference_rejected_logprob = reference_rejected_logprob,
            beta = beta
        )
        return loss


    def compute_dpo_loss_loader(self, data_loader, beta, num_batches=None):
        """Apply compute_dpo_loss_batch to a whole data loader"""
        total_loss = 0.0
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        for i, batch in enumerate(data_loader):
            if i < num_batches:
                loss = self.compute_dpo_loss_batch(batch, beta)
                total_loss += loss.item()
            else:
                break
        total_loss /= num_batches
        return total_loss


    def eval_dpo_loss_loader(self, train_loader, val_loader, beta, eval_iter):
        """Compute the DPO loss for the training and validation dataset"""
        self.policy_model.eval()
        with torch.no_grad():
            train_loss = self.compute_dpo_loss_loader(
                data_loader=train_loader,
                beta = beta,
                num_batches=eval_iter
            )
            val_loss = self.compute_dpo_loss_loader(
                data_loader=val_loader,
                beta = beta,
                num_batches=eval_iter
            )
        res = {
            "train_loss": train_loss,
            "val_loss": val_loss
        }
        self.policy_model.train()
        return res
