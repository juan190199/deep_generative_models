import torch

def likelihood(X_train, model, device):

    X_train = X_train.to(device)
    # compute log probabilities for each data point in the batch
    # The model's log_prob method returns a tensor of shape [batch_size]
    log_probs = model.log_prob(X_train)

    # compute average log-likelihood: (1 / |D|) * sum(log p(x_i))
    # negative of this average for minimization
    loss = -torch.mean(log_probs)

    return loss
