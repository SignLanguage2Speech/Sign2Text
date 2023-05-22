import torch
from torch import nn, Tensor
from torch.autograd import Variable


class CustomLoss(nn.Module):

    def __init__(self, ignore_index: int, label_smoothing: float = 0.0):
        super(CustomLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.criterion = nn.KLDivLoss(reduction="sum")

    def _apply_label_smoothing(self, targets, vocab_size):
        # batch*seq_len x vocab_size
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.label_smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0 - self.label_smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.ignore_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = torch.nonzero(targets.data == self.ignore_index)
        # pylint: disable=len-as-condition
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    # pylint: disable=arguments-differ
    def forward(self, log_probs, targets):
        targets = self._apply_label_smoothing(
            targets=targets.contiguous().view(-1), vocab_size=log_probs.size(-1)
        )
        # targets: distributions with batch*seq_len x vocab_size
        assert (
            log_probs.contiguous().view(-1, log_probs.size(-1)).shape
            == targets.shape
        )
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), targets
        )
        return loss