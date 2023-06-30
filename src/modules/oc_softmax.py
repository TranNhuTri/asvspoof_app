from torch import randn
from torch.nn import Module, Parameter, init, Softplus
from torch.nn.functional import normalize


class OCSoftmax(Module):
    def __init__(self, feat_dim=2, r_real=0.9, r_fake=0.5, alpha=20.0):
        super(OCSoftmax, self).__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.center = Parameter(randn(1, self.feat_dim))
        init.kaiming_uniform_(self.center, 0.25)
        self.soft_plus = Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        w = normalize(self.center, p=2, dim=1)
        x = normalize(x, p=2, dim=1)

        scores = x @ w.transpose(0, 1)
        output_scores = scores.clone()

        scores[labels == 0] = self.r_real - scores[labels == 0]
        scores[labels == 1] = scores[labels == 1] - self.r_fake

        loss = self.soft_plus(self.alpha * scores).mean()

        return loss, output_scores.squeeze(1)
