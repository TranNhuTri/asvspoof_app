from torch import Tensor, bmm, softmax, tanh, mul, randn, cat
from torch.nn import Module, Parameter
from torch.nn.init import kaiming_uniform_


class SelfAttention(Module):
    def __init__(self, hidden_size, mean_only=False):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.att_weights = Parameter(Tensor(1, hidden_size), requires_grad=True)
        self.mean_only = mean_only
        kaiming_uniform_(self.att_weights)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        weights = bmm(inputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))

        if inputs.size(0) == 1:
            attentions = softmax(tanh(weights), dim=1)
            weighted = mul(inputs, attentions.expand_as(inputs))
        else:
            attentions = softmax(tanh(weights.squeeze()), dim=1)
            weighted = mul(inputs, attentions.unsqueeze(2).expand_as(inputs))

        if self.mean_only:
            return weighted.sum(1)
        else:
            noise = 1e-5 * randn(weighted.size())

            if inputs.is_cuda:
                noise = noise.to(inputs.device)
            avg_repr, std_repr = weighted.sum(1), (weighted+noise).std(1)

            representations = cat((avg_repr, std_repr), 1)
            return representations
