from torch import nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

# shape = torch.prod(torch.tensor(x.shape[1:]))
# .item()
# return x.view(-1, shape)
