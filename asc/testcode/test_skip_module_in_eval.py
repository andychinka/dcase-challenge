import torch.nn as nn
import torch


class DumpModule(nn.Module):

    def forward(self, x):
        if self.training:
            return torch.tensor([[0], [0]])
        else:
            print("skip dumpModule")
            return x


class SimpleModel(nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()

        self.linear = nn.Linear(in_features=3, out_features=1)
        self.dump = DumpModule()

    def forward(self, x):
        out = self.linear(x)
        out = self.dump(out)
        return out


if __name__ == "__main__":

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    model = DumpModule()

    #train
    out = model(x)
    print(out)

    #eval
    x2 = torch.tensor([[2, 3, 4], [5, 6, 7]])
    model.eval()
    out = model(x2)
    print(out)