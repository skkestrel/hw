import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pdb

class Symbol():
    def __init__(self, name, data):
        self.name = name
        self.data = data

def load_symbols(folder):
    import glob
    import os.path
    results = []
    for index, i in enumerate(glob.glob(os.path.join(folder, "*.csv"))):
        if index % 10 > 0:
            continue
        print(i)
        csv = np.genfromtxt(i, usecols=(0, 2, 3, 4, 5, 6), delimiter=',').astype(np.float)
        results.append(Symbol(i, csv))
    return results
    # Time Open High Low Close Volume

def process_symbols(data):
    for i in data:
        # drop date
        table = i.data[:, 1:5]

        # calculate pivot, r1, s1
        pivot = table.mean(axis=1, keepdims=True)
        r1 = 2 * pivot - i.data[:, 3].reshape((-1, 1))
        s1 = 2 * pivot - i.data[:, 2].reshape((-1, 1))

        # add features
        table = np.concatenate([table[:, 1:6], r1, s1], axis=1)
        # normalize to percentage of pivot value
        table = table / pivot - 1.
        # normalize fluctations by std
        table = table / table.std(axis=(0, 1))

        # log initial value
        i.initial = pivot[0, 0]
        # express pivots by percentage change from previous day
        pivot[1:] = pivot[1:] / pivot[:-1] - 1
        pivot[0] = 0

        pivot = pivot / pivot.std()

        table = np.concatenate([table, pivot], axis=1)
        i.data = table

class Echo(nn.Module):
    def __init__(self, input, output, size, sparsity=0.05, feedback=0.01):
        super(Echo, self).__init__()
        self.input_features = input
        self.output_features = output
        self.size = size

        r_r = 2 * np.random.rand(size, size) - 1
        r_r *= np.random.rand(size, size) < sparsity

        radius = np.abs(np.linalg.eigvals(r_r)).max()

        self.r_o = nn.Linear(size, output).float().cuda()
        self.r_r = Variable(torch.from_numpy(r_r / radius), requires_grad=False).float().cuda()
        self.i_r = Variable(2 * torch.rand(input, size) - 1, requires_grad=False).float().cuda()
        self.o_r = Variable(feedback * (2 * torch.rand(output, size) - 1), requires_grad=False).float().cuda()


    def forward(self, input):
        state = Variable(torch.zeros(self.size), requires_grad=False).float().cuda()
        out = Variable(torch.zeros(input.size(1), self.output_features), requires_grad=False).float().cuda()

        outputs = []

        for i in range(input.size(0)):
            state = F.tanh(torch.mm(input[i, :, :], self.i_r) + torch.mv(self.r_r, state) + torch.mm(out, self.o_r)).squeeze()
            out = self.r_o(state.unsqueeze(0))
            outputs.append(out)

        return torch.stack(outputs, dim=0)

np.random.seed(0)
torch.manual_seed(0)

batch_size = 16
prediction_length = 28

data = load_symbols('data/daily')
process_symbols(data)

train = data[4:]
test = data[:4]

echo = Echo(1, 1, 300)
echo.float()
echo.cuda()

optimizer = optim.Adam(echo.r_o.parameters(), lr=0.01)

for i in range(1000):
    batch = []
    for k in train[0:1]:
        batch.append(k.data[:, -1].reshape((-1, 1)))
    input = Variable(torch.from_numpy(np.stack(batch, axis=1)), requires_grad=True).float().cuda()

    def closure():
        optimizer.zero_grad()
        out = echo(input)
        # pdb.set_trace()
        loss = torch.nn.functional.smooth_l1_loss(out[:-1, :, 0], input[1:, :, 0])
        print('loss:', loss.data[0])
        loss.backward()
        torch.nn.utils.clip_grad_norm(echo.parameters(), 5)
        return loss
    optimizer.step(closure)

    out = echo(input)

    if i % 3 == 0:
        plt.figure()
        plt.plot(range(0, k.data.shape[0]), k.data[:, -1])
        plt.plot(range(0, k.data.shape[0]), out.data.cpu().numpy()[:, 0, 0])
        plt.show()

