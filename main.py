import torch
import torch.nn as nn
from torch.autograd import Variable
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

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 32)
        self.lin = nn.Linear(32, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), 32), requires_grad=False).float().cuda()
        c_t = Variable(torch.zeros(input.size(0), 32), requires_grad=False).float().cuda()

        for i in range(input.size(1)):
            h_t, c_t = self.lstm1(input[:, i, :], (h_t, c_t))
            A = self.lin(h_t)
            outputs += [A]
        for i in range(future):
            h_t, c_t = self.lstm1(A, (h_t, c_t))
            A = self.lin(h_t)
            outputs += [A]
        outputs = torch.stack(outputs, 1).squeeze(2)

        return outputs

np.random.seed(0)
torch.manual_seed(0)

batch_size = 16
prediction_length = 28

data = load_symbols('data/daily')
process_symbols(data)

train = data[4:]
test = data[:4]

seq = Sequence()
seq.float()
seq.cuda()

optimizer = optim.Adam(seq.parameters())

for i in range(10):
    batch = []
    for k in train[0:1]:
        batch.append(k.data[:-100, -1].reshape((-1, 1)))
    input = Variable(torch.from_numpy(np.stack(batch, axis=0)), requires_grad=True).float().cuda()

    def closure():
        optimizer.zero_grad()
        out = seq(input)
        loss = torch.nn.functional.mse_loss(input[:, 1:, 0], out[:, :-1])
        print('loss:', loss.data[0])
        loss.backward()
        return loss
    optimizer.step(closure)

    out = seq(input, future=100)

    plt.figure()
    plt.plot(range(0, k.data.shape[0]), k.data[:, -1])
    plt.plot(range(0, k.data.shape[0]), out.data.cpu().numpy()[0, :])
    plt.show()
