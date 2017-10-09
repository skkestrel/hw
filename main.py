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
        self.lstm1 = nn.LSTMCell(6, 32)
        self.lin = nn.Linear(32, 1)

    def forward(self, input):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), 32), requires_grad=False).float().cuda()
        c_t = Variable(torch.zeros(input.size(0), 32), requires_grad=False).float().cuda()
        A = Variable(torch.zeros(input.size(0), 1)).cuda()

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
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

plt.plot(data[0].data)
plt.show()

train = data[4:]
test = data[:4]

seq = Sequence()
seq.float()
seq.cuda()

optimizer = optim.Adam(seq.parameters())

for i in range(10):
    np.random.shuffle(train)

    for j in range(0, len(train), batch_size):
        for i in range(10):
            batch = []
            for k in train[j:j+batch_size]:
                start = np.random.randint(0, k[1].shape[0] - prediction_length)
                batch.append(k[1][start:start+prediction_length])
            input = Variable(torch.from_numpy(np.stack(batch, axis=0)), requires_grad=False).float().cuda()

            def closure():
                optimizer.zero_grad()
                out = seq(input)
                loss = torch.nn.functional.mse_loss(input[:, 1:, 2], out[:, :-1])
                print('loss:', loss.data)
                loss.backward()
                return loss
            optimizer.step(closure)

    batch = []
    for k in test:
        batch.append(k[1][100:100+prediction_length])
    input = Variable(torch.from_numpy(np.stack(batch, axis=0)), requires_grad=False).float().cuda()

    out = seq(input)
    plt.figure()
    plt.plot(range(0, prediction_length-1), input[:, 1:, 2].data.cpu().numpy().T)
    plt.plot(range(1, prediction_length), out[:, :-1].data.cpu().numpy().T)
    plt.show()
