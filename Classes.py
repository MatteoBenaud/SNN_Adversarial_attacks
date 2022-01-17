import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from norse.torch.functional.lif import LIFParameters
import norse.torch.functional.encode as encode
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.lif import LIFCell

batchsize = 64

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5,stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LIFConvNet(torch.nn.Module):
    def __init__(
        self,
        input_features,
        seq_length,
        v_th,
        model="super",
        only_first_spike=False,
    ):
        super(LIFConvNet, self).__init__()
        self.constant_current_encoder = IFConstantCurrentEncoder(seq_length=seq_length,v_th=v_th)
        self.only_first_spike = only_first_spike
        self.input_features = input_features
        self.rsnn = ConvvNet4(method=model,device=device)
        self.seq_length = seq_length

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.constant_current_encoder(
            x.view(-1, self.input_features) * 1
        )
        if self.only_first_spike:
            # delete all spikes except for first
            zeros = torch.zeros_like(x.cpu()).detach().numpy()
            idxs = x.cpu().nonzero().detach().numpy()
            spike_counter = np.zeros((batchsize, 32 * 32))
            for t, batch, nrn in idxs:
                if spike_counter[batch, nrn] == 0:
                    zeros[t, batch, nrn] = 1
                    spike_counter[batch, nrn] += 1
            x = torch.from_numpy(zeros).to(x.device)

        x = x.reshape(self.seq_length, batch_size, 1, 32, 32)
        voltages = self.rsnn(x)
        m, _ = torch.max(voltages, 0)
        log_p_y = torch.nn.functional.log_softmax(m, dim=1)
        return log_p_y

class IFConstantCurrentEncoder(torch.nn.Module):
    def __init__(
        self,
        seq_length,
        tau_mem_inv=1.0 / 1e-2,
        v_th=1.0,
        v_reset=0.0,
        dt: float = 0.001,
    ):
        super(IFConstantCurrentEncoder, self).__init__()
        self.seq_length = seq_length
        self.tau_mem_inv = tau_mem_inv
        self.v_th = v_th
        self.v_reset = v_reset
        self.dt = dt

    def forward(self, x):
        lif_parameters = LIFParameters(tau_mem_inv=self.tau_mem_inv, v_th=self.v_th, v_reset=self.v_reset)
        return encode.constant_current_lif_encode(x, self.seq_length, p=lif_parameters, dt=self.dt)
      
class ConvvNet4(torch.nn.Module):
    def __init__(
        self, device, num_channels=1, feature_size=32, method="super", dtype=torch.float
    ):
        super(ConvvNet4, self).__init__()
        self.features = int(((feature_size - 4) / 2 - 4) / 2)

        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5,stride=1)
        self.conv3 = torch.nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.fc1 = torch.nn.Linear(120, 84)
#         self.fc2 = torch.nn.Linear(84, 10)

        self.lif0 = LIFCell(p=LIFParameters(method=method, alpha=100.0))
        self.lif1 = LIFCell(p=LIFParameters(method=method, alpha=100.0))
        self.lif2 = LIFCell(p=LIFParameters(method=method, alpha=100.0))
        self.lif3 = LIFCell(p=LIFParameters(method=method, alpha=100.0))
        self.out = LILinearCell(84, 10)

        self.device = device
        self.dtype = dtype

    def forward(self, x):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # specify the initial states
        s0 = None
        s1 = None
        s2 = None
        s3 = None
        so = None

        voltages = torch.zeros(
            seq_length, batch_size, 10, device=self.device, dtype=self.dtype
        )

        for ts in range(seq_length):
            z = self.conv1(x[ts, :])
            z, s0 = self.lif0(z, s0)
            z = torch.nn.functional.max_pool2d(torch.nn.functional.relu(z), 2, 2)
            z = 10 * self.conv2(z)
            z, s1 = self.lif1(z, s1)
            z = torch.nn.functional.max_pool2d(torch.nn.functional.relu(z), 2, 2)
            z = 10 * self.conv3(z)
            z, s2 = self.lif2(z, s2)
            z = torch.nn.functional.relu(z)
#           z = z.view(-1, 16*5*5)
            z = torch.flatten(z, 1)
            z = self.fc1(z)
            z, s3 = self.lif3(z, s3)
            v, so = self.out(torch.nn.functional.relu(z), so)
            voltages[ts, :, :] = v
        return voltages