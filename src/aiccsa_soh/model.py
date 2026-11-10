"""CNN–WNN–WLSTM–attention for lithium-ion SOH estimation (AICCSA 2024).

Morlet gate equations are transcribed from the conference paper. The standard
LSTM cell-state update is assumed where the publication is underspecified.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


def morlet(x: torch.Tensor) -> torch.Tensor:
    """Eq. (14): psi(x) = cos(1.75 x) exp(-x^2/2)."""
    return torch.cos(1.75 * x) * torch.exp(-.5 * x.square())


class WaveletDense(nn.Module):
    """Learnable wavelet translations and positive dilations (Eqs. 11–14)."""
    def __init__(self, features, units):
        super().__init__()
        self.proj = nn.Linear(features, units)
        self.shift = nn.Parameter(torch.zeros(units))
        self.log_scale = nn.Parameter(torch.zeros(units))

    def forward(self, x):
        scale = F.softplus(self.log_scale) + 1e-4
        return morlet((self.proj(x) - self.shift) / scale)


class WaveletLSTM(nn.Module):
    """Unidirectional recurrent cell with Morlet gates (Eqs. 15–19).

    A conventional LSTM cell-state update, c_t = f_t*c_{t-1}+i_t*g_t,
    is used: the published equations do not explicitly specify c_t.
    This is a documented modeling choice, not an exact-code claim.
    """
    def __init__(self, in_features, units):
        super().__init__()
        self.units = units
        self.gates = nn.Linear(in_features + units, 4 * units)

    def forward(self, x):
        b, t, _ = x.shape
        h = x.new_zeros((b, self.units))
        c = x.new_zeros((b, self.units))
        states=[]
        for k in range(t):
            f, i, o, candidate = self.gates(torch.cat((x[:, k], h), dim=-1)).chunk(4, dim=-1)
            f, i, o, candidate = (morlet(z) for z in (f, i, o, candidate))
            c = f*c + i*candidate
            h = o * morlet(c)
            states.append(h)
        return torch.stack(states, dim=1)


class TemporalAttention(nn.Module):
    """Sigmoid attention compatibility with Table I; weights normalized in time."""
    def __init__(self, in_dim, units):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, units), nn.Tanh(),
                                 nn.Linear(units, 1))
        self.last_weights = None

    def forward(self, states):
        w = torch.sigmoid(self.net(states)).squeeze(-1)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        self.last_weights = w.detach()
        return (states * w.unsqueeze(-1)).sum(dim=1)


class BatterySOH(nn.Module):
    """Paper architecture with lightweight reference baselines for comparison."""
    def __init__(self, features, variant='AM-CNN-WNN-WLSTM', filters=64,
                 wnn_units=300, recurrent_units=100, attention_units=100,
                 dropout=.3):
        super().__init__()
        if variant not in {'AM-CNN-WNN-WLSTM', 'CNN-LSTM', 'LSTM', 'CNN'}:
            raise ValueError(f'Unsupported variant {variant}')
        self.variant = variant
        self.is_cnn = variant != 'LSTM'
        self.is_recurrent = variant != 'CNN'
        if self.is_cnn:
            self.cnn=nn.Sequential(nn.Conv1d(features,filters,3,padding=1),nn.ReLU(),
                                   nn.MaxPool1d(2,2),nn.Dropout(dropout))
            input_dim=filters
        else:
            input_dim=features
        if variant == 'AM-CNN-WNN-WLSTM':
            self.wavelet=WaveletDense(input_dim,wnn_units)
            self.recurrent=WaveletLSTM(wnn_units,recurrent_units)
            self.attention=TemporalAttention(recurrent_units,attention_units)
            final=recurrent_units
        elif self.is_recurrent:
            self.recurrent=nn.LSTM(input_dim,recurrent_units,batch_first=True)
            final=recurrent_units
        else:
            final=input_dim
        self.head=nn.Linear(final,1)

    def forward(self,x):
        if x.ndim!=3 or x.shape[1]<4:
            raise ValueError('Input must be [batches, >=4 historical cycles, features]')
        h=self.cnn(x.transpose(1,2)).transpose(1,2) if self.is_cnn else x
        if self.variant == 'AM-CNN-WNN-WLSTM':
            h=self.wavelet(h)
            h=self.recurrent(h)
            h=self.attention(h)
        elif self.is_recurrent:
            h,_=self.recurrent(h)
            h=h[:, -1]
        else:
            h=h.mean(dim=1)
        return self.head(h).squeeze(-1)
