import torch
import pytest
from aiccsa_soh.model import BatterySOH, morlet, WaveletDense, WaveletLSTM


def test_morlet_equation():
    z=torch.tensor([0.,1.,-1.])
    expected=torch.cos(1.75*z)*torch.exp(-z.square()/2)
    assert torch.allclose(morlet(z),expected)
    assert morlet(torch.zeros(1)).item()==1


def test_wavelet_scaling_and_gradients():
    m=WaveletDense(7,9);x=torch.randn(2,8,7)
    y=m(x)
    assert y.shape==(2,8,9)
    y.mean().backward()
    assert m.log_scale.grad is not None
    assert torch.isfinite(m.log_scale.grad).all()


def test_recurrent_shape_and_backpropagation():
    net=WaveletLSTM(6,9)
    y=net(torch.randn(4,8,6));assert y.shape==(4,8,9)
    y.mean().backward();assert net.gates.weight.grad is not None


@pytest.mark.parametrize('variant',['AM-CNN-WNN-WLSTM','CNN-LSTM','CNN','LSTM'])
def test_all_architecture_variants(variant):
    m=BatterySOH(7,variant=variant,filters=8,wnn_units=12,recurrent_units=8,attention_units=8)
    out=m(torch.randn(3,16,7))
    assert out.shape==(3,)
    assert torch.isfinite(out).all()
    out.mean().backward()
    assert m.head.weight.grad is not None


def test_attention_normalizes_to_one():
    net=BatterySOH(7,filters=8,wnn_units=12,recurrent_units=8,attention_units=8)
    net(torch.randn(3,16,7))
    sums=net.attention.last_weights.sum(dim=1)
    assert torch.allclose(sums,torch.ones_like(sums),atol=1e-5)


def test_rejects_short_input():
    with pytest.raises(ValueError):
        BatterySOH(7)(torch.randn(2,3,7))
