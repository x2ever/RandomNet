import math
import collections

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class RandomLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        super(RandomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_var_weight = nn.Parameter(torch.Tensor(out_features, in_features))

        self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.mu_weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.log_var_weight, a=math.sqrt(5))

    def forward(self, input):
        std = torch.exp(0.5 * self.log_var_weight)
        eps = torch.randn_like(std)
        weight = eps.mul(std).add_(self.mu_weight)
        return F.linear(input, weight)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))


class repeat(object):
    """
    repeat(object [,times]) -> create an iterator which returns the object
    for the specified number of times.  If not specified, returns the object
    endlessly.
    """
    def __getattribute__(self, *args, **kwargs): # real signature unknown
        """ Return getattr(self, name). """
        pass

    def __init__(self, p_object, times=None): # real signature unknown; restored from __doc__
        pass

    def __iter__(self, *args, **kwargs): # real signature unknown
        """ Implement iter(self). """
        pass

    def __length_hint__(self, *args, **kwargs): # real signature unknown
        """ Private method returning an estimate of len(list(it)). """
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __next__(self, *args, **kwargs): # real signature unknown
        """ Implement next(self). """
        pass

    def __reduce__(self, *args, **kwargs): # real signature unknown
        """ Return state information for pickling. """
        pass

    def __repr__(self, *args, **kwargs): # real signature unknown
        """ Return repr(self). """
        pass


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_pair = _ntuple(2)


class _ConvNd(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias'}

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 transposed,
                 output_padding,
                 groups,
                 padding_mode) -> None:
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
        if transposed:
            self.weight = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class Conv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        padding_mode='zeros'  # TODO: refine this type
    ):
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, padding_mode)

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, None, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, None, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self._conv_forward(input, self.weight, self.bias)


if __name__ == "__main__":
    import numpy as np
    import torch.optim as optim
    import torchvision.datasets as dsets
    import torchvision.transforms as transforms

    train_data = dsets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
    test_data = dsets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)
    train_data_loader = torch.utils.data.DataLoader(train_data, 4096, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data, 4096, shuffle=True)

    model = torch.nn.Sequential(
        RandomLinear(28*28, 16),
        torch.nn.LeakyReLU(),
        RandomLinear(16, 10),
        nn.Softmax(dim=-1)
    ).cuda()
    # model = torch.nn.Sequential(
    #     nn.Linear(28*28, 256),
    #     torch.nn.ReLU(),
    #     nn.Linear(256, 10),
    #     nn.Softmax(dim=-1)
    # )
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for i in range(1000):
        losses = []
        for image, label in train_data_loader:

            y = model(image.cuda().view(-1, 28*28))
            loss = loss_fn(torch.log(y + 1e-16), label.cuda().view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(i, np.mean(losses))
