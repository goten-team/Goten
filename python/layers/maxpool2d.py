import torch

from python.layers.activation import SecretActivationLayer
from python.linear_shares import TensorLoader


class SecretMaxpool2dLayer(SecretActivationLayer):
    def __init__(self, sid, LayerName, filter_hw, is_enclave_mode=True):
        super().__init__(sid, LayerName, is_enclave_mode)
        self.ForwardFuncName = "Maxpool2d"
        self.BackwardFuncName = "DerMaxpool2d"
        self.filter_hw = filter_hw
        self.startmaxpool = False
        self.PlainFunc = torch.nn.MaxPool2d
        self.maxpoolpadding = None
        self.row_stride = None
        self.col_stride = None

        if is_enclave_mode:
            self.ForwardFunc = self.maxpoolfunc
            self.BackwardFunc = self.maxpoolbackfunc
            self.StoreInEnclave = True
        else:
            self.ForwardFunc = torch.nn.MaxPool2d
            self.StoreInEnclave = False

    def init_shape(self):
        self.InputShape = self.PrevLayer.get_output_shape()
        if len(self.InputShape) != 4:
            raise ValueError("Maxpooling2d apply only to 4D Tensor")
        if self.InputShape[2] != self.InputShape[3]:
            raise ValueError("The input tensor has to be square images")
        if self.InputShape[2] % self.filter_hw != 0:
            raise ValueError("The input tensor needs padding for this filter size")
        InputHw = self.InputShape[2]
        output_hw = InputHw // self.filter_hw
        self.OutputShape = [self.InputShape[0], self.InputShape[1], output_hw, output_hw]
        self.HandleShape = self.InputShape
        self.Shapefortranspose = [int(round(((self.InputShape[0] * self.InputShape[1] * self.InputShape[2] * self.InputShape[3])/262144)+1/2)), 262144, 1, 1]

    def init(self, start_enclave=True):
        if self.is_enclave_mode:
            self.PlainFunc = self.PlainFunc(self.filter_hw)

            TensorLoader.init(self, start_enclave)
            if self.startmaxpool is False:
                self.startmaxpool = True
                return self.maxpoolinit(self.LayerName, "inputtrans", "outputtrans")
        else:
            self.ForwardFunc = self.ForwardFunc(self.filter_hw)
            self.PlainFunc = self.PlainFunc(self.filter_hw)

            TensorLoader.init(self, start_enclave)

    def maxpoolfunc(self, namein, nameout):
        # assume row_stride and col_stride are both None or both not None
        # assume row_pad and col_pad are both None or both not None
        return self.maxpoolnew(self.LayerName, namein, nameout, self.InputShape, self.OutputShape[2], self.OutputShape[3],
                               self.filter_hw, self.filter_hw, self.row_stride, self.col_stride, self.maxpoolpadding,
                               self.maxpoolpadding)

    def maxpoolbackfunc(self, nameout, namedout, namedin):
        return self.maxpoolback(self.LayerName, namedout, namedin, self.InputShape, self.OutputShape[2], self.OutputShape[3],
                                self.filter_hw, self.filter_hw, self.row_stride, self.col_stride, self.maxpoolpadding,
                                self.maxpoolpadding)


