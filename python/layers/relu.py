import torch

from python.layers.activation import SecretActivationLayer


class SecretReLULayer(SecretActivationLayer):
    def __init__(self, sid, LayerName, is_enclave_mode=True):
        super().__init__(sid, LayerName, is_enclave_mode)
        self.ForwardFuncName = "ReLU"
        self.BackwardFuncName = "DerReLU"
        self.PlainFunc = torch.nn.ReLU
        if self.is_enclave_mode:
            self.ForwardFunc = self.relufunc
            self.BackwardFunc = self.relubackfunc
            self.StoreInEnclave = True
        else:
            self.ForwardFunc = torch.nn.ReLU
            self.StoreInEnclave = False

    def init(self, start_enclave=True):
        super().init(start_enclave)
        self.PlainFunc = self.PlainFunc()
        if not self.is_enclave_mode:
            self.ForwardFunc = self.ForwardFunc()

    def relufunc(self, namein, nameout):
        return self.relunew(namein, nameout, self.InputShape)

    def relubackfunc(self, nameout, namedout, namedin):
        return self.relubackward(nameout, namedout, namedin, self.InputShape)


