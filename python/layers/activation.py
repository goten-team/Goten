from python.layers.nonlinear import SecretNonlinearLayer
from python.tensor_loader import TensorLoader
from python.timer_utils import NamedTimerInstance, VerboseLevel


class SecretActivationLayer(SecretNonlinearLayer):
    def __init__(self, sid, LayerName, is_enclave_mode=False):
        super().__init__(sid, LayerName)
        self.is_enclave_mode = is_enclave_mode
        self.Shapefortranspose = None

    def init_shape(self):
        self.InputShape = self.PrevLayer.get_output_shape()
        self.OutputShape = self.InputShape
        self.HandleShape = self.InputShape

    def init(self, start_enclave=True):
        TensorLoader.init(self, start_enclave)

    def get_output_shape(self):
        return self.OutputShape

    def generate_tensor_name_list(self, force=False):
        if not force and self.tensor_name_list:
            return
        if self.sid == 2:
            self.tensor_name_list = {}
            return
        if len(self.InputShape) == 4:
            self.Shapefortranspose = [int(round(((self.InputShape[0] * self.InputShape[1] * self.InputShape[2] * self.InputShape[3])/262144+1/2))), 262144, 1, 1]
        else:
            self.Shapefortranspose = self.InputShape
        NeededTensorNames = [("output", self.OutputShape, None),
                             ("handle", self.HandleShape, None),
                             ("DerInput", self.InputShape, None),
                             ("input", self.InputShape, None),
                             ("DerOutput", self.OutputShape, None),
                             ("inputtrans", self.Shapefortranspose, None),
                             ("outputtrans", self.Shapefortranspose, None),
                             ]

        self.tensor_name_list = NeededTensorNames

    def forward(self):
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Forward", verbose_level=VerboseLevel.LAYER):
            self.forward_tensor_transfer()
            self.requires_grad_on_cpu("input")
            if self.is_enclave_mode:
                self.ForwardFunc("input", "output")
            else:
                self.set_cpu("output", self.ForwardFunc(self.get_cpu("input")))

    def backward(self):
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Backward", verbose_level=VerboseLevel.LAYER):
            self.backward_tensor_transfer()
            if self.is_enclave_mode:
                self.BackwardFunc("output", "DerOutput", "DerInput")
            else:
                self.set_cpu("DerInput", self.get_cpu("output").grad_fn(self.get_cpu("DerOutput")))


