from python.layers.base import SecretLayerBase


class SecretNonlinearLayer(SecretLayerBase):
    def __init__(self, sid, LayerName):
        super().__init__(sid, LayerName)

        self.InputShape = None
        self.OutputShape = None
        self.HandleShape = None
        self.NameTagRemap = {}

    def init_shape(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError
