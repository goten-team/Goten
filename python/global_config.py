import torch


class SecretConfig(object):
    worldSize = 3
    PrimeLimit = (1 << 21) - 9
    dtypeForCpuMod = torch.float32
    dtypeForCudaMm = torch.float64
    dtypeForCpuOp = torch.float32
    dtypeForSave = torch.float32
