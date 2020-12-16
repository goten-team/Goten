#!/usr/bin/env python
from __future__ import print_function
import os
from itertools import product
from collections import defaultdict, namedtuple

import torch
import torch.nn.functional as F
import torch.distributed as dist

from python.layers.input import SecretInputLayer
from python.layers.output import SecretOutputLayer
from python.basic_utils import str_hash
from python.enclave_interfaces import GlobalTensor
from python.timer_utils import NamedTimerInstance
from python.common_torch import SecretConfig, mod_move_down, union_dicts, \
    get_random_uniform, calc_shape_conv2d_weight, GlobalCppExtension
from python.torch_utils import compare_expected_actual, torch_sync
from python.tensor_loader import TensorLoader
from stateless_logger import StatelessLogger

torch.backends.cudnn.deterministic = True

LearnableParamTuple = namedtuple('LearnableParam', ('dw_name', 'w_name', 'shape'))


def conv2d_op(w, x, is_div=True):
    padding = 1

    batch_size, in_chan, img_hw, _ = x.size()
    out_chan, _, fil_hw, __ = w.size()
    y_shape = [batch_size, out_chan, img_hw, img_hw]
    dtype = x.dtype
    device = x.device
    is_cpu = True if device == torch.device("cpu") else False

    def base_conv2d(sub_x, sub_w):
        return F.conv2d(sub_x, sub_w, padding=padding)

    if is_cpu or (is_div is False):
        return base_conv2d(x, w)

    def sum_of_div(best_shape):
        best_batch_size, best_in_chan, best_out_chan = best_shape
        y = torch.zeros(y_shape, device=device, dtype=dtype)
        for idx_batch_size, idx_in_chan, idx_out_chan in product(range(batch_size // best_batch_size),
                                                                 range(in_chan // best_in_chan),
                                                                 range(out_chan // best_out_chan)):
            start_batch_size, end_batch_size = idx_batch_size * best_batch_size, (idx_batch_size + 1) * best_batch_size
            start_in_chan, end_in_chan = idx_in_chan * best_in_chan, (idx_in_chan + 1) * best_in_chan
            start_out_chan, end_out_chan = idx_out_chan * best_out_chan, (idx_out_chan + 1) * best_out_chan

            y[start_batch_size:end_batch_size, start_out_chan:end_out_chan, :, :] += \
                base_conv2d(x[start_batch_size:end_batch_size, start_in_chan:end_in_chan, :, :],
                            w[start_out_chan:end_out_chan, start_in_chan:end_in_chan, :, :])
        return y

    shapes_v100 = {
        (1024, 512, 512, 2): (1024, 512, 128),
        (1024, 512, 512, 4): (1024, 512, 128),
        (1024, 256, 512, 4): (1024, 128, 128),
        (1024, 256, 256, 8): (1024, 64, 128),
        (1024, 128, 256, 8): (1024, 64, 128),
        (512, 512, 512, 2): (512, 512, 128),
        (512, 512, 512, 4): (256, 256, 128),
        (512, 256, 512, 4): (256, 256, 128),
        (512, 256, 256, 8): (512, 128, 128),
        (512, 128, 256, 8): (512, 128, 128),
    }

    tunnable_shape = (batch_size, in_chan, out_chan, img_hw)
    if is_div and tunnable_shape in shapes_v100:
        return sum_of_div(shapes_v100[tunnable_shape])
    else:
        return base_conv2d(x, w)


def conv2d_input_grad_op(w, dy):
    return F.conv_transpose2d(dy, w, padding=1)


def conv2d_weight_grad_op(dy, x, is_div=True):
    batch_size, in_chan, img_hw, _ = x.size()
    _, out_chan, __, ___ = dy.size()
    w_shape = calc_shape_conv2d_weight(dy, x)
    dtype = x.dtype
    device = x.device
    is_cpu = True if device == torch.device("cpu") else False

    if is_cpu:
        return torch.transpose(F.conv2d(torch.transpose(x, 0, 1), torch.transpose(dy, 0, 1), padding=1), 0,
                               1).contiguous()

    def base_conv2d_weight_grad_op(sub_dy, sub_x):
        sub_w_shape = calc_shape_conv2d_weight(sub_dy, sub_x)
        return GlobalCppExtension.get_conv2d_cudnn().backward(sub_w_shape, sub_dy, sub_x, (1, 1), (1, 1), (1, 1), 1, 0, 0)

    if is_div is False:
        return base_conv2d_weight_grad_op(dy, x)

    def sum_of_div(best_shape):
        # print("running conv2d weight div")
        best_batch_size, best_in_chan, best_out_chan = best_shape
        dw = torch.zeros(w_shape, device=device, dtype=dtype)
        for idx_batch_size, idx_in_chan, idx_out_chan in product(range(batch_size // best_batch_size),
                                                                 range(in_chan // best_in_chan),
                                                                 range(out_chan // best_out_chan)):
            start_batch_size, end_batch_size = idx_batch_size * best_batch_size, (idx_batch_size + 1) * best_batch_size
            start_in_chan, end_in_chan = idx_in_chan * best_in_chan, (idx_in_chan + 1) * best_in_chan
            start_out_chan, end_out_chan = idx_out_chan * best_out_chan, (idx_out_chan + 1) * best_out_chan

            dw[start_out_chan:end_out_chan, start_in_chan:end_in_chan, :, :] += \
                base_conv2d_weight_grad_op(dy[start_batch_size:end_batch_size, start_out_chan:end_out_chan, :, :],
                                           x[start_batch_size:end_batch_size, start_in_chan:end_in_chan, :, :])
        return dw

    shapes_v100 = {
        (1024, 512, 512, 2): (1024, 512, 128),
        (1024, 512, 512, 4): (1024, 512, 128),
        (1024, 256, 512, 4): (1024, 128, 128),
        (1024, 128, 256, 8): (1024, 128, 128),
        (512, 512, 512, 2): (512, 512, 128),
        (512, 512, 512, 4): (512, 512, 128),
        (512, 256, 512, 4): (512, 128, 128),
        (512, 128, 256, 8): (128, 128, 256),
    }

    tunnable_shape = (batch_size, in_chan, out_chan, img_hw)
    if is_div and tunnable_shape in shapes_v100:
        return sum_of_div(shapes_v100[tunnable_shape])
    else:
        return base_conv2d_weight_grad_op(dy, x)


def matmul_op(w, x):
    return torch.mm(x, w.t())


def matmul_input_grad_op(w, dy):
    return torch.mm(dy, w)


def matmul_weight_grad_op(dy, x):
    return torch.mm(dy.t(), x)


def set_tensor_name_maybe_quantized(name, quantized):
    return name + ("Q" if quantized else "")


# target_op = conv2d_op
# idealC = ModOnCpu(target_op(AQ.type(torch.double), BQ.type(torch.double))).type(SecretConfig.dtypeForCpuOp)
# Forward
# A: Weight
# B: Input

# A: Weight
# B: dy
InputGradRemap = {
    "Af": "Af", "AQ": "AQ", "A0": "A0", "A1": "A1",
    "Bf": "DerCf", "BQ": "DerCQ", "B0": "DerC0", "B1": "DerC1",
    "E": "EForDerB", "F": "FForDerB",
    "C0": "C0ForDerB", "C1": "C1ForDerB", "CQ": "CQForDerB", "Cf": "CfForDerB", "Z": "ZForDerB",
}

# A: dy
# B: InputQ
WeightGradRemap = {
    "Af": "DerCf", "AQ": "DerCQ", "A0": "DerC0", "A1": "DerC1",
    "Bf": "Bf", "BQ": "BQ", "B0": "B0", "B1": "B1",
    "E": "EForDerA", "F": "FForDerA",
    "C0": "C0ForDerA", "C1": "C1ForDerA", "CQ": "CQForDerA", "Cf": "CfForDerA", "Z": "ZForDerA",
}


class SecretOpBase(TensorLoader):
    def __init__(self, name):
        super().__init__()
        self.sid = -1
        self.name = name
        self.AllDistWait = {}
        self.TensorDistOppo = {}
        self.tensor_name_list = None
        self.encryption_tensor_name_list = None

        self.a_shape = None
        self.b_shape = None
        self.c_shape = None

    def target_op(self, a, b):
        return NotImplemented

    def name_modifier(self, name):
        return self.name + name

    def set_shapes(self, a_shape, b_shape, c_shape):
        self.a_shape = a_shape
        self.b_shape = b_shape
        self.c_shape = c_shape

    def link_tensors(self):
        if self.sid != 2:
            GlobalTensor.link_tags(self.get_tag("C", remap=False), self.get_tag("CQ", remap=False))

    def generate_tensor_name_list(self, force=False):
        super().generate_tensor_name_list(force=force)

        all_tensor_name_list = dict()
        all_tensor_name_list["S0 or S1"] = \
            [("Af", self.a_shape, ["U", "A0"]), ("AQ", self.a_shape, ["U", "A0"]), ("E", self.a_shape, ["U"]),
             ("Bf", self.b_shape, ["V", "B0"]), ("BQ", self.b_shape, ["V", "B0"]), ("F", self.b_shape, ["V"]),
             ("C0", self.c_shape, None), ("C1", self.c_shape, None),
             ("CQ", self.c_shape, ["CM0", "CM1", "CMZ"]), ("Cf", self.c_shape, None), ("Z", self.c_shape, None), ]
        # Name, shape, SeedName

        all_tensor_name_list[0] = all_tensor_name_list["S0 or S1"] + \
                                  [("A0", self.a_shape, ["A0"]), ("B0", self.b_shape, ["B0"]),
                                   ("CM0", self.c_shape, None)]
        all_tensor_name_list[1] = all_tensor_name_list["S0 or S1"] + \
                                  [("A1", self.a_shape, ["A0"]), ("B1", self.b_shape, ["B0"]),
                                   ("CM1", self.c_shape, None)]
        all_tensor_name_list[2] = [("U", self.a_shape, ["U"]), ("V", self.b_shape, ["V"]), ("Z", self.c_shape, None),
                                   ("CMZ", self.c_shape, ["CMZ"])]

        if self.sid == -1:
            self.tensor_name_list = union_dicts([all_tensor_name_list[0], all_tensor_name_list[1],
                                                 all_tensor_name_list[2]])
        else:
            self.tensor_name_list = all_tensor_name_list[self.sid]

        self.encryption_tensor_name_list = [("C0", self.c_shape), ("C1", self.c_shape), ("Z", self.c_shape)]
        # Name, Original, Mask
        self.ShareTuple = {"E": ("AQ", "U"), "F": ("BQ", "V"), "A1": ("AQ", "A0"), "B1": ("BQ", "B0")}
        self.ShareVarName = list(map(lambda x: x[0], self.ShareTuple.items()))
        self.RandomVarName = list(map(lambda x: x[1][1], self.ShareTuple.items()))

    def get_output_shape(self):
        return self.c_shape

    def get_dist_tag(self, name, dst):
        return str_hash(str(self.get_tag(name)) + "to" + str(dst)) % ((1 << 31) - 1)

    def set_eid(self, eid):
        self.eid = eid

    def send_cpu(self, name, dst):
        opponent = dst
        self.TensorDistOppo[name] = opponent
        tag = self.get_dist_tag(name, opponent)
        self.AllDistWait[tag] = dist.isend(tensor=self.get_cpu(name), dst=dst, tag=tag)

    def recv_cpu(self, name, src):
        if self.sid == -1:
            raise Exception("sid should be 0, 1, or 2")
        opponent = self.sid
        self.TensorDistOppo[name] = opponent
        tag = self.get_dist_tag(name, opponent)
        self.AllDistWait[tag] = dist.irecv(tensor=self.get_cpu(name), src=src, tag=tag)

    def send_encrypted(self, name, dst):
        opponent = dst
        self.TensorDistOppo[name] = opponent
        tag = self.get_dist_tag(name, opponent)
        self.AllDistWait[tag] = dist.isend(tensor=self.get_encryption(name), dst=dst, tag=tag)

    def recv_encrypted(self, name, src):
        if self.sid == -1:
            raise Exception("sid should be 0, 1, or 2")
        opponent = self.sid
        self.TensorDistOppo[name] = opponent
        tag = self.get_dist_tag(name, opponent)
        self.AllDistWait[tag] = dist.irecv(tensor=self.get_encryption(name), src=src, tag=tag)

    def dist_wait(self, name, dst=-1):
        opponent = self.TensorDistOppo[name] if dst == -1 else dst
        tag = self.get_dist_tag(name, opponent)
        if tag not in self.AllDistWait:
            raise Exception("Tensor %s and %d has not been iSend/Recv" % (name, dst))
        self.AllDistWait[tag].wait()

    # Asynchronously
    def async_random_loading(self, NameList):
        task_ids = []
        for name in NameList:
            task_ids.append((self.generate_enclave_tensor(name), name))
        while len(task_ids) > 0:
            to_be_removed = []
            for task_id, name in task_ids:
                status = self.get_task_status(task_id)
                if status:
                    to_be_removed.append((task_id, name))
            for task_id, name in to_be_removed:
                task_ids.remove((task_id, name))
                self.transfer_cpu_to_gpu(name)


# S0
class SecretBaseS0(SecretOpBase):
    def __init__(self, name):
        super().__init__(name)
        self.sid = 0

    def target_op(self, a, b):
        raise NotImplemented

    def compute(self, need_quantize=True):
        logger = StatelessLogger("0")
        with NamedTimerInstance("S0: secret_sharing_compute"):
            with NamedTimerInstance("S0: PrepareComm"):
                self.recv_cpu("C1", 1)
                self.recv_cpu("Z", 2)

            RandomTensorList = ["A0", "B0", "E", "F"]

            torch_sync()
            logger.info("StartLinearLayer")
            logger.info("GoingToPrepareMasked")
            with NamedTimerInstance("S0: GetRandom"):
                gpu_names = []
                task_ids = []
                task_ids.append((self.async_get_random("A0", self.get_cpu("A0")), ["A0"]))
                task_ids.append((self.async_get_random("B0", self.get_cpu("B0")), ["B0"]))
                if need_quantize:
                    task_ids.append((self.fused_quantize_share("Af", "E", "Af", "U", is_async=True), ["E"]))
                    task_ids.append((self.fused_quantize_share("Bf", "F", "Bf", "V", is_async=True), ["F"]))

                else:
                    task_ids.append((self.async_get_share("AQ", self.get_cpu("E"), "U"), ["E"]))
                    task_ids.append((self.async_get_share("BQ", self.get_cpu("F"), "V"), ["F"]))

                while len(task_ids) > 0:
                    to_be_removed = []
                    for task_id, name in task_ids:
                        status = self.get_task_status(task_id)
                        if status:
                            to_be_removed.append((task_id, name))
                    for task_id, names in to_be_removed:
                        task_ids.remove((task_id, names))
                        gpu_names += names
                masking_task_id = self.async_masking_c01("CQ", "CM0", "CM1", "CMZ", self.get_cpu("CM0"))

            # Set Seed in C0 enclave
            # Create A CM0 cpu tensor

            torch_sync()
            logger.info("GoingToGPU")
            if SecretConfig.is_comptue_gpu:
                with NamedTimerInstance("S0: Gpu") as timer:
                    for name in gpu_names:
                        self.transfer_cpu_to_gpu(name)
                    self.set_gpu("C0", self.target_op(self.get_gpu("A0"), self.get_gpu("F"))
                                 + self.target_op(self.get_gpu("E"), self.get_gpu("B0"))
                                 - self.target_op(self.get_gpu("E"), self.get_gpu("F")))

                    while not self.get_task_status(masking_task_id):
                        pass
                    self.transfer_cpu_to_gpu("CM0")
                    self.set_gpu("C0", self.get_gpu("C0") - self.get_gpu("CM0"))

                    timer.end("target_op before sync")
                    # torch.cuda.synchronize()
                    timer.end("target_op")

                    self.set_gpu("C0", mod_move_down(self.get_gpu("C0")))
                    timer.end("mod_move_down")

                    self.transfer_gpu_to_cpu("C0")
                    timer.end("transfer_gpu_to_cpu")

            torch_sync()
            logger.info("GoingToReceive")
            with NamedTimerInstance("S0: Cpu Post-Preprocess") as timer:
                self.send_cpu("C0", 1)
                self.dist_wait("C1")
                timer.end("EnclaveAdd - C1 Receive")

            torch_sync()
            logger.info("GoingToPostProcess")
            with NamedTimerInstance("S0: Receive") as timer:
                self.enclave_add_from_cpu("C0", "CQ")
                self.dist_wait("Z")
                timer.end("EnclaveAdd - Z Receive")
                self.enclave_add_from_cpu("Z", "CQ")
                timer.end("EnclaveAdd - Z Compute")

            torch_sync()
            logger.info("GoingToReconstruct")
            with NamedTimerInstance("S0: Recon") as timer:
                if need_quantize:
                    self.fused_recon("Cf", "CQ", "C1", "Af", "Bf")
                else:
                    self.enclave_add_from_cpu("C1", "CQ")

            torch_sync()
            logger.info("Finished")


# S1
class SecretBaseS1(SecretOpBase):
    def __init__(self, name):
        super().__init__(name)
        self.sid = 1

    def compute(self, need_quantize=True):
        with NamedTimerInstance("S1: secret_sharing_compute"):
            with NamedTimerInstance("S1: PrepareComm"):
                self.recv_cpu("C0", 0)
                self.recv_cpu("Z", 2)

            torch_sync() # GoingToPrepareMasked
            with NamedTimerInstance("S1: GetRandom"):
                gpu_names = []
                task_ids = []
                if need_quantize:
                    task_ids.append((self.fused_quantize_share2("Af", "A1", "E", "Af", "A0", "U"), ["A1", "E"]))
                    task_ids.append((self.fused_quantize_share2("Bf", "B1", "F", "Bf", "B0", "V"), ["B1", "F"]))
                else:
                    task_ids.append((self.async_get_share("AQ", self.get_cpu("E"), "U"), ["E"]))
                    task_ids.append((self.async_get_share("AQ", self.get_cpu("A1"), "A0"), ["A1"]))
                    task_ids.append((self.async_get_share("BQ", self.get_cpu("F"), "V"), ["F"]))
                    task_ids.append((self.async_get_share("BQ", self.get_cpu("B1"), "B0"), ["B1"]))

                while len(task_ids) > 0:
                    to_be_removed = []
                    for task_id, name in task_ids:
                        status = self.get_task_status(task_id)
                        if status:
                            to_be_removed.append((task_id, name))
                    for task_id, names in to_be_removed:
                        task_ids.remove((task_id, names))
                        gpu_names += names
                masking_task_id = self.async_masking_c01("CQ", "CM1", "CM0", "CMZ", self.get_cpu("CM1"))

            torch_sync() # GoingToGPU
            if SecretConfig.is_comptue_gpu:
                with NamedTimerInstance("S1: Gpu"):
                    for name in gpu_names:
                        self.transfer_cpu_to_gpu(name)
                    self.set_gpu("C1", self.target_op(self.get_gpu("A1"), self.get_gpu("F"))
                                 + self.target_op(self.get_gpu("E"), self.get_gpu("B1")))

                    while not self.get_task_status(masking_task_id):
                        pass
                    self.transfer_cpu_to_gpu("CM1")
                    self.set_gpu("C1", self.get_gpu("C1") - self.get_gpu("CM1"))

                    self.set_gpu("C1", mod_move_down(self.get_gpu("C1")))
                    self.transfer_gpu_to_cpu("C1")

            torch_sync() # GoingToPostProcess
            with NamedTimerInstance("S1: Cpu Post-Preprocess") as timer:
                self.send_cpu("C1", 0)
                self.dist_wait("C0")
                print(self.get_cpu("C1").shape)
                timer.end("EnclaveAdd - C0 Receive")

            torch_sync() # GoingToReconstruct
            with NamedTimerInstance("S1: Receive") as timer:
                self.enclave_add_from_cpu("C1", "CQ")
                self.dist_wait("Z")
                timer.end("EnclaveAdd - Z Receive")
                self.enclave_add_from_cpu("Z", "CQ")
                timer.end("EnclaveAdd - Z Compute")

            torch_sync()
            with NamedTimerInstance("S1: Reconstruct") as timer:
                if need_quantize:
                    self.fused_recon("Cf", "CQ", "C0", "Af", "Bf")
                else:
                    self.enclave_add_from_cpu("C0", "CQ")
                timer.end("EnclaveAdd - C0 Compute")
            torch_sync()


# S2
class SecretBaseS2(SecretOpBase):
    def __init__(self, name):
        super().__init__(name)
        self.sid = 2

    def compute(self, need_quantize=True):
        # return

        torch_sync() # GoingToPrepareMasked
        with NamedTimerInstance("S2: secret_sharing_compute"):
            gpu_names = []
            RandomTensorList = ["U", "V"]
            task_ids = []
            for name in RandomTensorList:
                task_ids.append((self.generate_enclave_tensor(name), name))
            while len(task_ids) > 0:
                to_be_removed = []
                for task_id, name in task_ids:
                    status = self.get_task_status(task_id)
                    if status:
                        to_be_removed.append((task_id, name))
                for task_id, name in to_be_removed:
                    task_ids.remove((task_id, name))
                    gpu_names.append(name)
            # for name in RandomTensorList:
            #     self.CpuTensors[name].pin_memory()
            # with NamedTimerInstance("S2: GetRandom"):
            #     self.async_random_loading(RandomTensorList)
                # self.get_random("CMZ", self.get_cpu("CMZ"))
            masking_task_id = self.async_get_random("CMZ", self.get_cpu("CMZ"), "CMZ")

            torch_sync() # GoingToGPU
            if SecretConfig.is_comptue_gpu:
                with NamedTimerInstance("S2: Gpu"):
                    for name in gpu_names:
                        self.transfer_cpu_to_gpu(name)
                    self.set_gpu("Z", self.target_op(self.get_gpu("U"), self.get_gpu("V")))

                    while not self.get_task_status(masking_task_id):
                        pass
                    self.transfer_cpu_to_gpu("CMZ")
                    self.set_gpu("Z", self.get_gpu("Z") - self.get_gpu("CMZ"))

                    self.set_gpu("Z", mod_move_down(self.get_gpu("Z")))
                    self.transfer_gpu_to_cpu("Z")

            with NamedTimerInstance("S2: Result Communicate Send") as timer:
                self.send_cpu("Z", 0)
                self.send_cpu("Z", 1)
                timer.end("EnclaveAdd - Z Send")

            torch_sync() # GoingToPostProcess
            torch_sync() # GoingToReceive
            torch_sync() # GoingToReconstruct
            torch_sync() # Finished
            with NamedTimerInstance("S2: Result Communicate Wait") as timer:
                self.dist_wait("Z", 0)
                self.dist_wait("Z", 1)

            # print("S2: Secret OP U: ", self.get_cpu("U")[0, 0,])

    def secret_sharing_compute(self):
        print("Waiting for start")
        dist.barrier()

        self.compute()


def secret_op_class_factory(sid, target_op_name):
    all_target_op = {"Matmul": matmul_op, "MatmulInputGrad": matmul_input_grad_op,
                     "MatmulWeightGrad": matmul_weight_grad_op,
                     "Conv2d": conv2d_op, "Conv2dInputGrad": conv2d_input_grad_op,
                     "Conv2dWeightGrad": conv2d_weight_grad_op}
    all_sid_class = {0: SecretBaseS0, 1: SecretBaseS1, 2: SecretBaseS2}

    target_op_func = all_target_op[target_op_name]
    sid_class = all_sid_class[sid]
    class_name = "Secret%sS%d" % (target_op_name, sid)

    def __init__(self, name):
        sid_class.__init__(self, name)

    # noinspection PyUnusedLocal
    def target_op(self, a, b):
        return target_op_func(a, b)

    new_class = type(class_name, (sid_class,), {"__init__": __init__, "target_op": target_op})
    return new_class



class SecretNeuralNetwork(TensorLoader):
    nn_name = None
    layers = None

    def __init__(self, sid, nn_name):
        super().__init__()
        self.sid = sid
        self.init(start_enclave=False)
        self.nn_name = nn_name

    def set_layers(self, layers):
        self.layers = layers

        if not isinstance(self.layers[0], SecretInputLayer):
            raise ValueError("The first layer has to be input layer")
        if not isinstance(self.layers[-1], SecretOutputLayer):
            raise ValueError("The last layer has to be output layer")

        for i in range(len(self.layers) - 1):
            PrevLayer = self.layers[i]
            NextLayer = self.layers[i + 1]
            PrevLayer.register_next_layer(NextLayer)
            NextLayer.register_prev_layer(PrevLayer)

        for layer in self.layers:
            print(f"Before initialize layer {layer.LayerName}")
            layer.set_eid(self.get_eid())
            layer.init_shape()
            layer.link_tensors()

        for layer in self.layers:
            layer.init(start_enclave=False)

    def execute_for_each_layer(self, func, reverse=False):
        layers = self.layers[::-1] if reverse else self.layers
        for layer in layers:
            # print(f"SID: {self.sid} {layer.LayerName}, {func}")
            if self.sid == 2 and layer.IsDummyForS2:
                continue
            func(layer)

    def classifier_output(self):
        with NamedTimerInstance(f"S{self.sid}: {self.nn_name} classifier_output"):
            self.forward()
            if self.sid == 2:
                return
            # layers: input_layer, ..., fc_layer, output_layer
            last_fc = self.layers[-2]
            last_fc.transfer_enclave_to_cpu("output")
            outputs = last_fc.get_cpu("output")
            _, predicted = torch.max(outputs.data, 1)
            return predicted

    def get_loss(self):
        return self.layers[-1].get_loss()

    def forward(self):
        def run_forward(layer):
            layer.forward()
        with NamedTimerInstance(f"S{self.sid}: {self.nn_name} Forward"):
            self.execute_for_each_layer(run_forward)

    def backward(self):
        def run_backward(layer):
            layer.backward()
        with NamedTimerInstance(f"S{self.sid}: {self.nn_name} Backward"):
            self.execute_for_each_layer(run_backward, reverse=True)

    def plain_forward(self):
        with NamedTimerInstance(f"S{self.sid}: {self.nn_name} PlainForward"):
            self.execute_for_each_layer(lambda x: x.plain_forward())

    def plain_backward(self):
        with NamedTimerInstance(f"S{self.sid}: {self.nn_name} PlainBackward"):
            self.execute_for_each_layer(lambda x: x.plain_backward(), reverse=True)

    def show_plain_error(self):
        self.execute_for_each_layer(lambda x: x.show_plain_error())


# Take the registered learnable parameters list in layers and update them
# It may need take extra storage
# And execution depends on where the tensors are stored
# https://pytorch.org/docs/stable/optim.html#torch.optim.SGD
class SgdOptimizer(TensorLoader):
    def __init__(self, sid):
        super().__init__()
        self.sid = sid
        self.learning_rate = 0.05
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.momentum_init_flags = defaultdict(lambda: False)
        self.ideal_momentum_buf = {}

        self.lr_gamma = 0.5
        self.lr_step = 30
        self.step_counter = 0

        self.layers = None

    def set_layers(self, layers):
        self.layers = layers

    def generate_tensor_name_list(self, force=False):
        # Run if forced or self.tensor_name_list is not generated
        if not force and self.tensor_name_list:
            return
        if self.sid == 2:
            return

        self.tensor_name_list = []
        for layer in self.layers:
            for (DerName, ParamName, shape) in layer.LearnableParamsList:
                self.tensor_name_list.append((ParamName + "Momentum", shape, None))

    def update_params(self, test_with_ideal=False):
        if self.sid == 2:
            return
        for layer in self.layers:
            self.update_params_in_layer(layer, test_with_ideal=test_with_ideal)

    def update_params_in_layer(self, layer, test_with_ideal=False):
        # ref: https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
        if layer.LearnableParamsList is None:
            return

        task_ids = []
        for (der_name, param_name, shape) in layer.LearnableParamsList:
            momentum_name = param_name + "Momentum"
            global_momentum_name = layer.name_modifier(momentum_name)

            if layer.StoreInEnclave:
                if test_with_ideal:
                    ideal_p, ideal_momentum = self.ideal_update_params_with_name(layer, der_name, param_name, shape)
                first_momentum = not self.momentum_init_flags[global_momentum_name]
                if first_momentum:
                    # print("FIRST MOMENTUM")
                    self.momentum_init_flags[global_momentum_name] = True
                    layer.init_enclave_tensor(momentum_name, shape)
                task_id = layer.sgd_update(param_name=param_name, grad_name=der_name, momentum_name=momentum_name,
                                           lr=self.learning_rate, momentum=self.momentum,
                                           weight_decay=self.weight_decay,
                                           first_momentum=first_momentum, is_async=True)
                if test_with_ideal:
                    while not self.get_task_status(task_id):
                        pass
                    layer.generate_cpu_tensor(momentum_name, shape)
                    layer.transfer_enclave_to_cpu(momentum_name)
                    layer.transfer_enclave_to_cpu(param_name)
                    param_err = compare_expected_actual(ideal_p, layer.get_cpu(param_name), get_relative=True)
                    print(f"S{self.sid}: {layer.LayerName} Param Error: {param_err}")
                    momentum_err = compare_expected_actual(ideal_momentum, layer.get_cpu(momentum_name), get_relative=True)
                    print(f"S{self.sid}: {layer.LayerName} Momentum Error: {momentum_err}")
                else:
                    task_ids.append(task_id)
            else:
                DerCpu = layer.get_cpu(der_name)
                ParamsCpu = layer.get_cpu(param_name)

                if test_with_ideal:
                    ideal_p, ideal_momentum = self.ideal_update_params_with_name(layer, der_name, param_name, shape)

                DerCpu.add_(self.weight_decay, ParamsCpu)

                if not self.momentum_init_flags[global_momentum_name]:
                    self.momentum_init_flags[global_momentum_name] = True
                    layer.generate_cpu_tensor(momentum_name, shape)
                    layer.get_cpu(momentum_name).copy_(DerCpu)
                    MomentumCpu = layer.get_cpu(momentum_name)
                else:
                    MomentumCpu = layer.get_cpu(momentum_name)
                    MomentumCpu.mul_(self.momentum).add_(1, DerCpu)

                ParamsCpu.add_(-self.learning_rate, MomentumCpu)

                if test_with_ideal:
                    param_err = compare_expected_actual(ideal_p, layer.get_cpu(param_name), get_relative=True)
                    print(f"S{self.sid}: {layer.LayerName} Param Error: {param_err}")
                    momentum_err = compare_expected_actual(ideal_momentum, layer.get_cpu(momentum_name), get_relative=True)
                    print(f"S{self.sid}: {layer.LayerName} Momentum Error: {momentum_err}")

        # Wait for all tasks to be finished
        for task_id in task_ids:
            while not self.get_task_status(task_id):
                pass

    def ideal_update_params_with_name(self, layer, der_name, param_name, shape):
        weight_decay = self.weight_decay
        momentum = self.momentum
        dampening = 0
        nesterov = False
        lr = self.learning_rate

        global_momentum_name = layer.name_modifier(param_name + 'Momentum')

        if layer.StoreInEnclave:
            layer.transfer_enclave_to_cpu(der_name)
            layer.transfer_enclave_to_cpu(param_name)
        d_p = torch.clone(layer.get_cpu(der_name)).detach()
        p = torch.clone(layer.get_cpu(param_name)).detach()

        if weight_decay != 0:
            d_p.add_(weight_decay, p)
        if global_momentum_name not in self.ideal_momentum_buf:
            buf = self.ideal_momentum_buf[global_momentum_name] = torch.clone(d_p).detach()
        else:
            buf = self.ideal_momentum_buf[global_momentum_name]
            buf.mul_(momentum).add_(1 - dampening, d_p)
        if nesterov:
            d_p = d_p.add(momentum, buf)
        else:
            d_p = buf
        p.add_(-lr, d_p)

        return p, buf


def warming_up_cuda():
    device = torch.device("cuda:0")
    # device = torch.device("cpu")

    print("Execution device: ", device)
    print("PyTorch version: ", torch.__version__)
    print("CUDA version: ", torch.version.cuda)
    print("CUDA device:", torch.cuda.get_device_name(0))

    batch_size, n_input_channel, n_output_channel, img_hw, filter_hw = 512, 512, 256, 4, 3
    x_shape = [batch_size, n_input_channel, img_hw, img_hw]
    w_shape = [n_output_channel, n_input_channel, filter_hw, filter_hw]
    with NamedTimerInstance("Warming up Cuda double"):
        dummy_a = get_random_uniform(SecretConfig.PrimeLimit, x_shape).type(SecretConfig.dtypeForSave)
        dummy_b = get_random_uniform(SecretConfig.PrimeLimit, w_shape).type(SecretConfig.dtypeForSave)
        F.conv2d(dummy_a.cuda().type(SecretConfig.dtypeForCudaMm), dummy_b.cuda().type(SecretConfig.dtypeForCudaMm),
                 padding=1)

    with NamedTimerInstance("Warming up Cuda dobule 2nd"):
        F.conv2d(dummy_a.cuda().type(torch.double), dummy_b.cuda().type(torch.double),
                 padding=1)

    with NamedTimerInstance("Warming up Cuda float"):
        F.conv2d(dummy_a.cuda().type(torch.float), dummy_b.cuda().type(torch.float), padding=1)

    with NamedTimerInstance("Warming up Cuda float 2nd"):
        F.conv2d(dummy_a.cuda().type(torch.float), dummy_b.cuda().type(torch.float), padding=1)

    batch_size, n_input_channel, n_output_channel, img_hw, filter_hw = 64, 64, 64, 8, 3
    x_shape = [batch_size, n_input_channel, img_hw, img_hw]
    w_shape = [n_output_channel, n_input_channel, filter_hw, filter_hw]
    with NamedTimerInstance("Warming up Cpu"):
        dummy_a = get_random_uniform(SecretConfig.PrimeLimit, x_shape).type(SecretConfig.dtypeForSave)
        dummy_b = get_random_uniform(SecretConfig.PrimeLimit, w_shape).type(SecretConfig.dtypeForSave)
        F.conv2d(dummy_a.type(SecretConfig.dtypeForCpuOp), dummy_b.type(SecretConfig.dtypeForCpuOp),
                 padding=1)

    with NamedTimerInstance("Warming up CppExtension"):
        GlobalCppExtension.get_conv2d_cudnn()


def init_communicate(rank, master_address, master_port, backend='gloo'):
    os.environ['MASTER_ADDR'] = master_address
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend, rank=rank, world_size=SecretConfig.worldSize)
