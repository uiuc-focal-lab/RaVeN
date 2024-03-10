import copy
import torch

from enum import Enum

from src.network_converters import parse

from src.specs.out_spec import OutSpecType
from src.common.dataset import Dataset

class InputSpecType(Enum):
    LINF = 1
    PATCH = 2
    GLOBAL = 3
    # Untargeted uap.
    UAP = 4
    # Targeted uap.
    UAP_TARGETED = 5
    # Worst-case hamming distance.
    UAP_BINARY = 6

def get_mean_std(dataset):
    if dataset == Dataset.MNIST:
        means = [0]
        stds = [1]
    elif dataset == Dataset.CIFAR10:
        # For the model that is loaded from cert def this normalization was
        # used
        stds = [0.2023, 0.1994, 0.2010]
        means = [0.4914, 0.4822, 0.4465]
        # means = [0.0, 0.0, 0.0]
        # stds = [1, 1, 1]
    elif dataset == Dataset.ACAS:
        means = [19791.091, 0.0, 0.0, 650.0, 600.0]
        stds = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]
    else:
        raise ValueError("Unsupported Dataset!")
    return torch.tensor(means).reshape(-1, 1, 1), torch.tensor(stds).reshape(-1, 1, 1)


class InputProperty(object):
    def __init__(self, input_lb, input_ub, out_constr, dataset, input=None, targeted=False, monotone=False, monotone_prop = None):
        self.input_lb = input_lb
        self.input_ub = input_ub
        self.out_constr = out_constr
        self.dataset = dataset
        if input is not None:
            self.input = input.flatten()
        else:
            self.input = None
        self.targeted = targeted
        self.monotone = monotone
        self.monotone_prop = monotone_prop

    def update_input(self, eps):
        self.input += eps 
        self.input_lb += eps
        self.input_ub += eps

    
    def update_bounds(self, eps):
        ilb = torch.clip(self.input - eps, min=0., max=1.)
        iub = torch.clip(self.input + eps, min=0., max=1.)

        mean, std = get_mean_std(self.dataset)

        ilb = (ilb - mean) / std
        iub = (iub - mean) / std
        self.input = (self.input - mean) / std

        ilb = ilb.reshape(-1)
        iub = iub.reshape(-1)
        self.input_lb = ilb
        self.input_ub = iub

    def __hash__(self):
        return hash((self.input_lb.numpy().tobytes(), self.input_ub.numpy().tobytes(), self.dataset))

    # After has collision Python dict check for equality. Thus, our definition of equality should define both
    def __eq__(self, other):
        if not torch.all(self.input_lb == other.input_lb) or not torch.all(self.input_ub == other.input_ub) \
                or self.dataset != other.dataset or not torch.all(self.out_constr.constr_mat[0] == other.out_constr.constr_mat[0]):
            return False
        return True


    def is_local_robustness(self):
        return self.out_constr.constr_type == OutSpecType.LOCAL_ROBUST

    def get_label(self):
        if self.out_constr.constr_type is not OutSpecType.LOCAL_ROBUST:
            raise ValueError("Label only for local robustness properties!")
        return self.out_constr.label

    def get_input_size(self):
        return self.input_lb.shape[0]

    def is_conjunctive(self):
        return self.out_constr.is_conjunctive

    def output_constr_mat(self):
        return self.out_constr.constr_mat[0]

    def output_constr_const(self):
        return self.out_constr.constr_mat[1]

