from src.specs.input_spec import InputSpecType
from enum import IntEnum

NET_HOME = "/home/debangshu/adaptive-raven/raven/src/nets/"
DEVICE = 'cpu'

# Networks
# Hamming distance networks
MNIST_BINARY_PGD_RELU = "mnist_binary_relu_pgd.pt"
MNIST_BINARY_PGD_SIGMOID = "mnist_binary_sigmoid_pgd.pt"
MNIST_BINARY_PGD_TANH = "mnist_binary_tanh_pgd.pt"

# ConvSmall Diffai
MNIST_CONV_SMALL_DIFFAI = "mnistconvSmallRELUDiffAI.onnx"
# ConvBig Diffai
MNIST_CONV_BIG = 'mnist_convBigRELU__DiffAI.onnx'

# MNIST CROWN-IBP network
# IbpSmall network
MNIST_CROWN_IBP = "mnist_cnn_2layer_width_1_best.pth"
# Ibp big network
MNIST_CROWN_IBP_BIG = "mnist_cnn_3layer_fixed_kernel_3_width_1_best.pth"


# CIFAR CROWN-IBP network
CIFAR_CROWN_IBP = "cifar_cnn_2layer_width_2_best.pth"
CIFAR_CROWN_IBP_BIG = "crown_cifar_cnn_3layer_fixed_kernel_3_width_1_best.pth"

# CIFAR Networks
# ConvSmall Diffai
CIFAR_CONV_DIFFAI = "cifar10convSmallRELUDiffAI.onnx"
# ConvSmall Standard
CIFAR_CONV_SMALL = "convSmallRELU__Point.onnx"
# ConvBig Diffai
CIFAR_CONV_BIG = "cifar_convBigRELU__DiffAI.onnx"


# Housing price networks
HOUSING_RM_CRIM = 'monotonic.onnx'
HOUSING_2LAYER = 'monotonic_2layer.onnx'
HOUSING_2LAYER_100 = 'monotonic_2layer_100.onnx'
HOUSING_2LAYER_200 = 'monotonic_2layer_200.onnx'

# Adult network
ADULT_TANH = 'cls_tanh.onnx'


class MONOTONE_PROP(IntEnum):
    CRIM = 0
    ZN = 1
    INDUS = 2
    NOX = 3
    RM = 4
    AGE = 5
    DIS = 6
    RAD = 7
    TAX = 8
    PTRATIO = 9
    B = 10
    LSTAT = 11


log_file = "log.txt"
log_enabled = False

def write_log(log):
    """Appends string @param: str to log file"""
    if log_enabled:
        f = open(log_file, "a")
        f.write(log + '\n')
        f.close()


tool_name = "RaVeN"
baseline = "Baseline"



linear_models = []

def is_linear_model(net_name):
    for name in linear_models:
        if name in net_name:
            return True
    return False




def ACASXU(i, j):
    net_name = "acasxu/nets/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.onnx"
    return net_name
