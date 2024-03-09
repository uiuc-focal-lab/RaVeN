
from unittest import TestCase
from src import config
from src.specs.property import InputSpecType
from src.common import Domain
from src.common.dataset import Dataset
import src.relational_verifier as relational_ver 
import src.raven_args as raven_args


"""
Tests for untargeted UAP verification for smaller MNIST/CIFAR10 networks.
"""
class TestUntargetedUapSmall(TestCase):
    # Untargeted UAP for ConvSmall MNIST DIFFAI Network.
    def test_convsmall_diffai_mnist(self):
        uap_verfication_args = raven_args.RaVeNArgs(
                individual_prop_domain=Domain.DEEPZ,
                domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.MNIST,
                spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=0.12, 
                net=config.MNIST_CONV_SMALL_DIFFAI, timeout=300, output_dir='pldi-results/',
                uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=True)
        relational_ver.RelationalVerification(uap_verfication_args)

    # Untargeted UAP for IBPSmall MNIST Network.
    def test_ibpsmall_mnist(self):
        uap_verfication_args = raven_args.RaVeNArgs(
                individual_prop_domain=Domain.DEEPZ,
                domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.MNIST,
                spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=0.12, 
                net=config.MNIST_CROWN_IBP, timeout=300, output_dir='pldi-results/',
                uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=True)
        relational_ver.RelationalVerification(uap_verfication_args)

    # Untargeted UAP for ConvSmall CIFAR10 DIFFAI Network.
    def test_convsmall_diffai_cifar10(self):
        uap_verfication_args = raven_args.RaVeNArgs(
                individual_prop_domain=Domain.DEEPZ,
                domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.CIFAR10,
                spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=6.0/255, 
                net=config.CIFAR_CONV_DIFFAI, timeout=300, output_dir='pldi-results/',
                uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=True)
        relational_ver.RelationalVerification(uap_verfication_args)

    # Untargeted UAP for IBPSmall CIFAR10 Network.
    def test_ibpsmall_cifar10(self):
        uap_verfication_args = raven_args.RaVeNArgs(
                individual_prop_domain=Domain.DEEPZ,
                domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.CIFAR10,
                spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=6.0/255, 
                net=config.CIFAR_CROWN_IBP, timeout=300, output_dir='pldi-results/',
                uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=True)
        relational_ver.RelationalVerification(uap_verfication_args)


"""
Tests for untargeted UAP verification for larger MNIST/CIFAR10 networks.
"""
class TestUntargetedUapLarge(TestCase):
    # Untargeted UAP for ConvBig MNIST DIFFAI Network.
    def test_convbig_diffai_mnist(self):
        uap_verfication_args = raven_args.RaVeNArgs(
                individual_prop_domain=Domain.DEEPZ,
                domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.MNIST,
                spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=0.12, 
                net=config.MNIST_CONV_BIG, timeout=300, output_dir='pldi-results/',
                uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=True)
        relational_ver.RelationalVerification(uap_verfication_args)

    # Untargeted UAP for IBP MNIST Network.
    def test_ibp_mnist(self):
        uap_verfication_args = raven_args.RaVeNArgs(
                individual_prop_domain=Domain.DEEPZ,
                domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.MNIST,
                spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=0.12, 
                net=config.MNIST_CROWN_IBP_BIG, timeout=300, output_dir='pldi-results/',
                uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=True)
        relational_ver.RelationalVerification(uap_verfication_args)

    # Untargeted UAP for ConvBig CIFAR10 DIFFAI Network.
    def test_convbig_diffai_cifar10(self):
        uap_verfication_args = raven_args.RaVeNArgs(
                individual_prop_domain=Domain.DEEPZ,
                domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.CIFAR10,
                spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=6.0/255, 
                net=config.CIFAR_CONV_BIG, timeout=300, output_dir='pldi-results/',
                uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=True)
        relational_ver.RelationalVerification(uap_verfication_args)

    # Untargeted UAP for IBP CIFAR10 Network.
    def test_ibp_cifar10(self):
        uap_verfication_args = raven_args.RaVeNArgs(
                individual_prop_domain=Domain.DEEPZ,
                domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.CIFAR10,
                spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=6.0/255, 
                net=config.CIFAR_CROWN_IBP_BIG, timeout=300, output_dir='pldi-results/',
                uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=True)
        relational_ver.RelationalVerification(uap_verfication_args)



"""
Tests for Worst-case Hamming distance verification.
"""
class TestWorstCaseHamming(TestCase):
    # Worst-case Hamming distance verification for MNIST ReLU network.
    def test_mnist_hamming_relu(self):
        uap_verfication_args = raven_args.RaVeNArgs(
                individual_prop_domain=Domain.DEEPPOLY,
                domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.MNIST,
                spec_type=InputSpecType.UAP_BINARY, count=20, count_per_prop=20, eps=0.15, 
                net=config.MNIST_BINARY_PGD_RELU, timeout=300, output_dir='pldi-results/',
                uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=True, 
                lp_formulation_threshold=10)
        relational_ver.RelationalVerification(uap_verfication_args)
    

    # Worst-case Hamming distance verification for MNIST Sigmoid network.
    def test_mnist_hamming_sigmoid(self):
        uap_verfication_args = raven_args.RaVeNArgs(
                individual_prop_domain=Domain.DEEPPOLY,
                domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.MNIST,
                spec_type=InputSpecType.UAP_BINARY, count=20, count_per_prop=20, eps=0.13, 
                net=config.MNIST_BINARY_PGD_SIGMOID, timeout=300, output_dir='pldi-results/',
                uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=True, 
                lp_formulation_threshold=10)
        relational_ver.RelationalVerification(uap_verfication_args)

    # Worst-case Hamming distance verification for MNIST Tanh network.
    def test_mnist_hamming_tanh(self):
        uap_verfication_args = raven_args.RaVeNArgs(
                individual_prop_domain=Domain.DEEPPOLY  ,
                domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.MNIST,
                spec_type=InputSpecType.UAP_BINARY, count=20, count_per_prop=20, eps=0.1, 
                net=config.MNIST_BINARY_PGD_TANH, timeout=300, output_dir='pldi-results/',
                uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=True, 
                lp_formulation_threshold=10)
        relational_ver.RelationalVerification(uap_verfication_args)
