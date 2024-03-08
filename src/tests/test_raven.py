
from unittest import TestCase
from src import config
from src.specs.property import InputSpecType
from src.common import Domain
from src.common.dataset import Dataset
import src.relational_verifier as relational_ver 
import src.raven_args as raven_args


class TestBasicUap(TestCase):   
    def test_mnist_uap(self):
        uap_verfication_args = raven_args.RaVeNArgs(
            individual_prop_domain=Domain.DEEPPOLY,
            domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.MNIST, sink_label=None,
            spec_type=InputSpecType.UAP, count=1, count_per_prop=2, eps=0.15, net=config.MNIST_FFN_01,                                                                                                              
            timeout=100, output_dir='results_trial/', radius_l=0.002, radius_r=0.25,
            uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=False,
            no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=10,
            try_image_smoothing=False, filter_threshold=None, fold_conv_layers=False)
        relational_ver.RelationalVerification(uap_verfication_args)

    def test_mnist_uap_full(self):
        eps = 0.15
        for _ in range(10):
            uap_verfication_args = raven_args.RaVeNArgs(
                individual_prop_domain=Domain.DEEPPOLY,
                domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.MNIST, sink_label=None,
                spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=eps, net=config.MNIST_CONV_SMALL_DIFFAI,                                                                                                              
                timeout=100, output_dir='pldi-results/', radius_l=0.002, radius_r=0.25,
                uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=True,
                no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=2,
                try_image_smoothing=False, filter_threshold=None)
            eps += 0.005
            relational_ver.RelationalVerification(uap_verfication_args)
        
        eps = 0.15
        for _ in range(10):
            uap_verfication_args = raven_args.RaVeNArgs(
                individual_prop_domain=Domain.DEEPZ,
                domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.MNIST, sink_label=None,
                spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=eps, net=config.MNIST_CROWN_IBP,                                                                                                              
                timeout=100, output_dir='pldi-results/', radius_l=0.002, radius_r=0.25,
                uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=True,
                no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=3,
                try_image_smoothing=False, filter_threshold=None)
            eps += 0.005
            relational_ver.RelationalVerification(uap_verfication_args)

    def test_mnist_uap_full_med(self):
        eps = 0.1
        for _ in range(10):
            uap_verfication_args = raven_args.RaVeNArgs(
                individual_prop_domain=Domain.DEEPZ,
                domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.MNIST, sink_label=None,
                spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=eps, net=config.MNIST_CROWN_IBP_MED,                                                                                                              
                timeout=100, output_dir='pldi-results/', radius_l=0.002, radius_r=0.25,
                uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=True,
                no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=2,
                try_image_smoothing=False, filter_threshold=-15.0)
            eps += 0.005
            relational_ver.RelationalVerification(uap_verfication_args)
        
    # def test_mnist_hamming_full(self):
        # eps = 0.1
        # for _ in range(20):
        #     uap_verfication_args = raven_args.RaVeNArgs(
        #         individual_prop_domain=Domain.DEEPZ,
        #         domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.MNIST, sink_label=None,
        #         spec_type=InputSpecType.UAP_BINARY, count=20, count_per_prop=10, eps=eps, net=config.MNIST_BINARY_PGD_RELU,                                                                                                              
        #         timeout=100, output_dir='pldi-results/', radius_l=0.002, radius_r=0.25,
        #         uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=True,
        #         no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=5,
        #         try_image_smoothing=False, filter_threshold=-300.0)
        #     eps += 0.01
        #     relational_ver.RelationalVerification(uap_verfication_args)

        # eps = 0.05
        # for _ in range(5):
        #     uap_verfication_args = raven_args.RaVeNArgs(
        #         individual_prop_domain=Domain.DEEPPOLY,
        #         domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.MNIST, sink_label=None,
        #         spec_type=InputSpecType.UAP_BINARY, count=20, count_per_prop=10, eps=eps, net=config.MNIST_BINARY_PGD_SIGMOID,                                                                                                              
        #         timeout=100, output_dir='pldi-results/', radius_l=0.002, radius_r=0.25,
        #         uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=True,
        #         no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=5,
        #         try_image_smoothing=False, filter_threshold=-300.0)
        #     eps += 0.01
        #     relational_ver.RelationalVerification(uap_verfication_args)

        # eps = 0.1
        # for _ in range(20):
        #     uap_verfication_args = raven_args.RaVeNArgs(
        #         individual_prop_domain=Domain.DEEPPOLY,
        #         domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.MNIST, sink_label=None,
        #         spec_type=InputSpecType.UAP_BINARY, count=20, count_per_prop=10, eps=eps, net="mnist_binary_relu_pgd.pt",                                                                                                              
        #         timeout=100, output_dir='pldi-results-hamming/', radius_l=0.002, radius_r=0.25,
        #         uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=True,
        #         no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=5,
        #         try_image_smoothing=False, filter_threshold=-300.0)
        #     eps += 0.01
        #     relational_ver.RelationalVerification(uap_verfication_args)

    def test_mnist_hamming_full(self):
        eps = 0.23
        for _ in range(3):
            uap_verfication_args = raven_args.RaVeNArgs(
                individual_prop_domain=Domain.DEEPPOLY,
                domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.MNIST, sink_label=None,
                spec_type=InputSpecType.UAP_BINARY, count=20, count_per_prop=10, eps=eps, net="mnist_binary_relu_pgd.pt",                                                                                                              
                timeout=100, output_dir='pldi-results-hamming/', radius_l=0.002, radius_r=0.25,
                uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=True,
                no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=5,
                try_image_smoothing=False, filter_threshold=-350.0)
            eps += 0.01
            relational_ver.RelationalVerification(uap_verfication_args)
        

    def test_mnist_uap_full_big(self):        
        eps = 0.15
        for _ in range(10):
            uap_verfication_args = raven_args.RaVeNArgs(
                individual_prop_domain=Domain.DEEPZ,
                domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.MNIST, sink_label=None,
                spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=eps, net=config.MNIST_CONV_BIG,                                                                                                              
                timeout=100, output_dir='pldi-results/', radius_l=0.002, radius_r=0.25,
                uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=True,
                no_lp_for_verified = True, debug_mode=False, track_differences=False, lp_formulation_threshold=2,
                try_image_smoothing=False, filter_threshold=-2.0)
            eps += 0.01
            relational_ver.RelationalVerification(uap_verfication_args)




    def test_mnist_uap_debug(self):
        uap_verfication_args = raven_args.RaVeNArgs(
            individual_prop_domain=Domain.DEEPPOLY,
            domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.MNIST, sink_label=None,
            spec_type=InputSpecType.UAP, count=1, count_per_prop=2, eps=0.1, net=config.MNIST_BINARY,                                                                                                              
            timeout=100, output_dir='results/', radius_l=0.002, radius_r=0.25,
            uap_mode=raven_args.RavenMode.UAP, compute_proportion=False, write_file=False,
            no_lp_for_verified = True, debug_mode=True, track_differences=True)
        relational_ver.RelationalVerification(uap_verfication_args)

    def test_cifar_uap(self):
        uap_verfication_args = raven_args.RaVeNArgs(
            individual_prop_domain=Domain.DEEPPOLY,
            domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.CIFAR10, sink_label=None,
            spec_type=InputSpecType.UAP, count=1, count_per_prop=5, eps=4/255, net=config.CIFAR_CONV_DIFFAI,                                                                                                              
            timeout=100, output_dir='results/', radius_l=0.002, radius_r=0.25, 
            uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=False,
            no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=3,
            try_image_smoothing=False, filter_threshold=None)
        relational_ver.RelationalVerification(uap_verfication_args)


    def test_cifar_uap_full(self):

        eps = 2.0
        for _ in range(12):
            uap_verfication_args = raven_args.RaVeNArgs(
                individual_prop_domain=Domain.DEEPZ,
                domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.CIFAR10, sink_label=None,
                spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=eps/255, net=config.CIFAR_CROWN_IBP,                                                                                                              
                timeout=100, output_dir='results_trial/', radius_l=0.002, radius_r=0.25, 
                uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=True,
                no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=3,
                try_image_smoothing=False, filter_threshold=None)
            relational_ver.RelationalVerification(uap_verfication_args)
            eps += 0.35

        eps = 2.0
        for _ in range(12):
            uap_verfication_args = raven_args.RaVeNArgs(
                individual_prop_domain=Domain.DEEPZ,
                domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.CIFAR10, sink_label=None,
                spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=eps/255, net=config.CIFAR_CONV_DIFFAI,                                                                                                              
                timeout=100, output_dir='results_trial/', radius_l=0.002, radius_r=0.25, 
                uap_mode=raven_args.RavenMode.UAP, compute_proportion=True, write_file=True,
                no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=3,
                try_image_smoothing=False, filter_threshold=None)
            relational_ver.RelationalVerification(uap_verfication_args)
            eps += 0.35

class TestTargetedUap(TestCase):   
    def test_mnist_uap(self):
        uap_verfication_args = raven_args.RaVeNArgs(
            individual_prop_domain=Domain.DEEPZ,
            domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.MNIST, sink_label=None,
            spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=0.1, net=config.MNIST_CONV_PGD,                                                                                                              
            timeout=100, output_dir='results/', radius_l=0.002, radius_r=0.25,
            uap_mode=raven_args.RavenMode.TARGETED, compute_proportion=True, write_file=True,
            no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=2)
        relational_ver.RelationalVerification(uap_verfication_args)

    def test_mnist_uap_debug(self):
        uap_verfication_args = raven_args.RaVeNArgs(
            individual_prop_domain=Domain.DEEPZ,
            domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.MNIST, sink_label=None,
            spec_type=InputSpecType.UAP, count=1, count_per_prop=5, eps=0.1, net=config.MNIST_BINARY,                                                                                                              
            timeout=100, output_dir='results/', radius_l=0.002, radius_r=0.25,
            uap_mode=raven_args.RavenMode.TARGETED, compute_proportion=False, write_file=False,
            no_lp_for_verified = True, debug_mode=True, track_differences=True)
        relational_ver.RelationalVerification(uap_verfication_args)

    def test_cifar_uap(self):
        uap_verfication_args = raven_args.RaVeNArgs(
            individual_prop_domain=Domain.DEEPZ,
            domain=Domain.RAVEN, baseline_domain=Domain.IOFORMULATION, dataset=Dataset.CIFAR10, sink_label=None,
            spec_type=InputSpecType.UAP, count=2, count_per_prop=5, eps=3/255, net=config.CIFAR_CONV_BIG,                                                                                                              
            timeout=100, output_dir='results/', radius_l=0.002, radius_r=0.25, 
            uap_mode=raven_args.RavenMode.TARGETED, compute_proportion=True, write_file=False,
            no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=2)
        relational_ver.RelationalVerification(uap_verfication_args)
