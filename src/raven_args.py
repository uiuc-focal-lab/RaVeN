import src.config as config
from src.specs.input_spec import InputSpecType
from enum import Enum

class RavenMode(Enum):
    UAP = 1
    TARGETED = 2
    MONOTONICITY = 3


class RaVeNArgs:
    def __init__(self, individual_prop_domain, domain, baseline_domain, dataset='mnist', sink_label=None, 
                 spec_type=InputSpecType.UAP, count=None, count_per_prop=2, 
                 eps=0.01, net='', timeout=300, output_dir='', radius_l=0.1, 
                 radius_r=0.3, uap_mode=RavenMode.UAP, cutoff_percentage = 0.5,
                 compute_proportion=True, no_lp_for_verified=True, write_file = False, 
                 debug_mode=False, track_differences=True, enable_ablation = False,
                 monotone_prop = None, monotone_inv = False, lp_formulation_threshold=3, 
                 try_image_smoothing=False, filter_threshold=None, 
                 fold_conv_layers=False, ligweight_diffpoly=False,
                 monotone_splits = 1, monotone_lp = False, device='cpu') -> None:

        # Individual verification Domain e.g. DeepZ, DeepPoly, etc (see src/common/__init__.py).
        self.individual_prop_domain = individual_prop_domain
        # relational domain e.g. RaVeN (see src/common/__init__.py).
        self.domain = domain
        # Baseline relational domain e.g. I/O Formulation (see src/common/__init__.py).
        self.baseline_domain = baseline_domain
        # spec type e.g. UAP, UAP_BINARY (src/specs/input_spec.py).
        self.spec_type = spec_type
        # Dataset e.g. MNIST / CIFAR10 (src/common/dataset.py).
        self.dataset= dataset
        # Number of properties to verify e.g. 20.
        self.count = count
        # Number of executions (k) per property e.g. 5.
        self.count_per_prop = count_per_prop
        # Perturbation bound e.g. 0.13.
        self.eps = eps
        # Location of the network (see src/config.py).
        self.net = config.NET_HOME + net
        self.net_name = net
        # Raven mode e.g. UAP or MONOTONICITY
        self.uap_mode = uap_mode

        # Additional parameters for customization.
        self.output_dir = output_dir
        self.enable_ablation = enable_ablation
        self.timeout = timeout
        self.sink_label = sink_label
        self.radius_l = radius_l
        self.radius_r = radius_r
        self.uap_mode = uap_mode
        self.cutoff_percentage = cutoff_percentage
        self.compute_proportion = compute_proportion
        self.no_lp_for_verified = no_lp_for_verified
        self.write_file = write_file
        self.debug_mode = debug_mode
        self.track_differences = track_differences
        self.monotone_prop = monotone_prop
        self.monotone_inv = monotone_inv
        self.lp_formulation_threshold = lp_formulation_threshold
        self.try_image_smoothing = try_image_smoothing
        self.filter_threshold = filter_threshold
        self.all_layer_sub = True
        self.fold_conv_layers = fold_conv_layers
        self.lightweight_diffpoly = ligweight_diffpoly
        self.monotone_splits = monotone_splits
        self.monotone_lp = monotone_lp
        self.device = device
        # if debug mode on rewrite params.
        if debug_mode == True:
            self.count = 1
            self.count_per_prop = 2
            self.eps = 6.0
            self.net_name = 'debug.net'
            self.net = 'debug.net'
