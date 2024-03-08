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
                 eps=0.01, net='', timeout=30, output_dir='', radius_l=0.1, 
                 radius_r=0.3, uap_mode=RavenMode.UAP, cutoff_percentage = 0.5,
                 compute_proportion=False, no_lp_for_verified=False, write_file = False, 
                 debug_mode=False, track_differences=True, monotone_prop = None, monotone_inv = False, 
                 lp_formulation_threshold=2, try_image_smoothing=False, filter_threshold=None, 
                 fold_conv_layers=False, ligweight_diffpoly=False) -> None:
        self.individual_prop_domain = individual_prop_domain
        self.domain = domain
        self.baseline_domain = baseline_domain
        self.spec_type = spec_type
        self.dataset= dataset
        self.count = count
        self.count_per_prop = count_per_prop
        self.sink_label = sink_label
        self.eps = eps
        self.net = config.NET_HOME + net
        self.net_name = net
        self.timeout = timeout
        self.output_dir = output_dir
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
        # Always use all layer substitution for DiffPoly.
        self.all_layer_sub = True
        self.fold_conv_layers = fold_conv_layers
        self.lightweight_diffpoly = ligweight_diffpoly
        # if debug mode on rewrite params.
        if debug_mode == True:
            self.count = 1
            self.count_per_prop = 2
            self.eps = 6.0
            self.net_name = 'debug.net'
            self.net = 'debug.net'
