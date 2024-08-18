import torch
import src.util as util 
from src.raven_results import *
from src.common import Status, Domain
from src.domains.domain_transformer import domain_transformer
from src.baseline_analyzer import BaselineAnalyzerBackend
from src.relational_domains.relational_domain_transformer import get_relational_domain_transformer
from src.common.network import LayerType
import time


"""
RaVeN backend
Inputs: properties to verify, args.
Output: % of verified properties (UAP, monotonicity) or worst-case hamming distance.
"""
class RelationalVerifierBackendWrapper:
    def __init__(self, props, args) -> None:
        self.props = props
        self.args = args
        with torch.no_grad():
            self.net = util.get_net(self.args.net, self.args.dataset, debug_mode=self.args.debug_mode)    
    
    # Monotonicity verification.
    def run_monotone(self, monotone_prop) -> RavenResult:
        start_time = time.time()
        with torch.no_grad():
            baseline_verifier = BaselineAnalyzerBackend(props=self.props, net=self.net, args=self.args)
            individual_verification_results = baseline_verifier.run()
        individual_time = time.time() - start_time
        start_time = time.time()
        uap_algorithm_res = self.run_raven_backend(domain=self.args.domain, individual_verification_results=individual_verification_results, monotone = True, monotonic_inv = self.args.monotone_inv)
        uap_time = time.time() - start_time        
        return RavenResult(baseline_res = None, raven_res=uap_algorithm_res, individual_res=individual_verification_results, monotone = True, props = self.props, times = uap_time)

    # Targeted UAP verification.
    def run_targeted_uap(self) -> RavenResult:
        start_time = time.time()
        baseline_verifier = BaselineAnalyzerBackend(props=self.props, net=self.net, args=self.args)
        individual_verification_results = baseline_verifier.run()
        individual_time = time.time() - start_time

        # Run the I/O formulation verifier.
        start_time = time.time()
        baseline_res = self.run_raven_backend(domain=self.args.baseline_domain, individual_verification_results=individual_verification_results, targeted = True)
        baseline_time = time.time() - start_time
        baseline_res.timings = LP_TIMINGS(total_time=individual_time + baseline_time,
                                                            constraint_formulation_time=None,
                                                            optimization_time=None)

        # Run the UAP Verifier for Targeted UAP.
        start_time = time.time()
        uap_algorithm_res = self.run_raven_backend(domain=self.args.domain, individual_verification_results=individual_verification_results, targeted = True)
        uap_time = time.time() - start_time

        # Populate Timings.
        uap_algorithm_res.timings = LP_TIMINGS(total_time=(individual_time + uap_time), 
                                     constraint_formulation_time=uap_algorithm_res.constraint_time,
                                     optimization_time=uap_algorithm_res.optimize_time)

        return RavenResult(baseline_res=baseline_res, raven_res=uap_algorithm_res, individual_res=individual_verification_results, targeted = True, times = [individual_time, uap_time], individual_time=individual_time, props = self.props)

    # Untargeted UAP and worst-case hamming distance verification.
    def run_untargeted_uap(self) -> RavenResult:
        start_time = time.time()
        baseline_verifier = BaselineAnalyzerBackend(props=self.props, net=self.net, args=self.args)
        individual_verification_results = baseline_verifier.run()
        individual_time = time.time() - start_time

        start_time = time.time()
        baseline_res = self.run_raven_backend(domain=self.args.baseline_domain, 
                                                individual_verification_results=individual_verification_results, diff = False)
        baseline_time = time.time() - start_time
        baseline_res.timings = LP_TIMINGS(total_time=individual_time + baseline_time,
                                                            constraint_formulation_time=None,
                                                            optimization_time=None)
        
        # Run the uap verifier without diff constraints.
        start_time = time.time()
        uap_algorithm_no_diff_res = self.run_raven_backend(domain=self.args.domain, 
                                                 individual_verification_results=individual_verification_results, 
                                                 diff=False)
        uap_diff_time = time.time() - start_time

        # Populate timings.
        uap_algorithm_no_diff_res.timings = LP_TIMINGS(total_time=(individual_time + uap_diff_time), 
                                     constraint_formulation_time=uap_algorithm_no_diff_res.constraint_time,
                                     optimization_time=uap_algorithm_no_diff_res.optimize_time)
        # Run the uap verifier with diff constraints.
        uap_algorithm_res = None
        uap_timing = None
        if self.args.track_differences is True:
            start_time = time.time()
            uap_algorithm_res = self.run_raven_backend(domain=self.args.domain, 
                                                    individual_verification_results=individual_verification_results, 
                                                    diff=True)
            if uap_algorithm_res.verified_proportion is not None:
                uap_algorithm_res.verified_proportion = max(uap_algorithm_no_diff_res.verified_proportion, 
                                                        uap_algorithm_res.verified_proportion) 
            uap_time = time.time() - start_time
            uap_algorithm_res.timings = LP_TIMINGS(total_time=(individual_time + uap_time), 
                                constraint_formulation_time=uap_algorithm_res.constraint_time,
                                optimization_time=uap_algorithm_res.optimize_time)
        
        return RavenResult(baseline_res=baseline_res, raven_res=uap_algorithm_res, individual_res=individual_verification_results,
                          result_with_no_diff=uap_algorithm_no_diff_res, times=None, individual_time=individual_time)

    def run(self) -> RavenResult:
        # Baseline results correspond to running each property individually.
        with torch.no_grad():
            baseline_verifier = BaselineAnalyzerBackend(props=self.props, net=self.net, args=self.args)
            individual_verification_results = baseline_verifier.run()
        baseline_res = self.run_raven_backend(domain=self.args.baseline_domain, 
                                                 individual_verification_results=individual_verification_results)
        uap_algorithm_res = self.run_raven_backend(domain=self.args.domain, 
                                                 individual_verification_results=individual_verification_results)
        return RavenResult(baseline_res=baseline_res, raven_res=uap_algorithm_res)


    def run_raven_backend(self, domain, individual_verification_results, targeted = False,
                                    monotone = False, monotone_prop = 0, monotonic_inv = False,
                                    diff = True):
        uap_verifier = get_relational_domain_transformer(domain=domain, net=self.net, props=self.props, 
                                                           args=self.args, 
                                                           baseline_results=individual_verification_results)
        if self.args.no_lp_for_verified == True and domain == Domain.RAVEN:
            uap_verifier.no_lp_for_verified = True
        return uap_verifier.run(proportion=self.args.compute_proportion, targeted = targeted,
                                 monotone = monotone, monotonic_inv = monotonic_inv, diff = diff)
