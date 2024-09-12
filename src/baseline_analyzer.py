import torch
import src.util as util 
from src.raven_results import *
from src.common import Status
from src.domains.domain_transformer import domain_transformer
from src.common import Domain
import time

class BaselineAnalyzerBackend:
    def __init__(self, props, net, args):
        self.props = props
        self.net = net 
        self.args = args

    # Return a list of baseline results containing all important metadata.
    def run(self):
        baseline_results = []
        lirpa_domains = [Domain.LIRPA, Domain.LIRPA_ALPHA_CROWN]
        if self.args.enable_batch_processing and self.args.individual_prop_domain in lirpa_domains:
            prop_list = []
            start_time = time.time()
            for prop in self.props:
                assert prop.get_input_clause_count() == 1
                prop_list.append(prop.get_input_clause(0))
            transformer = domain_transformer(net=self.net, prop=self.props[0].get_input_clause(0), 
                                            domain=self.args.individual_prop_domain, 
                                            args=self.args, prop_list=prop_list)
            baseline_results = transformer.populate_result_list()
            time_diff = time.time() - start_time
            assert len(baseline_results) > 0
            for baseline_res in baseline_results:
                baseline_res.time = time_diff / len(baseline_results)
            
            print(f"Time diff {time_diff}\n")

        else:       
            for prop in self.props:
                start_time = time.time()
                assert prop.get_input_clause_count() == 1
                transformer = domain_transformer(net=self.net, prop=prop.get_input_clause(0), 
                                                domain=self.args.individual_prop_domain, 
                                                args=self.args)
                baseline_res = transformer.populate_baseline_verifier_result(args=self.args)
                time_diff = time.time() - start_time
                baseline_res.time = time_diff
                baseline_results.append(baseline_res)
        return baseline_results