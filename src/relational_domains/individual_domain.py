import torch
from src.raven_results import RavenSingleRes
from src.common import Status

class Individual:
    def __init__(self, net, props, args, baseline_results) -> None:
        self.net = net
        self.props = props
        self.args = args
        self.baseline_results = baseline_results


    def run(self) -> RavenSingleRes:
        verified_count = 0
        total_time = 0
        lbs = []
        for baseline_res in self.baseline_results:
            lb = torch.min(baseline_res.final_lb)
            lbs.append(lb)
            if lb >= 0.0:
                verified_count += 1
            total_time += baseline_res.time
        verified_status = Status.UNKNOWN
        verified_proportion = verified_count / len(self.baseline_results)
        if verified_proportion >= self.args.cutoff_percentage:
            verified_status = Status.VERIFIED
        lbs.sort()
        print("lbs", lbs)
        return RavenSingleRes(domain=self.args.domain, input_per_prop=self.args.count_per_prop,
                            status=verified_status, global_lb=None, time_taken=total_time, 
                            verified_proportion=verified_proportion)