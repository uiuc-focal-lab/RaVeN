from src.common import Domain
from src.relational_domains.individual_domain import Individual
from src.relational_domains.io_formulation_domain import IOFormulation
from src.relational_domains.raven import RaVeN

def get_relational_domain_transformer(domain, net, props, args, baseline_results):
    if domain is Domain.INDIVIDUAL:
        transformer = Individual(net=net, props=props, args=args, baseline_results=baseline_results)
    elif domain is Domain.IOFORMULATION:
        transformer = IOFormulation(net=net, props=props, args=args, baseline_results=baseline_results)
    elif domain is Domain.RAVEN:
        transformer = RaVeN(net=net, props=props, args=args, baseline_results=baseline_results)
    else:
         raise ValueError(f"Unrecognized relational domain {domain}") 
    return transformer