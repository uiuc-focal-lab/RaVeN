import src.util as util
import src.network_converters.parse as parse
from src.common import Domain

def domain_transformer(net, prop, domain, args, prop_list=None):
    domain_builder = util.get_domain_builder(domain)
    transformer = domain_builder(prop, complete=False, args=args, prop_list=prop_list)
    transformer = parse.get_transformer(transformer, net)
    return transformer
