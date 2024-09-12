import sys
# Should load the local local libraries like auto_LiRPA  
sys.path.append('../')

from src.raven_args import RaVeNArgs, RavenMode
import src.specs.spec as specs
from src.relational_verifier_backend import RelationalVerifierBackendWrapper
from src.raven_results import RavenResultList
import time
from src.common import Status
import math


"""
RaVeN : Relational Verifier of Neural Networks
Currently supported relational properties
1. Untargeted UAP
2. Targeted UAP
3. Monotonicity
4. Worst-case hamming distance
"""
def RelationalVerification(raven_args: RaVeNArgs):
    total_local_prop_count = raven_args.count * raven_args.count_per_prop
    # Load the input and output specifications.
    props, _ = specs.get_specs(raven_args.dataset, spec_type=raven_args.spec_type,
                                    count=total_local_prop_count, eps=raven_args.eps, 
                                    sink_label=raven_args.sink_label,
                                    debug_mode=raven_args.debug_mode, 
                                    monotone_prop = raven_args.monotone_prop, 
                                    monotone_inv = raven_args.monotone_inv, 
                                    monotone_splits = raven_args.monotone_splits,
                                    try_input_smoothing=raven_args.try_image_smoothing,
                                    count_per_prop=raven_args.count_per_prop, 
                                    net_name=raven_args.net)
    
    if raven_args.uap_mode is RavenMode.UAP:
        UapVerificationBackend(props=props, raven_args=raven_args)
    elif raven_args.uap_mode is RavenMode.TARGETED:
        UapTargetedBackend(props=props, raven_args=raven_args)
    elif raven_args.uap_mode is RavenMode.MONOTONICITY:
        MonotonicityBackend(props=props, raven_args=raven_args)

# Backend for verifying untargeted UAP and worst-case hamming distance.
def UapVerificationBackend(props, raven_args):
    uap_prop_count = raven_args.count
    input_per_prop = raven_args.count_per_prop
    raven_result_list = RavenResultList()
    for i in range(uap_prop_count):
        print(f"\n\n ***** verifying property {i+1} ***** \n\n")
        props_to_analyze = props[i * input_per_prop : (i+1) * input_per_prop] 
        uap_verifier = RelationalVerifierBackendWrapper(props=props_to_analyze, args=raven_args)
        
        # Run Untargeted UAP verification. 
        res = uap_verifier.run_untargeted_uap()
        raven_result_list.add_results(res)
        raven_result_list.print_last_Uap_targeted(args=raven_args)
    if raven_args.write_file == True:
       raven_result_list.analyze(raven_args)

# Backend for verifying targeted UAP.
def UapTargetedBackend(props, raven_args):
    uap_prop_count = raven_args.count
    input_per_prop = raven_args.count_per_prop
    raven_result_list = RavenResultList()
    for i in range(uap_prop_count):
        print(f"\n\n ***** verifying property {i+1} ***** \n\n")
        props_to_analyze = props[i * input_per_prop : (i+1) * input_per_prop]
        targeted_uap_verifier = RelationalVerifierBackendWrapper(props=props_to_analyze, args=raven_args)
        # Run the targeted uap verification
        res = targeted_uap_verifier.run_targeted_uap()
        raven_result_list.add_results(res)
        raven_result_list.print_last_Uap_targeted(args=raven_args)
    if raven_args.write_file == True:
       raven_result_list.analyze_targeted(raven_args)



# Backend for verifying monotonicity.

def MonotonicityBackend(props, raven_args):
    uap_prop_count = raven_args.count * raven_args.monotone_splits
    #print(uap_prop_count)
    input_per_prop = raven_args.count_per_prop * 2
    raven_result_list = RavenResultList()
    continue_until = -1
    for i in range(uap_prop_count):
        if continue_until != -1:
            if i < continue_until:
                #print(f'Skipping Property {i} until {continue_until}')
                continue
            continue_until = -1
        #print(f"\n\n ***** verifying property {i} ***** \n\n")
        props_to_analyze = props[i * input_per_prop : (i+1) * input_per_prop]
        monotonicity_verifier = RelationalVerifierBackendWrapper(props=props_to_analyze, args=raven_args)
        # run the uap verification
        res = monotonicity_verifier.run_monotone(raven_args.monotone_prop)
        raven_result_list.add_results(res)
        if res.raven_res.status != Status.VERIFIED:
            continue_until = int(math.ceil(i/raven_args.monotone_splits) * raven_args.monotone_splits)
    if raven_args.write_file == True:
       raven_result_list.analyze_monotone(raven_args)

# def MonotonicityBackend(props, raven_args):
#     uap_prop_count = raven_args.count
#     input_per_prop = raven_args.count_per_prop * 2
#     raven_result_list = RavenResultList()
#     for i in range(uap_prop_count):
#         print(f"\n\n ***** verifying property {i+1} ***** \n\n")
#         props_to_analyze = props[i * input_per_prop : (i+1) * input_per_prop]
#         monotonicity_verifier = RelationalVerifierBackendWrapper(props=props_to_analyze, args=raven_args)
#         # Run the monotonicity verification
#         res = monotonicity_verifier.run_monotone(raven_args.monotone_prop)
#         raven_result_list.add_results(res)
#     if raven_args.write_file == True:
#        raven_result_list.analyze_monotone(raven_args)

