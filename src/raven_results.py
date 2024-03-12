from src.common import Status
import torch
from src.common.dataset import Dataset
from src.specs.input_spec import InputSpecType

class LP_TIMINGS:
    def __init__(self, total_time, constraint_formulation_time, optimization_time):
        self.total_time = total_time
        self.constraint_formulation_time = constraint_formulation_time
        self.optimization_time = optimization_time

class RavenSingleRes:
    def __init__(self, domain, input_per_prop, status, global_lb, 
                 time_taken, verified_proportion, constraint_time = None, 
                 optimize_time = None, bin_size = None, timings=None):
        self.domain = domain
        self.status = status
        self.input_per_prop = input_per_prop
        self.global_lb = global_lb
        self.time_taken = time_taken
        self.verified_proportion = verified_proportion
        self.constraint_time = constraint_time
        self.optimize_time = optimize_time
        self.bin_size = bin_size
        # Holds the final timings of the entire analysis.
        self.timings = timings

    def print(self):
        print("Domain ", self.domain)
        print("Time taken ", self.time_taken)
        print("Global lb ", self.global_lb)
        print("Status ", self.status)
        print("Input per prop ", self.input_per_prop)
        print("Verified proportion", self.verified_proportion)

class RavenResult:
    def __init__(self, raven_res, baseline_res, individual_res = None, result_with_no_diff=None,
                targeted = False, times = None, props = None, monotone = False, individual_time=None):
        self.baseline_res = baseline_res
        self.individual_time = individual_time
        self.raven_res = raven_res
        self.result_with_no_diff = result_with_no_diff
        self.individual_res = individual_res
        self.times = times
        self.targeted = targeted
        self.props = props
        self.monotone = monotone
    

class RavenResultList:
    def __init__(self) -> None:
        self.result_list = []

    def add_results(self, res: RavenResult):
        self.result_list.append(res)

    def print_last_Uap_targeted(self, args):
        count = 1
        individual_verified_count = 0
        baseline_verified_count = 0
        uap_verified_without_diff = 0
        uap_verified_count = 0
        times = [0, 0, 0, 0]
        layerwise_constraint_time = 0
        layerwise_optimization_time = 0
        diff_constraint_time = 0
        diff_optimization_time = 0
        if len(self.result_list) <= 0:
            return
        res = self.result_list[-1]
        individual_res = res.individual_res
        baseline_res = res.baseline_res
        uap_no_diff_res = res.result_with_no_diff
        raven_res = res.raven_res
        
        if args.spec_type == InputSpecType.UAP_TARGETED:
            raven_per = torch.tensor([a*100.0 for a in raven_res.verified_proportion])
            print('RaVeN certified UAP accuracy: {}  %\n'.format(raven_per))
            if individual_res is not None:
                deepz_res = [[] for i in range(10)]
                for i in range(len(individual_res)):
                    for j in range(10):
                        if res.props[i].out_constr.label == j:
                            continue
                        deepz_res[j].append((individual_res[i].target_ubs[j]).min() <= 0)
                veri = torch.tensor([(sum(res)).item() for res in deepz_res])
                individual_verified_count += veri
                individual_per = torch.tensor([(veri[i]/len(deepz_res[i])).item()* 100.0 for i in range(len(deepz_res))])
                print(f"Individual certified UAP accuracy: {individual_per}%\n")
            
            diff_individual = (raven_per - individual_per)
            print(f'Improvement over Individual {diff_individual} %\n')
            return
        if individual_res is not None:
            veri = sum([torch.min(res.final_lb) >= 0 for res in individual_res])
            individual_verified_count += veri
            individual_verified_count = float(individual_verified_count/args.count_per_prop)
            individual_verified_count *= 100
            if res.individual_time is not None:
                times[0] += res.individual_time
        if baseline_res is not None and baseline_res.verified_proportion is not None:
            baseline_verified_count += baseline_res.verified_proportion * 100.0
            if times[1] is not None and baseline_res.timings is not None:
                times[1] += baseline_res.timings.total_time
        if uap_no_diff_res is not None and uap_no_diff_res.verified_proportion is not None:
            uap_verified_without_diff += uap_no_diff_res.verified_proportion * 100.0
            if times[2] is not None and uap_no_diff_res.timings is not None:
                times[2] += uap_no_diff_res.timings.total_time
                if uap_no_diff_res.timings.constraint_formulation_time is not None:
                    layerwise_constraint_time += uap_no_diff_res.timings.constraint_formulation_time
                if uap_no_diff_res.timings.optimization_time is not None:
                    layerwise_optimization_time += uap_no_diff_res.timings.optimization_time
        if raven_res is not None and raven_res.verified_proportion is not None:
            uap_verified_count += raven_res.verified_proportion * 100.0
            if times[3] is not None and raven_res.timings is not None:
                times[3] += raven_res.timings.total_time
                if raven_res.timings.constraint_formulation_time is not None:
                    diff_constraint_time += raven_res.timings.constraint_formulation_time
                if raven_res.timings.optimization_time is not None:
                    diff_optimization_time += raven_res.timings.optimization_time
        for i, _  in enumerate(times):
            if times[i] is not None and count > 0:
                times[i] /= count
        if count > 0:
            layerwise_constraint_time /= count
            layerwise_optimization_time /= count
            diff_constraint_time /= count
            diff_optimization_time /= count

        if args.spec_type == InputSpecType.UAP:
            print('Individual certified UAP accuracy: {:0.2f} %\n'.format(individual_verified_count))
            print('I/O Formulation certified UAP accuracy: {:0.2f} %\n'.format(baseline_verified_count))
            if args.enable_ablation:
                print('RaVeN no difference constraints certified UAP accuracy: {:0.2f} %\n'.format(uap_verified_without_diff))
            print('RaVeN certified UAP accuracy: {:0.2f}  %\n'.format(uap_verified_count))
            diff_individual = (uap_verified_count - individual_verified_count)
            print('Improvement over Individual {:0.2f} %\n'.format(diff_individual))
            diff_ioformulation = (uap_verified_count - baseline_verified_count)
            print('Improvement over I/O Formulation {:0.2f} %\n'.format(diff_ioformulation))
            if args.enable_ablation:
                diff_ablation = (uap_verified_count - uap_verified_without_diff)
                print('Improvement over no difference constraints {:0.2f} %\n'.format(diff_ablation))

        if args.spec_type == InputSpecType.UAP_BINARY:
            print('Individual worst-case hamming distance: {:0.2f} \n'.format(args.count_per_prop - individual_verified_count * args.count_per_prop/ 100.0))
            print('I/O Formulation worst-case hamming distance: {:0.2f} \n'.format(args.count_per_prop - baseline_verified_count* args.count_per_prop/ 100.0))
            if args.enable_ablation:
                print('RaVeN no difference constraints worst-case hamming distance: {:0.2f} \n'.format(args.count_per_prop - uap_verified_without_diff* args.count_per_prop/ 100.0))
            print('RaVeN worst-case hamming distance: {:0.2f}  \n'.format(args.count_per_prop - uap_verified_count* args.count_per_prop/ 100.0))
            diff_individual = (uap_verified_count - individual_verified_count)
            print('Reduction over Individual {:0.2f} \n'.format(diff_individual* args.count_per_prop/ 100.0))
            diff_ioformulation = (uap_verified_count - baseline_verified_count)
            print('Reduction over I/O Formulation {:0.2f} \n'.format(diff_ioformulation* args.count_per_prop/ 100.0))
            if args.enable_ablation:
                diff_ablation = (uap_verified_count - uap_verified_without_diff)
                print('Reduction over no difference constraints {:0.2f} \n'.format(diff_ablation * args.count_per_prop/ 100.0))



    def analyze(self, args):
        count = args.count
        individual_verified_count = 0
        baseline_verified_count = 0
        uap_verified_without_diff = 0
        uap_verified_count = 0
        times = [0, 0, 0, 0]
        layerwise_constraint_time = 0
        layerwise_optimization_time = 0
        diff_constraint_time = 0
        diff_optimization_time = 0
        if args.dataset == Dataset.CIFAR10:
            filename = args.output_dir + '{}_{}_{}_{:.2f}_{}.dat'.format(args.net_name, args.count_per_prop, args.count, args.eps*255, args.individual_prop_domain)
        else:
            filename = args.output_dir + f'{args.net_name}_{args.count_per_prop}_{args.count}_{args.eps}_{args.individual_prop_domain}.dat'
        file = open(filename, 'a+')
        for i, res in enumerate(self.result_list):
            individual_res = res.individual_res
            baseline_res = res.baseline_res
            uap_no_diff_res = res.result_with_no_diff
            raven_res = res.raven_res
            if individual_res is not None:
                veri = sum([torch.min(res.final_lb) >= 0 for res in individual_res])
                individual_verified_count += veri
                if res.individual_time is not None:
                    times[0] += res.individual_time
            if baseline_res is not None and baseline_res.verified_proportion is not None:
                baseline_verified_count += baseline_res.verified_proportion * args.count_per_prop
                if times[1] is not None and baseline_res.timings is not None:
                    times[1] += baseline_res.timings.total_time
            if uap_no_diff_res is not None and uap_no_diff_res.verified_proportion is not None:
                uap_verified_without_diff += uap_no_diff_res.verified_proportion * args.count_per_prop
                if times[2] is not None and uap_no_diff_res.timings is not None:
                    times[2] += uap_no_diff_res.timings.total_time
                    if uap_no_diff_res.timings.constraint_formulation_time is not None:
                        layerwise_constraint_time += uap_no_diff_res.timings.constraint_formulation_time
                    if uap_no_diff_res.timings.optimization_time is not None:
                        layerwise_optimization_time += uap_no_diff_res.timings.optimization_time
            if raven_res is not None and raven_res.verified_proportion is not None:
                uap_verified_count += raven_res.verified_proportion * args.count_per_prop
                if times[3] is not None and raven_res.timings is not None:
                    times[3] += raven_res.timings.total_time
                    if raven_res.timings.constraint_formulation_time is not None:
                        diff_constraint_time += raven_res.timings.constraint_formulation_time
                    if raven_res.timings.optimization_time is not None:
                        diff_optimization_time += raven_res.timings.optimization_time
        for i, _  in enumerate(times):
            if times[i] is not None and count > 0:
                times[i] /= count
        if count > 0:
            layerwise_constraint_time /= count
            layerwise_optimization_time /= count
            diff_constraint_time /= count
            diff_optimization_time /= count
    
        file.write(f'\n\n\nEps : {args.eps}\n')
        file.write(f'Individual verified: {individual_verified_count}\n')
        file.write(f'Baseline verified: {baseline_verified_count}\n')
        file.write(f'Uap no diff verified {uap_verified_without_diff if uap_no_diff_res is not None and uap_no_diff_res.verified_proportion is not None else "x"}\n')
        file.write(f'Uap verified: {uap_verified_count}\n')
        file.write(f'Extra verified: {uap_verified_count - baseline_verified_count}\n')
        file.write(f'Extra diff verified {uap_verified_count - uap_verified_without_diff}\n')
        file.write(f'Avg. times {times}\n')

        print("\n\n******************** Aggregated Result ********************\n\n")



        if args.spec_type == InputSpecType.UAP:
            count = count * args.count_per_prop
            print('Individual certified UAP accuracy: {:0.2f} %\n'.format(individual_verified_count/ count * 100))
            print('I/O Formulation certified UAP accuracy: {:0.2f} %\n'.format(baseline_verified_count/ count * 100))
            if args.enable_ablation:
                print('RaVeN no difference constraints certified UAP accuracy: {:0.2f} %\n'.format(uap_verified_without_diff/ count * 100))
            print('RaVeN certified UAP accuracy: {:0.2f}  %\n'.format(uap_verified_count/ count * 100))
            diff_individual = (uap_verified_count - individual_verified_count)
            print('Improvement over Individual {:0.2f} %\n'.format(diff_individual/ count * 100))
            diff_ioformulation = (uap_verified_count - baseline_verified_count)
            print('Improvement over I/O Formulation {:0.2f} %\n'.format(diff_ioformulation/ count * 100))
            if args.enable_ablation:
                diff_ablation = (uap_verified_count - uap_verified_without_diff)
                print('Improvement over no difference constraints {:0.2f} %\n'.format(diff_ablation/ count * 100))

        if args.spec_type == InputSpecType.UAP_BINARY:
            print('Individual worst-case hamming distance: {:0.2f} \n'.format(args.count_per_prop - individual_verified_count/ count))
            print('I/O Formulation worst-case hamming distance: {:0.2f} \n'.format(args.count_per_prop - baseline_verified_count/ count))
            if args.enable_ablation:
                print('RaVeN no difference constraints worst-case hamming distance: {:0.2f} \n'.format(args.count_per_prop - uap_verified_without_diff/ count))
            print('RaVeN worst-case hamming distance: {:0.2f}  \n'.format(args.count_per_prop - uap_verified_count/ count))
            diff_individual = (uap_verified_count - individual_verified_count)
            print('Reduction over Individual {:0.2f} \n'.format(diff_individual/ count))
            diff_ioformulation = (uap_verified_count - baseline_verified_count)
            print('Reduction over I/O Formulation {:0.2f} \n'.format(diff_ioformulation/ count))
            if args.enable_ablation:
                diff_ablation = (uap_verified_count - uap_verified_without_diff)
                print('Reduction over no difference constraints {:0.2f} \n'.format(diff_ablation/ count))

        print("\n\n******************** Aggregated Runtime ********************\n\n")

        print('Avg. Individual time: {:0.3f} sec.\n'.format(times[0]))
        print('Avg. I/O Formulation time: {:0.3f} sec.\n'.format(times[1]))
        if args.enable_ablation:
            print('Avg. RaVeN no difference time: {:0.3f} sec.\n'.format(times[2]))
        print('Avg. RaVeN time: {:0.3f} sec.\n'.format(times[3]))

        # Write the formulation and optimization times.
        file.write('\n\n\n')
        if uap_no_diff_res is not None and uap_no_diff_res.timings is not None:
            file.write(f'No diff constraint time {layerwise_constraint_time}\n')
            file.write(f'No diff optimization time {layerwise_optimization_time}\n')

        if raven_res is not None and raven_res.timings is not None:
            file.write(f'With diff constraint time {diff_constraint_time}\n')
            file.write(f'With diff optimization time {diff_optimization_time}\n')

        file.close()

    def analyze_monotone(self, args):
        diff_verified_count = 0
        # lp_verified_count = 0
        filename = args.output_dir + f'{args.net_name}_{args.count_per_prop}_{args.count}_{args.eps}.dat'
        file = open(filename, 'a+')
        times = 0
        num_split = args.monotone_splits
        cur_i = 0
        cur_status = True
        min_val = 10
        for i, res in enumerate(self.result_list):
            #file.write(f'\nProperty No. {i}\n\n')
            times += res.times
            raven_res = res.raven_res
            if raven_res.status != Status.VERIFIED:
                #diff_verified_count += 1
                cur_status = False
                min_val = raven_res.global_lb
                cur_i = num_split - 1
            cur_i += 1
            if cur_i == num_split:
                if cur_status:
                    diff_verified_count += 1
                cur_i = 0
                cur_status = True
                min_val = 10
        file.write(f'\n\n\nProp : {args.monotone_prop/args.count * 100.0}%\n')
        file.write(f'\n\n\nEps : {args.eps}\n')
        file.write(f'Diff verified: {diff_verified_count}\n')
        file.write(f'Time: {times}')
        # file.write(f'LP verified: {lp_verified_count}\n')
        print(f"******************** Feature {args.monotone_prop}, Eps {args.eps} ********************\n")
        print(f'Verified: {diff_verified_count/args.count * 100.0}%, Time: {times}')
        file.close() 

    # def analyze_monotone(self, args):
    #     diff_verified_count = 0
    #     # lp_verified_count = 0
    #     filename = args.output_dir + f'{args.net_name}_{args.count_per_prop}_{args.count}_{args.eps}.dat'
    #     file = open(filename, 'a+')
    #     times = 0
    #     for i, res in enumerate(self.result_list):
    #         #file.write(f'\nProperty No. {i}\n\n')
    #         raven_res = res.raven_res
    #         if raven_res.status == Status.VERIFIED:
    #             diff_verified_count += 1
    #         times += res.times
    #     file.write(f'\n\n\nProp : {args.monotone_prop}\n')
    #     file.write(f'\n\n\nEps : {args.eps}\n')
    #     file.write(f'Diff verified: {diff_verified_count}\n')
    #     file.write(f'Time: {times}')
    #     # file.write(f'LP verified: {lp_verified_count}\n')
    #     file.close()  
        
    def analyze_targeted(self, args):
        count = args.count
        counts = torch.zeros(10)
        individual_verified_count = torch.zeros(10)
        baseline_verified_count = torch.zeros(10)
        raven_verified_count = torch.zeros(10)
        diff_constraint_time = 0
        diff_optimization_time = 0
        times = [0, 0]
        diff = ''
        if args.track_differences is True:
            diff = '_diff'        
        filename = args.output_dir + f'target_{args.net_name}_{args.count_per_prop}_{args.count}_{args.eps}{diff}.dat'
        file = open(filename, 'a+')
        for i, res in enumerate(self.result_list):
            individual_res = res.individual_res
            baseline_res = res.baseline_res
            raven_res = res.raven_res
            file.write(f'\nProperty No. {i}\n\n')
            #print(individual_res[i].target_ubs)
            if individual_res is not None:
                #print("DeepZ lbs", [res.final_lb for res in self.baseline_results])
                deepz_res = [[] for i in range(10)]
                for i in range(len(individual_res)):
                    for j in range(10):
                        if res.props[i].out_constr.label == j:
                            continue
                        deepz_res[j].append((individual_res[i].target_ubs[j]).min() <= 0)
                veri = torch.tensor([(sum(res)).item() for res in deepz_res])
                individual_verified_count += veri
                file.write(f"Individual certified UAP accuracy: {[(veri[i]/len(deepz_res[i])).item()*100.0 for i in range(len(deepz_res))]}%\n")
            if raven_res.verified_proportion is not None:
                raven_verified_count += torch.tensor([raven_res.verified_proportion[i] * raven_res.bin_size[i] for i in range(len(raven_res.verified_proportion))])
                file.write(f"RaVeN certified UAP accuracy: {[a * 100.0 for a in raven_res.verified_proportion]}%\n")
                counts += torch.tensor(raven_res.bin_size)
                if raven_res.timings.constraint_formulation_time is not None:
                    diff_constraint_time += raven_res.timings.constraint_formulation_time
                if raven_res.timings.optimization_time is not None:
                    diff_optimization_time += raven_res.timings.optimization_time
            if res.times is not None:
                times[0] += res.times[0]
                times[1] += res.times[0] + res.times[1]
                file.write(f'Times: {res.times}\n')
        file.write(f'\n\n\nEps : {args.eps}\n')
        file.write(f'Individual verified: {individual_verified_count.tolist()} total: {sum(individual_verified_count).item()}\n')
        file.write(f'RaVeN verified: {raven_verified_count.tolist()} total: {sum(raven_verified_count).item()}\n')
        file.write(f'Extra verified: {(raven_verified_count - individual_verified_count).tolist()} total: {sum(raven_verified_count).item() - sum(baseline_verified_count).item()}\n')
        file.write(f'times: {times}\n')

        print("\n\n******************** Aggregated Result ********************\n\n")

        count = count * args.count_per_prop
        print('Individual certified UAP accuracy: {} %\n'.format(raven_verified_count/ counts * 100))
        print('RaVeN certified UAP accuracy: {}  %\n'.format(raven_verified_count/ counts * 100))
        diff_individual = (raven_verified_count - individual_verified_count)
        print('Improvement over Individual {} %\n'.format(diff_individual/ counts * 100))

        print("\n\n******************** Aggregated Runtime ********************\n\n")

        print('Avg. Individual time: {:0.3f} sec.\n'.format(times[0]))
        print('Avg. RaVeN time: {:0.3f} sec.\n'.format(times[1]))

        if raven_res is not None and raven_res.timings is not None:
            file.write(f'With diff constraint time {diff_constraint_time}\n')
            file.write(f'With diff optimization time {diff_optimization_time}\n')

        file.close()                    
