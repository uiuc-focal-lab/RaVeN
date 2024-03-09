import torch
from src.raven_results import RavenSingleRes
import gurobipy as grb
from src.common import Status



class IOFormulation:
    """
    Baseline LP formulation implementation for computing I/O Formulation results using either DeepZ or DeepPoly/CROWN. 
    The code is adapted from https://github.com/ruoxi-jia-group/Universal_Pert_Cert.
    """
    def __init__(self, net, props, args, baseline_results) -> None:
        self.net = net
        self.props = props 
        self.args = args
        self.baseline_results = baseline_results
        self.batch_size = len(self.baseline_results)
        self.input_size = self.baseline_results[0].input.shape[0]
        self.baseline_lbs = []
        # zonotope baseline
        self.zono_centers = None
        self.zono_coefs = None
        # crown baseline
        self.lb_coefs = None
        self.lb_biases = None
        # Store the unperturbed inputs.
        self.inputs = None 
        # Model
        self.model = grb.Model()
        # stores minimum of each individual property.
        self.prop_mins = []
        self.eps = None

    def populate_info(self):
        for res in self.baseline_results:
            self.baseline_lbs.append(torch.min(res.final_lb))
            if res.zono_center is not None:
                if self.zono_centers is None:
                    self.zono_centers = [res.zono_center]
                else:
                    self.zono_centers.append(res.zono_center)

            if res.zono_coef is not None:
                if self.zono_coefs is None:
                    self.zono_coefs = [res.zono_coef]
                else:
                    self.zono_coefs.append(res.zono_coef)

            if res.lb_coef is not None:
                if self.lb_coefs is None:
                    self.lb_coefs = [res.lb_coef]
                else:
                    self.lb_coefs.append(res.lb_coef)

            if res.lb_bias is not None:
                if self.lb_biases is None:
                    self.lb_biases = [res.lb_bias]
                else:
                    self.lb_biases.append(res.lb_bias)
            
            if res.input is not None:
                if self.inputs is None:
                    self.inputs = [res.input]
                else:
                    self.inputs.append(res.input)

            self.eps = res.eps

    def formulate_zono_lb(self):
        if self.zono_coefs is None or self.zono_centers is None:
            raise ValueError("coefs or center is NULL.")
        assert len(self.zono_coefs) == len(self.zono_centers)
        self.model.setParam('OutputFlag', False)
        actual_coefs = []
        lbs = []
        for i, coef in enumerate(self.zono_coefs):
            center = self.zono_centers[i]
            input_coefs = coef[:self.input_size]
            input_coefs = input_coefs.T
            other_coefs = coef[self.input_size:]
            cof_abs = torch.sum(torch.abs(other_coefs), dim=0)
            lb = center - cof_abs
            lbs.append(lb.detach().numpy())
            actual_coefs.append(input_coefs)
        
        epsilons = self.model.addMVar(self.input_size, lb=-1, ub=1, name='epsilons')
        individual_lbs = []
        
        for i, input_coefs in enumerate(actual_coefs):
            input_coefs = input_coefs.detach().numpy()
            t = self.model.addMVar(input_coefs.shape[0], lb=float('-inf'), ub=float('inf'), name=f'individual_lbs_{i}')
            self.model.addConstr(input_coefs @ epsilons + lbs[i] == t)
            individual_lbs.append(t)
            var_min = self.model.addVar(lb=-float('inf'), ub=float('inf'), 
                                                vtype=grb.GRB.CONTINUOUS, 
                                                name=f'var_min_{i}')
            self.model.addGenConstrMin(var_min, t.tolist())
            self.prop_mins.append(var_min)
            self.model.update()


    def optimize_lp(self, proportion):
        if proportion == False:
            problem_min = self.model.addVar(lb=-float('inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, 
                                name='problem_min')
            self.model.addGenConstrMax(problem_min, self.prop_mins)
            self.model.setObjective(problem_min, grb.GRB.MINIMIZE)
            self.model.optimize()
            if self.model.status == 2:
                return problem_min.X
            else:
                return -1e6
        else:
            binary_vars = []
            for i, var_min in enumerate(self.prop_mins):
                binary_vars.append(self.model.addVar(vtype=grb.GRB.BINARY, name=f'b{i}')) 
                # BIG M formulation 
                BIG_M = 1e11

                # Force binary_vars[-1] to be '1' when t_min > 0
                self.model.addConstr(BIG_M * binary_vars[-1] >= var_min)

                # Force binary_vars[-1] to be '0' when t_min < 0 or -t_min  > 0
                self.model.addConstr(BIG_M * (binary_vars[-1] - 1) <= var_min)
            p = self.model.addVar(vtype=grb.GRB.CONTINUOUS, name='p')
            self.model.addConstr(p == grb.quicksum(binary_vars[i] for i in range(self.batch_size)) / self.batch_size)
            # self.model.reset()
            self.model.update()
            self.model.setObjective(p, grb.GRB.MINIMIZE)
            self.model.optimize()
            if self.model.status == 2:
                return p.X
            else:
                return 0.0

    def run_zono_lp_baseline(self, proportion):
        self.baseline_lbs.sort()
        self.formulate_zono_lb()
        return self.optimize_lp(proportion=proportion)

    def pos_neg_weight_decomposition(self, coef):
        neg_comp = torch.where(coef < 0, coef, torch.zeros_like(coef, device='cpu'))
        pos_comp = torch.where(coef >= 0, coef, torch.zeros_like(coef, device='cpu'))
        return neg_comp, pos_comp
    
    def debug_lb_coef(self):
        for i, lb_coef in enumerate(self.lb_coefs):
            input_t = self.inputs[i]
            lb_bias = self.lb_biases[i]
            lb_layer = input_t - self.eps
            ub_layer = input_t + self.eps
            neg_comp_lb, pos_comp_lb = self.pos_neg_weight_decomposition(lb_coef)
            lb = neg_comp_lb @ ub_layer + pos_comp_lb @ lb_layer + lb_bias  


    
    def formulate_crown_lp(self):
        if self.lb_coefs is None or self.lb_biases is None or self.eps is None:
            raise ValueError("lb_coefs or lb_bias or eps is NULL.")
        assert len(self.lb_coefs) == len(self.lb_biases)
        self.model.setParam('OutputFlag', False)
        # self.debug_lb_coef()
        vs = [self.model.addMVar(self.input_size, lb = self.inputs[i].detach().numpy() - self.eps.item(),
                                  ub = self.inputs[i].detach().numpy() + self.eps.item(), vtype=grb.GRB.CONTINUOUS,
                                name=f'input_{i}') for i in range(len(self.inputs))]
        delta = self.model.addMVar(self.input_size, lb =-self.eps.item(), ub = self.eps.item(), vtype=grb.GRB.CONTINUOUS, name='uap_delta')
        for i, lb_coef in enumerate(self.lb_coefs):
            lb_bias = self.lb_biases[i]
            self.model.addConstr(vs[i] == self.inputs[i].detach().numpy() + delta)
            t = self.model.addMVar(lb_bias.shape[0], lb=float('-inf'), ub=float('inf'), name=f'individual_lbs_{i}')
            lb_bias = lb_bias.detach().numpy()
            lb_coef = lb_coef.detach().numpy()
            self.model.addConstr(t == lb_coef @ vs[i] + lb_bias)
            var_min = self.model.addVar(lb=-float('inf'), ub=float('inf'), 
                                                vtype=grb.GRB.CONTINUOUS, 
                                                name=f'var_min_{i}')
            self.model.addGenConstrMin(var_min, t.tolist())
            self.prop_mins.append(var_min)
            self.model.update()            

            
    def run_crown_lp_baseline(self, proportion):
        self.baseline_lbs.sort()
        self.formulate_crown_lp()    
        return self.optimize_lp(proportion=proportion)    
    


    # Calculates verification results using LP formulation and then invokes GUROBI for solving the LP.
    def run(self, proportion=False, targeted = False, monotone = False, monotonic_inv = False, diff=None) -> RavenSingleRes:
        self.populate_info()
        if self.zono_centers is not None:
            ans = self.run_zono_lp_baseline(proportion=proportion)
            if ans is None:
                verified_status = Status.UNKNOWN
                verified_proportion = sum([self.baseline_lbs[i] >= 0 for i in range(len(self.baseline_lbs))])/len(self.baseline_lbs)
                if verified_proportion >= self.args.cutoff_percentage:
                    verified_status = Status.VERIFIED
                return RavenSingleRes(domain=self.args.domain, input_per_prop=self.args.count_per_prop,
                    status=verified_status, global_lb=None, time_taken=None, 
                    verified_proportion=verified_proportion)
            else:
                global_lb = None
                verified_proportion = None
                verified_status = Status.UNKNOWN
                if proportion == False:
                    global_lb = ans
                    if global_lb >= 0:
                        verified_status = Status.VERIFIED
                else:
                    # print("Baseline Verified ", ans)
                    verified_proportion = ans
                    if verified_proportion >= self.args.cutoff_percentage:
                        verified_status = Status.VERIFIED

            return RavenSingleRes(domain=self.args.domain, input_per_prop=self.args.count_per_prop,
                    status=verified_status, global_lb=global_lb, time_taken=None, 
                    verified_proportion=verified_proportion)
        elif self.lb_coefs is not None:
            ans = self.run_crown_lp_baseline(proportion=proportion)
            global_lb = None
            verified_proportion = None
            verified_status = Status.UNKNOWN
            if proportion == False:
                global_lb = ans
                if global_lb >= 0:
                    verified_status = Status.VERIFIED
            else:
                # print("Baseline Verified  ", ans)
                verified_proportion = ans
                if verified_proportion >= self.args.cutoff_percentage:
                    verified_status = Status.VERIFIED

            return RavenSingleRes(domain=self.args.domain, input_per_prop=self.args.count_per_prop,
                status=verified_status, global_lb=global_lb, time_taken=None, 
                verified_proportion=verified_proportion)
