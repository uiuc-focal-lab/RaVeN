import gurobipy as grb
from gurobipy import GRB
import torch
from torch import nn
from src.common.network import LayerType
import numpy as np
import time
from src.specs.out_spec import create_out_targeted_uap_matrix


# Manual callback to terminate the gurobi model.
def softtime(model, where):
    if where == GRB.Callback.MIP:
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        if runtime > 200 and objbnd > 0.0:
            model.terminate()


class RaVeNMILPtransformer:
    """
    Implementation of RaVeN MILP formulation. The key steps are
    1. Creating the input constraints.
    2. Formulating the constraints at the intermediate layers and the difference constraints.
    3. Encode the output specification as MILP objective.
    4. Invoke the Gurobi to obtain the optimal solution. 
    """
    def __init__(self, networks, xs, eps, x_lbs, x_ubs, d_lbs, d_ubs, 
                 constraint_matrices, last_conv_diff_structs=None, debug_mode=False, 
                 track_differences=True, props=None, monotone = False, args=None):
        
        # Populte the networks, input lower bounds and constraint matrices for the outputs/
        self.networks = networks
        # The unperturbed inputs.
        self.xs = xs
        self.batch_size = len(xs)
        self.input_size = None
        self.shape = None
        if len(xs) > 0:
            self.input_size = xs[0].shape[0]
            self.set_shape()
        self.eps = eps
        self.x_lbs = x_lbs
        self.x_ubs = x_ubs
        self.d_lbs = d_lbs
        self.d_ubs = d_ubs
        self.props = props
        self.last_conv_diff_structs = last_conv_diff_structs
        self.linear_layer_idx = -1 

        # Gurobi model and variables.
        self.gmdl = grb.Model()
        self.gurobi_variables = []
        self.gurobi_var_dict = {}
        self.debug = True
        self.constraint_matrices = constraint_matrices
        self.tolerence = 1e-3
        self.debug_mode = False
        # Addition of difference constraints.
        self.track_differences = track_differences
        self.props = props
        self.monotone = monotone
        self.constraint_time = None
        self.optimize_time = None
        self.args = args
        self.g_constrs = []
        self.par_constraints = False
        self.tanh_binary_count = 0
        self.bint = 1e15
        self.lightweight_difference = self.args.lightweight_diffpoly
        if self.lightweight_difference and self.args.fold_conv_layers:
            self.track_differences = False


    def set_shape(self):
        if self.input_size == 784:
            self.shape = (1, 28, 28)
        elif self.input_size == 3072:
            self.shape = (3, 32, 32)
        # For debug input.
        elif self.input_size == 2:
            self.shape = (1, 1, 2)
        elif self.input_size == 87:
            self.shape = (1, 1, 87)
        else:
            raise ValueError("Unsupported dataset!")


        
    # Optimization for monotonicity.
    def optimize_monotone(self, monotone):
        assert len(self.constraint_matrices) == self.batch_size
        if self.batch_size <= 0:
            return 0.0
        self.gmdl.update()
        self.gmdl.setObjective(self.gurobi_variables[-1]['ds'][0][0], grb.GRB.MINIMIZE)
        
        self.constraint_time += time.time()
        self.optimize_time = - time.time()        
        self.gmdl.optimize(softtime)
        self.optimize_time += time.time()
         
        if self.debug_mode is True:
            print("Here")
            self.gmdl.write("./debug_logs/model.lp")
            # self.gmdl.write("./debug_logs/out.sol")
        
        if self.gmdl.status in [2, 6, 10]:
            #self.debug_log_file.write(f"proportion {p.X}\n")
            # print(f"verified proportion {p.X}\n")
            self.debug_log_file.close()
            # print("Final MIP gap value: %f" % self.gmdl.MIPGap)
            # print("Final ObjBound: %f" % self.gmdl.ObjBound)
            return self.gmdl.ObjBound
        else:
            if self.gmdl.status == 4:
                return 0.0
            elif self.gmdl.status in [9, 11, 13]:
                print("Suboptimal solution")
                self.debug_log_file.close()
                print("Final MIP gap value: %f" % self.gmdl.MIPGap)
                try:
                    print("Final MIP best value: %f" % p.X)
                except:
                    print("No solution obtained")
                print("Final ObjBound: %f" % self.gmdl.ObjBound)
                if self.gmdl.SolCount > 0:
                    return self.gmdl.ObjBound
                else:
                    return 0.0
            self.debug_log_file.close()    
            print("Gurobi model status", self.gmdl.status)
            print("The optimization failed\n")
            # print("Computing computeIIS")
            # self.gmdl.computeIIS()
            # print("Computing computeIIS finished")            
            # self.gmdl.write("model.ilp")
            self.debug_log_file.close()
            return 0.0

    # MILP optimization for targeted UAP.
    def optimize_targeted(self):
        percentages = []
        bin_sizes = []
        if self.batch_size <= 0:
            return 0.0
        for j in range(10):
            bs = []
            final_vars = []
            final_min_vars = []
            constraint_mat = create_out_targeted_uap_matrix(torch.tensor(j))
            for i, _ in enumerate(self.constraint_matrices):
                if self.props[i].out_constr.label == j:
                    continue
                final_var = self.gmdl.addMVar(constraint_mat.shape[1], lb=-float('inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, 
                                                name=f'final_var_{i}')
                self.gmdl.addConstr(final_var == constraint_mat.T.detach().numpy() @ self.gurobi_variables[-1]['vs'][i])
                final_vars.append(final_var)
                final_var_max = self.gmdl.addVar(lb=-float('inf'), ub=float('inf'), 
                                                    vtype=grb.GRB.CONTINUOUS, 
                                                    name=f'final_var_min_{i}')
                self.gmdl.addGenConstrMax(final_var_max, final_var.tolist())
                final_min_vars.append(final_var_max)
                bs.append(self.gmdl.addVar(vtype=grb.GRB.BINARY, name=f'b{i}'))

                # Binary encoding (Big M formulation )
                BIG_M = 1e11

                # Force bs[-1] to be '1' when t_min > 0
                self.gmdl.addConstr(BIG_M * bs[-1] >= final_var_max)

                # Force bs[-1] to be '0' when t_min < 0 or -t_min  > 0
                self.gmdl.addConstr(BIG_M * (bs[-1] - 1) <= final_var_max)
            
            p = self.gmdl.addVar(vtype=grb.GRB.CONTINUOUS, name=f'p')
            self.gmdl.addConstr(p == grb.quicksum(bs[i] for i in range(len(bs))) / len(bs))
            self.gmdl.update()
            self.gmdl.setObjective(p, grb.GRB.MINIMIZE)
            
            self.constraint_time += time.time()
            self.optimize_time = - time.time()
            self.gmdl.optimize()
            self.optimize_time += time.time()
            self.constraint_time -= time.time()
            
            
            if self.gmdl.status == 2:
                percentages.append(p.X)
                bin_sizes.append(len(bs))
            else:
                if self.gmdl.status == 4:
                    self.gmdl.setParam('PreDual',0)
                    self.gmdl.setParam('DualReductions', 0)
                    self.gmdl.optimize()
                elif self.gmdl.status == 13 or self.gmdl.status == 9:
                    if self.gmdl.SolCount > 0:
                        percentages.append(p.X)
                        bin_sizes.append(len(bs))
                    else:
                        percentages.append(0.0)
                        bin_sizes.append(len(bs))
        return percentages, bin_sizes


    # MILP optimization for untargeted UAP and for worst-case hamming distance.
    def optimize_milp_percent(self):
        assert len(self.constraint_matrices) == self.batch_size
        if self.batch_size <= 0:
            return 0.0
        bs = []
        final_vars = []
        final_min_vars = []
        for i, constraint_mat in enumerate(self.constraint_matrices):
            final_var = self.gmdl.addMVar(constraint_mat.shape[1], lb=-float('inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, 
                                            name=f'final_var_{i}')
            self.gmdl.addConstr(final_var == constraint_mat.T.detach().numpy() @ self.gurobi_variables[-1]['vs'][i])
            final_vars.append(final_var)
            final_var_min = self.gmdl.addVar(lb=-float('inf'), ub=float('inf'), 
                                                vtype=grb.GRB.CONTINUOUS, 
                                                name=f'final_var_min_{i}')
            self.gmdl.addGenConstrMin(final_var_min, final_var.tolist())
            final_min_vars.append(final_var_min)
            bs.append(self.gmdl.addVar(vtype=grb.GRB.BINARY, name=f'b{i}'))

            # Binary encoding (Big M formulation )
            BIG_M = 1e11

            # Force bs[-1] to be '1' when t_min > 0
            self.gmdl.addConstr(BIG_M * bs[-1] >= final_var_min)

            # Force bs[-1] to be '0' when t_min < 0 or -t_min  > 0
            self.gmdl.addConstr(BIG_M * (bs[-1] - 1) <= final_var_min)
        
        p = self.gmdl.addVar(vtype=grb.GRB.CONTINUOUS, name=f'p')
        self.gmdl.addConstr(p == grb.quicksum(bs[i] for i in range(self.batch_size)) / self.batch_size)
        self.gmdl.update()
        self.gmdl.setObjective(p, grb.GRB.MINIMIZE)
        
        self.constraint_time += time.time()
        self.optimize_time = - time.time()        
        self.gmdl.optimize(softtime)
        self.optimize_time += time.time()
         
        
        if self.gmdl.status in [2, 6, 10]:
            return self.gmdl.ObjBound
        else:
            if self.gmdl.status == 4:
                return 0.0
            elif self.gmdl.status in [9, 11, 13]:

                if self.gmdl.SolCount > 0:
                    return self.gmdl.ObjBound
                else:
                    return 0.0    
            return 0.0

    def create_input_constraints(self):
        # The uap perturbation.
        if len(self.xs) <= 0:
            return
        delta = self.gmdl.addMVar(self.xs[0].shape[0], lb = -self.eps, ub = self.eps, vtype=grb.GRB.CONTINUOUS, name='uap_delta')
        vs = [self.gmdl.addMVar(self.xs[i].shape[0], lb = self.xs[i].detach().numpy() - self.eps, ub = self.xs[i].detach().numpy() + self.eps, vtype=grb.GRB.CONTINUOUS, name=f'input_{i}') for i in range(self.batch_size)]
        # Ensure all inputs are perturbed by the same uap delta.
        for i, v in enumerate(vs):
            self.gmdl.addConstr(v == self.xs[i].detach().numpy() + delta)              
        
        self.gurobi_variables.append({'delta': delta, 'vs': vs, 'ds': None})

    
    # Create constraints for each layer.
    # We currently support the following layers
    # 1. Affine layers
    # 2. Convolutional layers
    # 3. Activation layers
    #   3a. ReLU layers
    #   3b. Sigmoid layers
    #   3c. Tanh layers    
    def create_constraints(self):
        self.create_input_constraints()
        layers = self.networks
        
        for layer_idx, layer in enumerate(layers):
            layer_type = self.get_layer_type(layer)
            if layer_type == LayerType.Linear:
                self.linear_layer_idx += 1
                self.create_linear_constraints(layer, layer_idx)
            elif layer_type == LayerType.ReLU:
                self.create_relu_constraints(layer_idx)
                if self.args is not None and self.args.all_layer_sub is True:
                    self.linear_layer_idx += 1
            elif layer_type == LayerType.Conv2D:
                self.linear_layer_idx += 1                
                self.create_conv2d_constraints(layer, layer_idx)
            elif layer_type == LayerType.Sigmoid:
                self.create_sigmoid_constraints(layer_idx=layer_idx)
                if self.args is not None and self.args.all_layer_sub is True:
                    self.linear_layer_idx += 1
            elif layer_type == LayerType.TanH:
                self.create_tanh_constraints(layer_idx=layer_idx)
                if self.args is not None and self.args.all_layer_sub is True:
                    self.linear_layer_idx += 1
            elif layer_type == LayerType.Flatten:
                continue
            else:
                raise TypeError(f"Unsupported Layer Type '{layer_type}'")
    
    def create_parallel_constraints(self):
        t = -time.time()
        self.create_variables()

        t = -time.time()
        self.parallel_generate_constraints()
        t = -time.time()
        self.create_gurobi_model()

    def create_variables(self):
        # Input constraint variables.
        delta = self.gmdl.addMVar(self.xs[0].shape[0], lb = -self.eps, ub = self.eps, vtype=grb.GRB.CONTINUOUS, name='uap_delta')
        vs = [self.gmdl.addMVar(self.xs[i].shape[0], lb = self.xs[i].detach().numpy() - self.eps, ub = self.xs[i].detach().numpy() + self.eps, vtype=grb.GRB.CONTINUOUS, name=f'input_{i}') for i in range(self.batch_size)]
        self.gurobi_var_dict[-1] = {'delta': delta, 'vs': vs, 'ds': None}

        # Layer constraint variables.
        layers = self.networks
        for layer_idx, layer in enumerate(layers):
            layer_type = self.get_layer_type(layer)
            if layer_type == LayerType.Linear:
                vs, ds = self.create_vars(layer_idx, 'linear')
            elif layer_type == LayerType.ReLU:
                vs, ds = self.create_vars(layer_idx, 'relu')
            elif layer_type == LayerType.Conv2D:            
                vs, ds = self.create_vars(layer_idx, 'conv2d')           
            elif layer_type in [LayerType.Sigmoid, LayerType.TanH]:            
                vs, ds = self.create_vars(layer_idx, 'sigmoid')            
            elif layer_type == LayerType.Flatten:
                continue
            else:
                raise TypeError(f"Unsupported Layer Type '{layer_type}'")
            self.gurobi_var_dict[layer_idx] = {'vs': vs, 'ds': ds}

    # Parallelly adding constraints to Gurobi for reducing runtime.
    def parallel_generate_constraints(self):
        self.parallel_gen_input_constraints()
        layers = self.networks
        for layer_idx, layer in enumerate(layers):
            layer_type = self.get_layer_type(layer)
            if layer_type == LayerType.Linear:
                self.linear_layer_idx += 1
                self.parallel_gen_linear_constraints(layer, layer_idx)
                #lin_constrs+= len(self.g_constrs) - pre_num_constrs
            elif layer_type == LayerType.ReLU:
                self.parallel_gen_relu_constraints(layer, layer_idx)
                #relu_constrs+= len(self.g_constrs) - pre_num_constrs
                if self.args is not None and self.args.all_layer_sub is True:
                    self.linear_layer_idx += 1
            elif layer_type == LayerType.Conv2D:
                self.linear_layer_idx += 1
                self.parallel_gen_conv2d_constraints(layer, layer_idx)
            elif layer_type == LayerType.Flatten:
                continue
            else:
                raise TypeError(f"Unsupported Layer Type '{layer_type}'")
    
    def parallel_gen_input_constraints(self):
        for i, v in enumerate(self.gurobi_var_dict[-1]['vs']):
            self.g_constrs.append((v, self.xs[i].detach().numpy() + self.gurobi_var_dict[-1]['delta'], '='))


    def parallel_gen_linear_constraints(self, layer, layer_idx):
            weight, bias = layer.weight.detach().numpy(), layer.bias.detach().numpy()
            vs, ds = self.gurobi_var_dict[layer_idx]['vs'], self.gurobi_var_dict[layer_idx]['ds']
            vs1 = self.gurobi_var_dict[layer_idx-1]['vs']
            for v_idx, v in enumerate(vs):
                self.g_constrs.append((v, weight @ vs1[v_idx] + bias, '='))
            if self.track_differences is True:
                for i in range(self.batch_size):
                    for j in range(i+1, self.batch_size):
                        self.g_constrs.append((ds[i][j - i - 1], vs[i] - vs[j], '='))

    def parallel_gen_relu_constraints(self, layer, layer_idx):
        vs, ds = self.gurobi_var_dict[layer_idx]['vs'], self.gurobi_var_dict[layer_idx]['ds']
        vs1, ds1 = self.gurobi_var_dict[layer_idx-1]['vs'], self.gurobi_var_dict[layer_idx-1]['ds']

        for i in range(self.batch_size):
            self.g_constrs.append((vs[i], 0, '>='))
            self.g_constrs.append((vs[i], vs1[i], '>='))
            tensor_length = self.x_lbs[i][self.linear_layer_idx].shape[0] 
            up_scales, up_biases = np.zeros(tensor_length), self.bint * np.ones(tensor_length)
            for j in range(tensor_length):               
                if self.x_lbs[i][self.linear_layer_idx][j] >= 0:
                    up_scales[j], up_biases[j] = 1, 0
                    continue 
                elif self.x_ubs[i][self.linear_layer_idx][j] <= 0:
                    up_scales[j], up_biases[j] = 0, 0
                    continue
                else:
                    ub = self.x_ubs[i][self.linear_layer_idx][j]
                    lb = self.x_lbs[i][self.linear_layer_idx][j]
                    slope = ub / (ub - lb + 1e-15)
                    mu = -slope * lb
                    up_scales[j], up_biases[j] = slope, mu
            self.g_constrs.append((vs[i], up_scales * vs1[i] + up_biases, '<='))
        
        if self.track_differences is True:
            for i in range(self.batch_size):
                for j in range(i+1, self.batch_size):
                    self.g_constrs.append((ds[i][j - i - 1], vs[i] - vs[j], '='))

            for i in range(self.batch_size):
                for j in range(i+1, self.batch_size):
                    tensor_length = self.x_lbs[i][self.linear_layer_idx].shape[0]
                    up_scale, up_bias, down_scale, down_bias = np.zeros(tensor_length), self.bint * np.ones(tensor_length), np.zeros(tensor_length), -self.bint * np.ones(tensor_length)
                    for k in range(tensor_length):
                        # case 1 x unsettled & y passive
                        x_active = (self.x_lbs[i][self.linear_layer_idx][k] >= 0)
                        x_passive = (self.x_ubs[i][self.linear_layer_idx][k] <= 0)
                        x_unsettled = (~x_active) & (~x_passive)
                        y_active = (self.x_lbs[j][self.linear_layer_idx][k] >= 0)
                        y_passive = (self.x_ubs[j][self.linear_layer_idx][k] <= 0)
                        y_unsettled = (~y_active) & (~y_passive)
                        delta_active = (self.d_lbs[(i, j)][self.linear_layer_idx][k] >= 0)
                        delta_passive = (self.d_ubs[(i, j)][self.linear_layer_idx][k] <= 0)
                        delta_unsettled = (~delta_active) & (~delta_passive)                                                                                
                        if x_unsettled and y_passive and delta_active:
                            up_scale[k], up_bias[k], down_scale[k], down_bias[k] = 1, 0, 0, -self.bint
                        elif x_unsettled and y_active:
                            up_scale[k], up_bias[k], down_scale[k], down_bias[k] = 0, self.bint, 1, 0                     
                        elif x_passive and y_unsettled and delta_passive:
                            up_scale[k], up_bias[k], down_scale[k], down_bias[k] = 0, self.bint, 1, 0
                        elif x_active and y_unsettled:
                            up_scale[k], up_bias[k], down_scale[k], down_bias[k] = 1, 0, 0, -self.bint
                        elif x_unsettled and y_unsettled and delta_active:
                            up_scale[k], up_bias[k], down_scale[k], down_bias[k] = 1, 0, 0, 0
                        elif x_unsettled and y_unsettled and delta_passive:
                            up_scale[k], up_bias[k], down_scale[k], down_bias[k] = 0, 0, 1, 0
                        elif x_unsettled and y_unsettled and delta_unsettled:
                            temp_lb = self.d_lbs[(i, j)][self.linear_layer_idx][k]
                            temp_ub = self.d_ubs[(i, j)][self.linear_layer_idx][k]
                            d_lambda_lb = temp_lb / (temp_lb - temp_ub + 1e-15)
                            d_lambda_ub = temp_ub / (temp_ub - temp_lb + 1e-15)
                            d_mu_lb = d_lambda_ub * temp_lb
                            d_mu_ub = -d_lambda_ub * temp_lb
                            up_scale[k], up_bias[k], down_scale[k], down_bias[k] = d_lambda_ub, d_mu_ub, d_lambda_lb, d_mu_lb
                    self.g_constrs.append((ds[i][j - i -1], up_scale * ds1[i][j-i-1] + up_bias, '<='))
                    self.g_constrs.append((ds[i][j - i -1], down_scale * ds1[i][j-i-1] + down_bias, '>='))
            if self.lightweight_difference is True:
                self.track_differences = False


    def parallel_gen_conv2d_constraints_helper(self, vars, pre_vars, num_kernel, output_h, 
                                         output_w, bias, weight, layer, input_h, input_w):
        out_idx = 0
        gvars_array = [np.array(pre_var.tolist()).reshape((-1, input_h, input_w)) for pre_var in pre_vars]
        pre_lb_size = [None, None, input_h, input_w]
        gmLinExprs = [0 * v for i, v in enumerate(vars)]
        for out_chan_idx in range(num_kernel):
            for out_row_idx in range(output_h):
                for out_col_idx in range(output_w):
                    lin_expressions = [grb.LinExpr() for i in range(len(pre_vars))]

                    for in_chan_idx in range(layer.weight.shape[1]):

                        # New version of conv layer for building mip by skipping kernel loops.
                        ker_row_min, ker_row_max = 0, layer.weight.shape[2]
                        in_row_idx_min = -layer.padding[0] + layer.stride[0] * out_row_idx
                        in_row_idx_max = -layer.padding[0] + layer.stride[0] * out_row_idx + layer.weight.shape[2] - 1
                        if in_row_idx_min < 0: 
                            ker_row_min = 0 - in_row_idx_min
                        if in_row_idx_max >= pre_lb_size[2]: 
                            ker_row_max = ker_row_max - in_row_idx_max + pre_lb_size[2] -1
                        in_row_idx_min, in_row_idx_max = max(in_row_idx_min, 0), min(in_row_idx_max, pre_lb_size[2] - 1)

                        ker_col_min, ker_col_max = 0, layer.weight.shape[3]
                        in_col_idx_min = -layer.padding[1] + layer.stride[1] * out_col_idx
                        in_col_idx_max = -layer.padding[1] + layer.stride[1] * out_col_idx + layer.weight.shape[3] - 1
                        if in_col_idx_min < 0: ker_col_min = 0 - in_col_idx_min
                        if in_col_idx_max >= pre_lb_size[3]: 
                            ker_col_max = ker_col_max - in_col_idx_max + pre_lb_size[3] -1
                        in_col_idx_min, in_col_idx_max = max(in_col_idx_min, 0), min(in_col_idx_max, pre_lb_size[3] - 1)

                        coeffs = layer.weight[out_chan_idx, in_chan_idx, ker_row_min:ker_row_max, ker_col_min:ker_col_max].reshape(-1)
                        for i, gvars in enumerate(gvars_array):
                            gvar = gvars[in_chan_idx, in_row_idx_min:in_row_idx_max+1, in_col_idx_min:in_col_idx_max+1].reshape(-1)
                            lin_expressions[i] += grb.LinExpr(coeffs, gvar)
                    for i, var in enumerate(vars):
                        gmLinExprs[i][out_idx] = lin_expressions[i] + bias[out_chan_idx].item()
                        #self.g_constrs.append((var[out_idx], lin_expressions[i] + bias[out_chan_idx].item(), '='))
                    out_idx += 1
        for i, var in enumerate(vars):
            self.g_constrs.append((var, gmLinExprs[i], '='))

    def parallel_gen_conv2d_constraints(self, layer, layer_idx):
        vs, ds = self.gurobi_var_dict[layer_idx]['vs'], self.gurobi_var_dict[layer_idx]['ds']
        vs1, ds1 = self.gurobi_var_dict[layer_idx-1]['vs'], self.gurobi_var_dict[layer_idx-1]['ds']
        weight = layer.weight
        bias = layer.bias
        assert layer.dilation == (1, 1)

        # Ref. https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d 
        k_h, k_w = layer.kernel_size
        s_h, s_w = layer.stride
        p_h, p_w = layer.padding
        num_kernel = weight.shape[0]
        input_h, input_w = self.shape[1:]
        output_h = int((input_h + 2 * p_h - k_h) / s_h + 1)
        output_w = int((input_w + 2 * p_w - k_w) / s_w + 1)

        # Updated shape
        self.shape = (num_kernel, output_h, output_w)
        self.parallel_gen_conv2d_constraints_helper(vars=vs, pre_vars=vs1,
                                                   num_kernel=num_kernel, output_h=output_h, output_w=output_w,
                                                  bias=bias, weight=weight, layer=layer, input_h=input_h, input_w=input_w)

        if self.track_differences is True:
            for i in range(self.batch_size):
                for j in range(i+1, self.batch_size):
                    self.g_constrs.append((ds[i][j - i - 1], vs[i] - vs[j], '='))

    def create_gurobi_model(self):
        for i in range(len(self.g_constrs)):
            lhs, rhs, sense = self.g_constrs[i]
            if sense == '=':
                if type(lhs) is grb.MVar:
                    self.gmdl.addConstr(lhs == rhs)
                else:
                    self.gmdl.addLConstr(lhs == rhs)
            elif sense == '>=':
                if type(lhs) is grb.MVar:
                    self.gmdl.addConstr(lhs >= rhs)
                else:
                    self.gmdl.addLConstr(lhs >= rhs)
            elif sense == '<=':
                if type(lhs) is grb.MVar:
                    self.gmdl.addConstr(lhs <= rhs)
                else:
                    self.gmdl.addLConstr(lhs <= rhs)
            else:
                raise TypeError(f"Unsupported Sense Type '{sense}'")


    def add_coefs_lb_ub(self, layer_idx):
        if self.last_conv_diff_structs is None:
            raise ValueError(f'Diff sturcts is None')
        
        if len(self.last_conv_diff_structs) != self.batch_size:
            raise ValueError(f'Diff sturcts length: {len(self.last_conv_diff_structs)} != batch size: {self.batch_size}')

        vs, ds = self.create_vars(layer_idx, 'linear')

        for i, v in enumerate(vs):
            diff_struct = self.last_conv_diff_structs[i]
            lb_coef = diff_struct.lb_coef.detach().numpy()
            lb_bias = diff_struct.lb_bias.detach().numpy()
            ub_coef = diff_struct.ub_coef.detach().numpy()
            ub_bias = diff_struct.ub_bias.detach().numpy()
            self.gmdl.addConstr(v >= lb_coef @ self.gurobi_variables[-1]['vs'][i] + lb_bias)     
            self.gmdl.addConstr(v <= ub_coef @ self.gurobi_variables[-1]['vs'][i] + ub_bias)

        if self.track_differences is True:
            for i in range(self.batch_size):
                for j in range(i+1, self.batch_size):
                    self.gmdl.addConstr(ds[i][j - i - 1] == vs[i] - vs[j])
        
        self.gurobi_variables.append({'vs': vs, 'ds': ds})
    
    def create_constraints_folded_conv_layers(self):
        self.create_input_constraints()
        layers = self.networks
        # Prune conv layers
        layer_idx = 0
        for layer in layers:
            layer_type = self.get_layer_type(layer)
            if layer_type == LayerType.Linear:
                break
            elif layer_type in [LayerType.ReLU, LayerType.Sigmoid, LayerType.TanH]:
                if self.args is not None and self.args.all_layer_sub is True:
                    self.linear_layer_idx += 1
            elif layer_type == LayerType.Conv2D:
                self.linear_layer_idx += 1                                
            elif layer_type == LayerType.Flatten:
                continue
            else:
                raise TypeError(f"Unsupported Layer Type '{layer_type}'")
            layer_idx += 1

        # Go back one idx 
        layer_idx -= 2
        if self.args is not None and self.args.all_layer_sub is True:
            self.linear_layer_idx -= 1        

        self.add_coefs_lb_ub(layer_idx=layer_idx)
        layer_idx += 1

        for i in range(layer_idx, len(layers)):
            layer = layers[i]
            layer_type = self.get_layer_type(layer)
            if layer_type == LayerType.Linear:
                self.linear_layer_idx += 1
                self.create_linear_constraints(layer, i)
            elif layer_type == LayerType.ReLU:
                self.create_relu_constraints(i)
                if self.args is not None and self.args.all_layer_sub is True:
                    self.linear_layer_idx += 1
            elif layer_type == LayerType.Conv2D:
                self.linear_layer_idx += 1                
                self.create_conv2d_constraints(layer, i)
            elif layer_type == LayerType.Sigmoid:
                self.create_sigmoid_constraints(layer_idx=i)
                if self.args is not None and self.args.all_layer_sub is True:
                    self.linear_layer_idx += 1
            elif layer_type == LayerType.TanH:
                self.create_tanh_constraints(layer_idx=i)
                if self.args is not None and self.args.all_layer_sub is True:
                    self.linear_layer_idx += 1                
            elif layer_type == LayerType.Flatten:
                continue
            else:
                raise TypeError(f"Unsupported Layer Type '{layer_type}'")



    def create_vars(self, layer_idx, layer_type=''):
        ds = None
        if layer_type in ['linear', 'conv2d']:            
            vs = [self.gmdl.addMVar(self.x_lbs[i][layer_idx].shape[0], lb = self.x_lbs[i][layer_idx], ub = self.x_ubs[i][layer_idx], vtype=grb.GRB.CONTINUOUS, name=f'layer_{layer_idx}_{layer_type}_x{i}') for i in range(self.batch_size)]
            if self.track_differences is True:
                ds = [[self.gmdl.addMVar(self.d_lbs[(i, j)][layer_idx].shape[0], 
                                         lb=self.d_lbs[(i, j)][layer_idx] - self.tolerence, 
                                         ub=self.d_ubs[(i, j)][layer_idx] + self.tolerence, 
                                         vtype=grb.GRB.CONTINUOUS, name=f'layer{layer_idx}_d({i}-{j})') 
                                         for j in range(i+1, self.batch_size)] 
                                         for i in range(self.batch_size)]
        elif layer_type == 'relu':
            vs = [self.gmdl.addMVar(self.x_lbs[i][layer_idx].shape[0], lb=self.x_lbs[i][layer_idx],
                                     ub = self.x_ubs[i][layer_idx], vtype=grb.GRB.CONTINUOUS, 
                                     name=f'layer_{layer_idx}_{layer_type}_x{i}') for i in range(self.batch_size)]
            if self.track_differences is True:
                if self.args is not None and self.args.all_layer_sub is True:
                    if self.batch_size > 1 and layer_idx < len(self.d_lbs[(0, 1)]):
                        ds = [[self.gmdl.addMVar(self.d_lbs[(i, j)][layer_idx].shape[0], lb=self.d_lbs[(i, j)][layer_idx] -self.tolerence,
                                ub=self.d_ubs[(i, j)][layer_idx] + self.tolerence,
                                vtype=grb.GRB.CONTINUOUS, name=f'layer{layer_idx}_d({i}-{j})') for j in range(i+1, self.batch_size)] 
                                for i in range(self.batch_size)]
                    else:
                        ds = [[self.gmdl.addMVar(self.d_lbs[(i, j)][layer_idx-1].shape[0], lb=self.x_lbs[i][layer_idx] - self.x_ubs[i][layer_idx] -self.tolerence,
                                ub=self.x_ubs[i][layer_idx] - self.x_lbs[i][layer_idx] + self.tolerence,
                                vtype=grb.GRB.CONTINUOUS, name=f'layer{layer_idx}_d({i}-{j})') for j in range(i+1, self.batch_size)] 
                                for i in range(self.batch_size)]

                else:     
                    ds = [[self.gmdl.addMVar(self.d_lbs[(i, j)][layer_idx].shape[0], lb=self.d_lbs[(i, j)][layer_idx],
                                     ub=self.d_ubs[(i, j)][layer_idx],
                                      vtype=grb.GRB.CONTINUOUS, name=f'layer{layer_idx}_d({i}-{j})') for j in range(i+1, self.batch_size)] for i in range(self.batch_size)]
            if self.lightweight_difference is True:
                self.track_differences = False
        elif layer_type in ['sigmoid', 'tanh']:
            vs = [self.gmdl.addMVar(self.x_lbs[i][layer_idx].shape[0], lb = self.x_lbs[i][layer_idx],
                                     ub = self.x_ubs[i][layer_idx], vtype=grb.GRB.CONTINUOUS, 
                                     name=f'layer_{layer_idx}_{layer_type}_x{i}') for i in range(self.batch_size)]
            if self.track_differences is True:
                ds = [[self.gmdl.addMVar(self.d_lbs[(i, j)][layer_idx].shape[0], 
                                        lb=self.d_lbs[(i, j)][layer_idx] - self.tolerence,
                                        ub=self.d_ubs[(i, j)][layer_idx] + self.tolerence,
                                        vtype=grb.GRB.CONTINUOUS, name=f'layer{layer_idx}_d({i}-{j})') 
                                        for j in range(i+1, self.batch_size)] for i in range(self.batch_size)]        
        else:
            raise ValueError(f'layer type {layer_type} is supported yet')

        return vs, ds
    
    def create_linear_constraints(self, layer, layer_idx):
        weight, bias = layer.weight, layer.bias
       
        weight = weight.detach().numpy()
        bias = bias.detach().numpy()
        vs, ds = self.create_vars(layer_idx, 'linear')
        
        for v_idx, v in enumerate(vs):
            self.gmdl.addConstr(v == weight @ self.gurobi_variables[-1]['vs'][v_idx] + bias)

        if self.track_differences is True:
            for i in range(self.batch_size):
                for j in range(i+1, self.batch_size):
                    self.gmdl.addConstr(ds[i][j - i - 1] == vs[i] - vs[j])

        self.gurobi_variables.append({'vs': vs, 'ds': ds})


    def create_relu_ub(self, x, lb, ub): #probably breaks on degenerate lb=ub case, should fix
        rlb, rub = np.max(0, lb), np.max(0, ub)
        return (rub-rlb)/(ub-lb) * (x-lb) + rlb

    def create_relu_constraints(self, layer_idx):
        vs, ds = self.create_vars(layer_idx, 'relu')

        for i in range(self.batch_size):
            self.gmdl.addConstr(vs[i] >= 0)
            self.gmdl.addConstr(vs[i] >= self.gurobi_variables[-1]['vs'][i])
            tensor_length = self.x_lbs[i][self.linear_layer_idx].shape[0] 
            for j in range(tensor_length):               
                if self.x_lbs[i][self.linear_layer_idx][j] >= 0:
                    self.gmdl.addConstr(vs[i][j] <= self.gurobi_variables[-1]['vs'][i][j])
                    continue 
                elif self.x_ubs[i][self.linear_layer_idx][j] <= 0:
                    self.gmdl.addConstr(vs[i][j] <= 0)
                    continue
                else:
                    ub = self.x_ubs[i][self.linear_layer_idx][j]
                    lb = self.x_lbs[i][self.linear_layer_idx][j]
                    slope = ub / (ub - lb + 1e-15)
                    mu = -slope * lb
                    self.gmdl.addConstr(vs[i][j] <= slope * self.gurobi_variables[-1]['vs'][i][j] + mu)
        
        if self.track_differences is True:
            for i in range(self.batch_size):
                for j in range(i+1, self.batch_size):
                    self.gmdl.addConstr(ds[i][j - i - 1] == vs[i] - vs[j])

            for i in range(self.batch_size):
                for j in range(i+1, self.batch_size):
                    tensor_length = self.x_lbs[i][self.linear_layer_idx].shape[0]
                    for k in range(tensor_length):
                        x_active = (self.x_lbs[i][self.linear_layer_idx][k] >= 0)
                        x_passive = (self.x_ubs[i][self.linear_layer_idx][k] <= 0)
                        x_unsettled = (~x_active) & (~x_passive)
                        y_active = (self.x_lbs[j][self.linear_layer_idx][k] >= 0)
                        y_passive = (self.x_ubs[j][self.linear_layer_idx][k] <= 0)
                        y_unsettled = (~y_active) & (~y_passive)
                        delta_active = (self.d_lbs[(i, j)][self.linear_layer_idx][k] >= 0)
                        delta_passive = (self.d_ubs[(i, j)][self.linear_layer_idx][k] <= 0)
                        delta_unsettled = (~delta_active) & (~delta_passive)                                                                                
                        if x_unsettled and y_passive and delta_active:
                            self.gmdl.addConstr(ds[i][j - i -1][k] <= self.gurobi_variables[-1]['ds'][i][j - i -1][k])
                        elif x_unsettled and y_active:
                            self.gmdl.addConstr(ds[i][j - i -1][k] >= self.gurobi_variables[-1]['ds'][i][j - i -1][k])
                        elif x_passive and y_unsettled and delta_passive:
                            self.gmdl.addConstr(ds[i][j - i -1][k] >= self.gurobi_variables[-1]['ds'][i][j - i -1][k])
                        elif x_active and y_unsettled:
                            self.gmdl.addConstr(ds[i][j - i -1][k] <= self.gurobi_variables[-1]['ds'][i][j - i -1][k])
                        elif x_unsettled and y_unsettled and delta_active:
                            self.gmdl.addConstr(ds[i][j - i -1][k] <= self.gurobi_variables[-1]['ds'][i][j - i -1][k])
                            self.gmdl.addConstr(ds[i][j - i -1][k] >= 0.0)                        
                        elif x_unsettled and y_unsettled and delta_passive:
                            self.gmdl.addConstr(ds[i][j - i -1][k] >= self.gurobi_variables[-1]['ds'][i][j - i -1][k])
                            self.gmdl.addConstr(ds[i][j - i -1][k] <= 0.0)
                        elif x_unsettled and y_unsettled and delta_unsettled:
                            temp_lb = self.d_lbs[(i, j)][self.linear_layer_idx][k]
                            temp_ub = self.d_ubs[(i, j)][self.linear_layer_idx][k]
                            d_lambda_lb = temp_lb / (temp_lb - temp_ub + 1e-15)
                            d_lambda_ub = temp_ub / (temp_ub - temp_lb + 1e-15)
                            d_mu_lb = d_lambda_ub * temp_lb
                            d_mu_ub = -d_lambda_ub * temp_lb
                            self.gmdl.addConstr(ds[i][j - i -1][k] >= d_lambda_lb * self.gurobi_variables[-1]['ds'][i][j - i -1][k] + d_mu_lb)
                            self.gmdl.addConstr(ds[i][j - i -1][k] <= d_lambda_ub * self.gurobi_variables[-1]['ds'][i][j - i -1][k] + d_mu_ub)                                                                        
        self.gurobi_variables.append({'vs': vs, 'ds': ds})

    def get_sigmoid_lambda_mu(self, lb_layer, ub_layer):
        sigmoid_lb, sigmoid_ub = torch.sigmoid(lb_layer), torch.sigmoid(ub_layer)
        lmbda = torch.where(lb_layer < ub_layer, (sigmoid_ub - sigmoid_lb) / (ub_layer - lb_layer + 1e-15),  sigmoid_lb * (1 - sigmoid_lb))
        lmbda_ = torch.min(sigmoid_ub * (1 - sigmoid_ub), sigmoid_lb * (1 - sigmoid_lb))
        lambda_lb = torch.where(lb_layer > 0, lmbda, lmbda_)
        mu_lb = torch.where(lb_layer > 0, sigmoid_lb - torch.mul(lmbda, lb_layer),  sigmoid_lb - torch.mul(lmbda_, lb_layer))
        lambda_ub = torch.where(ub_layer < 0, lmbda, lmbda_)
        mu_ub =  torch.where(ub_layer < 0, sigmoid_ub - torch.mul(lmbda, ub_layer),  sigmoid_ub - torch.mul(lmbda_, lb_layer))
        return lambda_lb, mu_lb, lambda_ub, mu_ub


    def get_sigmoid_diff_lambda_mu(self, lb, ub, delta_lb_layer, delta_ub_layer):
        sigmoid_lb, sigmoid_ub = torch.sigmoid(lb), torch.sigmoid(ub)
        lambda_lower, lambda_upper = sigmoid_lb * (1.0 - sigmoid_lb), sigmoid_ub * (1.0 - sigmoid_ub)
        
        input_active = (lb >= 0)
        input_passive = (ub <= 0) 
        input_unsettled = ~(input_active) & ~(input_passive)

        deriv_min = torch.min(lambda_lower, lambda_upper)
        deriv_max = 0.25 * torch.ones(lb.size(), device='cpu')
        deriv_max = torch.where(~input_unsettled, torch.max(lambda_lower, lambda_upper), deriv_max)



        delta_active = (delta_lb_layer >= 0)
        delta_passive = (delta_ub_layer <= 0)
        delta_unsettled = ~(delta_active) & ~(delta_passive)

        lambda_lb = torch.zeros(lb.size(), device='cpu')
        lambda_ub = torch.zeros(lb.size(), device='cpu')

        mu_lb = torch.zeros(lb.size(), device='cpu')
        mu_ub = torch.zeros(lb.size(), device='cpu')

        # case 1 delta_lb >= 0 
        lambda_lb = torch.where(delta_active, deriv_min, lambda_lb)
        lambda_ub = torch.where(delta_active, deriv_max, lambda_ub)

        # case 2 delta_ub <= 0 
        lambda_lb = torch.where(delta_passive, deriv_max, lambda_lb)
        lambda_ub = torch.where(delta_passive, deriv_min, lambda_ub)

        # case 3 delta_lb < 0 and delta_ub > 0
        prod_lb_ub = delta_lb_layer * delta_ub_layer
        diff_lb_ub = (delta_ub_layer - delta_lb_layer + 1e-15)
        temp_lambda_lb = (deriv_min * delta_ub_layer - 0.25* delta_lb_layer) / diff_lb_ub
        temp_lambda_ub = (0.25 * delta_ub_layer - deriv_min * delta_lb_layer) / diff_lb_ub
    
        temp_mu_lb = (0.25 - deriv_min) * prod_lb_ub
        temp_mu_lb = temp_mu_lb / diff_lb_ub
        temp_mu_ub = (deriv_min - 0.25) * prod_lb_ub
        temp_mu_ub = temp_mu_ub / diff_lb_ub

        lambda_lb = torch.where(delta_unsettled, temp_lambda_lb, lambda_lb)
        lambda_ub = torch.where(delta_unsettled, temp_lambda_ub, lambda_ub)
        mu_lb = torch.where(delta_unsettled, temp_mu_lb, mu_lb)
        mu_ub = torch.where(delta_unsettled, temp_mu_ub, mu_ub)

        return lambda_lb, mu_lb, lambda_ub, mu_ub


    def create_sigmoid_constraints(self, layer_idx):
        vs, ds = self.create_vars(layer_idx, 'sigmoid')
        for i in range(self.batch_size):
            tensor_length = self.x_lbs[i][self.linear_layer_idx].shape[0]
            lb, ub = self.x_lbs[i][layer_idx-1], self.x_ubs[i][layer_idx-1]
            lambda_lb, mu_lb, lambda_ub, mu_ub = self.get_sigmoid_lambda_mu(lb_layer=lb, ub_layer=ub)
            for j in range(tensor_length):
                self.gmdl.addConstr(vs[i][j] <= lambda_ub[j] * self.gurobi_variables[-1]['vs'][i][j] + mu_ub[j])
                self.gmdl.addConstr(vs[i][j] >= lambda_lb[j] * self.gurobi_variables[-1]['vs'][i][j] + mu_lb[j])

        if self.track_differences is True:
            for i in range(self.batch_size):
                for j in range(i+1, self.batch_size):
                    self.gmdl.addConstr(ds[i][j - i - 1] == vs[i] - vs[j])

            for i in range(self.batch_size):
                for j in range(i+1, self.batch_size):
                    tensor_length = self.x_lbs[i][self.linear_layer_idx].shape[0]
                    lb = torch.min(self.x_lbs[i][layer_idx-1], self.x_lbs[j][layer_idx-1])
                    ub = torch.max(self.x_ubs[i][layer_idx-1], self.x_ubs[j][layer_idx-1])
                    d_lb = self.d_lbs[(i, j)][layer_idx-1]
                    d_ub = self.d_ubs[(i, j)][layer_idx-1]
                    lambda_lb, mu_lb, lambda_ub, mu_ub = self.get_sigmoid_diff_lambda_mu(lb=lb, ub=ub,
                                                                                        delta_lb_layer=d_lb,
                                                                                        delta_ub_layer=d_ub)
                    for k in range(tensor_length):
                        self.gmdl.addConstr(ds[i][j - i -1][k] >= lambda_lb[k]*self.gurobi_variables[-1]['ds'][i][j - i -1][k]
                                                                 + mu_lb[k])
                        self.gmdl.addConstr(ds[i][j - i -1][k] <= lambda_ub[k]*self.gurobi_variables[-1]['ds'][i][j - i -1][k]
                                                                 + mu_ub[k])
            if self.lightweight_difference is True:
                self.track_differences = False
        self.gurobi_variables.append({'vs': vs, 'ds': ds})

    
    def get_tanh_lambda_mu(self, lb_layer, ub_layer):
        tanh_lb, tanh_ub = torch.tanh(lb_layer), torch.tanh(ub_layer)
        lmbda = torch.where(lb_layer < ub_layer, (tanh_ub - tanh_lb) / (ub_layer - lb_layer + 1e-15),  1 - tanh_lb * tanh_lb)
        lmbda_ = torch.min(1 - tanh_ub * tanh_ub, 1 - tanh_lb * tanh_lb)

        lambda_lb = torch.where(lb_layer > 0, lmbda, lmbda_)
        mu_lb = torch.where(lb_layer > 0, tanh_lb - torch.mul(lmbda, lb_layer),  tanh_lb - torch.mul(lmbda_, lb_layer))
        
        lambda_ub = torch.where(ub_layer < 0, lmbda, lmbda_)
        mu_ub =  torch.where(ub_layer < 0, tanh_ub - torch.mul(lmbda, ub_layer),  tanh_ub - torch.mul(lmbda_, lb_layer))

        return lambda_lb, mu_lb, lambda_ub, mu_ub

    def get_deriv_min_max(self, lb, ub):
        tanh_lb, tanh_ub = torch.tanh(lb), torch.tanh(ub)
        lambda_lower, lambda_upper = 1.0 - (tanh_lb * tanh_lb), 1.0 - (tanh_ub * tanh_ub)
        input_active = (lb >= 0)
        input_passive = (ub <= 0) 
        input_unsettled = ~(input_active) & ~(input_passive)

        deriv_min = torch.min(lambda_lower, lambda_upper)
        deriv_max = torch.ones(lb.size(), device='cpu')
        deriv_max = torch.where(~input_unsettled, torch.max(lambda_lower, lambda_upper), deriv_max)
        return deriv_min, deriv_max

    def get_tanh_diff_lambda_mu(self, lb, ub, delta_lb_layer, delta_ub_layer):
        tanh_lb, tanh_ub = torch.tanh(lb), torch.tanh(ub)
        lambda_lower, lambda_upper = 1.0 - (tanh_lb * tanh_lb), 1.0 - (tanh_ub * tanh_ub)

        input_active = (lb >= 0)
        input_passive = (ub <= 0) 
        input_unsettled = ~(input_active) & ~(input_passive)

        deriv_min = torch.min(lambda_lower, lambda_upper)
        deriv_max = torch.ones(lb.size(), device='cpu')
        deriv_max = torch.where(~input_unsettled, torch.max(lambda_lower, lambda_upper), deriv_max)



        delta_active = (delta_lb_layer >= 0)
        delta_passive = (delta_ub_layer <= 0)
        delta_unsettled = ~(delta_active) & ~(delta_passive)

        lambda_lb = torch.zeros(lb.size(), device='cpu')
        lambda_ub = torch.zeros(lb.size(), device='cpu')

        mu_lb = torch.zeros(lb.size(), device='cpu')
        mu_ub = torch.zeros(lb.size(), device='cpu')

        # case 1 delta_lb >= 0 
        lambda_lb = torch.where(delta_active, deriv_min, lambda_lb)
        lambda_ub = torch.where(delta_active, deriv_max, lambda_ub)

        # case 2 delta_ub <= 0 
        lambda_lb = torch.where(delta_passive, deriv_max, lambda_lb)
        lambda_ub = torch.where(delta_passive, deriv_min, lambda_ub)

        # case 3 delta_lb < 0 and delta_ub > 0
        prod_lb_ub = delta_lb_layer * delta_ub_layer
        diff_lb_ub = (delta_ub_layer - delta_lb_layer + 1e-15)
        temp_lambda_lb = (deriv_min * delta_ub_layer - delta_lb_layer) / diff_lb_ub
        temp_lambda_ub = (delta_ub_layer - deriv_min * delta_lb_layer) / diff_lb_ub
    
        temp_mu_lb = (1.0 - deriv_min) * prod_lb_ub
        temp_mu_lb = temp_mu_lb / diff_lb_ub
        temp_mu_ub = (deriv_min - 1.0) * prod_lb_ub
        temp_mu_ub = temp_mu_ub / diff_lb_ub

        lambda_lb = torch.where(delta_unsettled, temp_lambda_lb, lambda_lb)
        lambda_ub = torch.where(delta_unsettled, temp_lambda_ub, lambda_ub)
        mu_lb = torch.where(delta_unsettled, temp_mu_lb, mu_lb)
        mu_ub = torch.where(delta_unsettled, temp_mu_ub, mu_ub)

        return lambda_lb, mu_lb, lambda_ub, mu_ub


    def create_tanh_constraints(self, layer_idx):
        vs, ds = self.create_vars(layer_idx, 'tanh')
        for i in range(self.batch_size):
            tensor_length = self.x_lbs[i][self.linear_layer_idx].shape[0]
            lb, ub = self.x_lbs[i][layer_idx-1], self.x_ubs[i][layer_idx-1]
            lambda_lb, mu_lb, lambda_ub, mu_ub = self.get_tanh_lambda_mu(lb_layer=lb, ub_layer=ub)
            for j in range(tensor_length):
                self.gmdl.addConstr(vs[i][j] <= lambda_ub[j] * self.gurobi_variables[-1]['vs'][i][j] + mu_ub[j])
                self.gmdl.addConstr(vs[i][j] >= lambda_lb[j] * self.gurobi_variables[-1]['vs'][i][j] + mu_lb[j])

        if self.track_differences is True:
            for i in range(self.batch_size):
                for j in range(i+1, self.batch_size):
                    self.gmdl.addConstr(ds[i][j - i - 1] == vs[i] - vs[j])

            for i in range(self.batch_size):
                for j in range(i+1, self.batch_size):
                    tensor_length = self.x_lbs[i][self.linear_layer_idx].shape[0]
                    lb = torch.min(self.x_lbs[i][layer_idx-1], self.x_lbs[j][layer_idx-1])
                    ub = torch.max(self.x_ubs[i][layer_idx-1], self.x_ubs[j][layer_idx-1])
                    d_lb = self.d_lbs[(i, j)][layer_idx-1]
                    d_ub = self.d_ubs[(i, j)][layer_idx-1]
                    lambda_lb, mu_lb, lambda_ub, mu_ub = self.get_tanh_diff_lambda_mu(lb=lb, ub=ub,
                                                                                        delta_lb_layer=d_lb,
                                                                                        delta_ub_layer=d_ub)
                    deriv_min, deriv_max = self.get_deriv_min_max(lb=lb, ub=ub)
                    for k in range(tensor_length):
                        self.gmdl.addConstr(ds[i][j - i -1][k] >= lambda_lb[k]*self.gurobi_variables[-1]['ds'][i][j - i -1][k]
                                                                 + mu_lb[k])
                        self.gmdl.addConstr(ds[i][j - i -1][k] <= lambda_ub[k]*self.gurobi_variables[-1]['ds'][i][j - i -1][k]
                                                                 + mu_ub[k])
                        if d_lb[k] < 0 and d_ub[k] > 0:
                            ind = self.gmdl.addVar(vtype=grb.GRB.BINARY, name=f"ind_{self.tanh_binary_count}")
                            self.addGenConstrIndicator(ind, True, self.gurobi_variables[-1]['ds'][i][j - i -1][k] >= 0)
                            self.addGenConstrIndicator(ind, False, self.gurobi_variables[-1]['ds'][i][j - i -1][k] <= -1e-9)
                            self.gmdl.addConstr(ds[i][j - i -1][k] >= (1 - ind) * deriv_max * d_lb[k] + deriv_min[k] * self.gurobi_variables[-1]['ds'][i][j - i -1][k])
                            self.gmdl.addConstr(ds[i][j - i -1][k] >= (ind) * (-deriv_max) * d_ub[k]  + deriv_max[k] * self.gurobi_variables[-1]['ds'][i][j - i -1][k])
                            self.gmdl.addConstr(ds[i][j - i -1][k] <= (ind) * deriv_max * d_ub[k] + deriv_min[k] * self.gurobi_variables[-1]['ds'][i][j - i -1][k])
                            self.gmdl.addConstr(ds[i][j - i -1][k] <= (1- ind) * (deriv_max) * (-d_lb[k]) + deriv_max[k] * self.gurobi_variables[-1]['ds'][i][j - i -1][k])

            if self.lightweight_difference is True:
                self.track_differences = False 
        self.gurobi_variables.append({'vs': vs, 'ds': ds})

    # Crate constraints for con
    def create_conv2d_constraints_helper(self, vars, pre_vars, num_kernel, output_h, 
                                         output_w, bias, weight, layer, input_h, input_w):
        out_idx = 0
        gvars_array = [np.array(pre_var.tolist()).reshape((-1, input_h, input_w)) for pre_var in pre_vars]
        pre_lb_size = [None, None, input_h, input_w]

        for out_chan_idx in range(num_kernel):
            for out_row_idx in range(output_h):
                for out_col_idx in range(output_w):
                    lin_expressions = [grb.LinExpr() for i in range(len(pre_vars))]

                    for in_chan_idx in range(layer.weight.shape[1]):

                        # New version of conv layer for building mip by skipping kernel loops
                        ker_row_min, ker_row_max = 0, layer.weight.shape[2]
                        in_row_idx_min = -layer.padding[0] + layer.stride[0] * out_row_idx
                        in_row_idx_max = -layer.padding[0] + layer.stride[0] * out_row_idx + layer.weight.shape[2] - 1
                        if in_row_idx_min < 0: 
                            ker_row_min = 0 - in_row_idx_min
                        if in_row_idx_max >= pre_lb_size[2]: 
                            ker_row_max = ker_row_max - in_row_idx_max + pre_lb_size[2] -1
                        in_row_idx_min, in_row_idx_max = max(in_row_idx_min, 0), min(in_row_idx_max, pre_lb_size[2] - 1)

                        ker_col_min, ker_col_max = 0, layer.weight.shape[3]
                        in_col_idx_min = -layer.padding[1] + layer.stride[1] * out_col_idx
                        in_col_idx_max = -layer.padding[1] + layer.stride[1] * out_col_idx + layer.weight.shape[3] - 1
                        if in_col_idx_min < 0: ker_col_min = 0 - in_col_idx_min
                        if in_col_idx_max >= pre_lb_size[3]: 
                            ker_col_max = ker_col_max - in_col_idx_max + pre_lb_size[3] -1
                        in_col_idx_min, in_col_idx_max = max(in_col_idx_min, 0), min(in_col_idx_max, pre_lb_size[3] - 1)

                        coeffs = layer.weight[out_chan_idx, in_chan_idx, ker_row_min:ker_row_max, ker_col_min:ker_col_max].reshape(-1)
                        for i, gvars in enumerate(gvars_array):
                            gvar = gvars[in_chan_idx, in_row_idx_min:in_row_idx_max+1, in_col_idx_min:in_col_idx_max+1].reshape(-1)
                            lin_expressions[i] += grb.LinExpr(coeffs, gvar)
                    for i, var in enumerate(vars):
                        self.gmdl.addConstr(var[out_idx] == lin_expressions[i] + bias[out_chan_idx].item())
                    out_idx += 1


    # LP formulation only used for debugging.
    def create_lp(self):
        if self.batch_size <= 0:
            return
        self.gmdl.setParam('OutputFlag', False)
        self.gmdl.setParam('TimeLimit', 600)
        self.gmdl.Params.MIPFocus = 3
        self.gmdl.Params.ConcurrentMIP = 3
        self.constraint_time = - time.time()
        if self.args is not None and self.args.fold_conv_layers is True:
            self.create_constraints_folded_conv_layers()
        else:
            if self.par_constraints:
                self.create_parallel_constraints()
            else:
                self.create_constraints()


    # LP optimization for debugging only.
    def optimize_lp(self):
        assert len(self.constraint_matrices) == self.batch_size
        if self.batch_size <= 0:
            return 0.0
        final_vars = []
        final_min_vars = []
        for i in range(self.batch_size):
            constraint_mat = self.constraint_matrices[i]
            final_var = self.gmdl.addMVar(constraint_mat.shape[1], lb=-float('inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, 
                                            name=f'final_var_{i}')
            self.gmdl.addConstr(final_var == constraint_mat.T.detach().numpy() @ self.gurobi_variables[-1]['vs'][i])
            final_vars.append(final_var)
            final_var_min = self.gmdl.addVar(lb=-float('inf'), ub=float('inf'), 
                                                vtype=grb.GRB.CONTINUOUS, 
                                                name=f'final_var_min_{i}')
            self.gmdl.addGenConstrMin(final_var_min, final_var.tolist())
            final_min_vars.append(final_var_min)
        problem_min = self.gmdl.addVar(lb=-float('inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, 
                                        name='problem_min')
        self.gmdl.addGenConstrMax(problem_min, final_min_vars)
        self.gmdl.setObjective(problem_min, grb.GRB.MINIMIZE)

        self.constraint_time += time.time()
        self.optimize_time = - time.time()        
        self.gmdl.optimize()
        self.optimize_time += time.time() 


        if self.gmdl.status == 2:

            return problem_min.X
        else:
            if self.gmdl.status == 4:
                self.gmdl.setParam('PreDual',0)
                self.gmdl.setParam('DualReductions', 0)
                self.gmdl.optimize()
            elif self.gmdl.status == 13 or self.gmdl.status == 9:
                if self.gmdl.SolCount > 0:
                    return self.gmdl.ObjBound
                else:
                    return 0.0
            if self.gmdl.status == 3:
                return -1e6

        
    def create_conv2d_constraints(self, layer, layer_idx):
        vs, ds = self.create_vars(layer_idx, 'conv2d')
        weight = layer.weight
        bias = layer.bias
        assert layer.dilation == (1, 1)

        # Ref. https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d ###
        k_h, k_w = layer.kernel_size
        s_h, s_w = layer.stride
        p_h, p_w = layer.padding
        num_kernel = weight.shape[0]
        input_h, input_w = self.shape[1:]
        output_h = int((input_h + 2 * p_h - k_h) / s_h + 1)
        output_w = int((input_w + 2 * p_w - k_w) / s_w + 1)

        # Updated shape
        self.shape = (num_kernel, output_h, output_w)
        self.create_conv2d_constraints_helper(vars=vs, pre_vars=self.gurobi_variables[-1]['vs'],
                                                   num_kernel=num_kernel, output_h=output_h, output_w=output_w,
                                                  bias=bias, weight=weight, layer=layer, input_h=input_h, input_w=input_w)

        if self.track_differences is True:
            for i in range(self.batch_size):
                for j in range(i+1, self.batch_size):
                    self.gmdl.addConstr(ds[i][j - i - 1] == vs[i] - vs[j])
                
        self.gurobi_variables.append({'vs': vs, 'ds': ds})

    def get_layer_type(self, layer):
        return layer.type