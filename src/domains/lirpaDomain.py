import torch
import numpy as np
from src.network_converters.network_conversion_helper import get_pytorch_net
from auto_LiRPA.operators import BoundLinear, BoundConv, BoundRelu
from auto_LiRPA import BoundedModule, PerturbationLpNorm, BoundedTensor
from src.baseline_res import BaselineVerifierRes

class LirpaTransformer:
    def __init__(self, prop, complete=False, device=None, args=None):
        self.prop = prop
        self.device = device if device is not None else 'cpu'
        self.ilb = prop.input_lb
        self.iub = prop.input_ub
        self.eps = torch.max(self.iub - self.ilb) / 2.0
        self.input = prop.input
        self.base_method = 'CROWN-Optimized'
        self.lbs = []
        self.ubs = []
        self.input_name = None
        self.last_name = None
        self.la_coef = None
        self.lbias = None
        self.baseline_result = None
        self.size = prop.get_input_size()
        self.number_of_class = 10
        if self.size == 784:
            self.shape = (1, 28, 28)
        elif self.size == 3072:
            self.shape = (3, 32, 32)
        elif self.size == 2:
            # For debug network
            self.shape = (1, 1, 2)
            self.number_of_class = 2
        elif self.size == 87:
            self.shape = (1, 1, 87)
            self.number_of_class = None
    
    def get_output_layer_name(self, bounded_model):
        i = 0
        last_name = None
        for node_name, node in bounded_model._modules.items():
            if i == 0:
                self.input_name = node_name
            i += 1
            if type(node) in [BoundLinear, BoundConv]:
                last_name = node_name
        assert last_name is not None
        self.last_name = last_name
    
    def proceed_bound_propagation_output(self, bounded_model, lower_bnd, 
                                         A_dict, final_lower_bnd=None, final_upper_bnd=None):
        i = 0
        supported_operators = [BoundConv, BoundLinear, BoundRelu]
        for node_name, node in bounded_model._modules.items():
            if node_name is self.last_name:
                continue
            if type(node) in supported_operators:
                if type(node) != BoundRelu:
                    assert node.lower is not None
                    assert node.upper is not None                    
                    self.lbs.append(node.lower.reshape(-1))
                    self.ubs.append(node.upper.reshape(-1))
                else:
                    assert min(len(self.lbs), len(self.ubs)) > 0
                    self.lbs.append(torch.zeros(self.lbs[-1].shape, device=self.lbs[-1].device).reshape(-1))
                    self.ubs.append(torch.max(self.ubs[-1], torch.zeros_like(self.ubs[-1], device=self.ubs[-1].device)).reshape(-1))
            # import pdb; pdb.set_trace()
        if final_lower_bnd is None or final_upper_bnd is None:
            negative_inf = torch.zeros(self.number_of_class, device=self.lbs[-1].device).fill_(float('-inf'))
            positive_inf = torch.zeros(self.number_of_class, device=self.lbs[-1].device).fill_(float('-inf'))
            self.lbs.append(negative_inf)
            self.ubs.append(positive_inf)
        else:
            self.lbs.append(final_lower_bnd)
            self.ubs.append(final_upper_bnd)
        self.lbs.append(self.ilb.reshape(-1))
        self.ubs.append(self.iub.reshape(-1))
        lA = A_dict[self.last_name][self.input_name]['lA'].squeeze().reshape(self.number_of_class-1, -1)
        lbias = A_dict[self.last_name][self.input_name]['lbias'].squeeze()
        # import pdb;pdb.set_trace()
        self.baseline_result = BaselineVerifierRes(input=self.input.reshape(-1), layer_lbs=self.lbs, layer_ubs=self.ubs, final_lb=lower_bnd, 
                                        final_ub = None, lb_bias=lbias, lb_coef=lA, 
                                        eps=self.eps, last_conv_diff_struct=None)


    def handle_prop(self, net):
        torch_net = get_pytorch_net(model=net, remove_last_layer=False, all_linear=False)
        rand_input = torch.rand(self.shape)
        if len(rand_input.shape) < 4:
            rand_input = rand_input.view(1, *rand_input.shape)
        bounded_model = BoundedModule(torch_net, rand_input, bound_opts={})
        self.input = self.input.view(-1, *self.shape)
        self.ilb = self.ilb.view(-1, *self.shape)
        self.iub = self.iub.view(-1, *self.shape)
        self.get_output_layer_name(bounded_model=bounded_model)
        ptb = PerturbationLpNorm(norm = np.inf, x_L=self.ilb, x_U=self.iub)
        bounded_images = BoundedTensor(self.input, ptb)
        coef_dict = { self.last_name: [self.input_name]}
        constraint_matrix = self.prop.output_constr_mat().T
        if len(constraint_matrix.shape) < 3:
            constraint_matrix = constraint_matrix.reshape(1, *constraint_matrix.shape)
        # first extract upper and lower bound of final layer
        result = bounded_model.compute_bounds(x=(bounded_images,), method=self.base_method, C=None,
                                        bound_upper=True, return_A=True, needed_A_dict=coef_dict)
        final_lower_bnd, final_upper_bnd, _ = result
        if len(final_lower_bnd.shape) > 1:
            final_lower_bnd = final_lower_bnd.squeeze()
        if len(final_upper_bnd.shape) > 1:
            final_upper_bnd = final_upper_bnd.squeeze()
        
        result = bounded_model.compute_bounds(x=(bounded_images,), method=self.base_method, C=constraint_matrix,
                                        bound_upper=False, return_A=True, needed_A_dict=coef_dict)
        lower_bnd, upper, A_dict = result
        self.proceed_bound_propagation_output(bounded_model=bounded_model, lower_bnd=lower_bnd, A_dict=A_dict,
                                              final_lower_bnd=final_lower_bnd, final_upper_bnd=final_upper_bnd)
        return self


    def populate_baseline_verifier_result(self, args=None):
        assert self.baseline_result is not None
        return self.baseline_result
