import torch
import numpy as np
from src.network_converters.network_conversion_helper import get_pytorch_net
from auto_LiRPA.operators import BoundLinear, BoundConv, BoundRelu
from auto_LiRPA import BoundedModule, PerturbationLpNorm, BoundedTensor
from src.baseline_res import BaselineVerifierRes
from src.common import Domain

class LirpaTransformer:
    def __init__(self, prop, complete=False, device=None, args=None, prop_list=None):
        self.prop = prop
        self.args = args
        self.prop_list = prop_list
        self.device = device if self.args.device is not None else 'cpu'
        self.ilb = prop.input_lb
        self.iub = prop.input_ub
        self.eps = torch.max(self.iub - self.ilb) / 2.0
        self.input = prop.input
        self.input_list = None
        # I/O formulation uses crown as the individual verifier.
        self.base_method_io_formulation = 'CROWN'
        self.raven_domain = 'CROWN' if args.individual_prop_domain == Domain.LIRPA else 'CROWN-Optimized'
        self.lbs = []
        self.ubs = []
        self.input_name = None
        self.last_name = None
        self.la_coef = None
        self.lbias = None
        self.supported_operators = [BoundConv, BoundLinear, BoundRelu]
        self.baseline_result = None
        self.baseline_result_list = None
        # Populate the layer names for the conv and linear layers.
        self.layer_names = []
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
    
    def shift_to_device(self):
        self.ilb = self.ilb.to(self.device)
        self.iub = self.iub.to(self.device)
        self.input = self.input.to(self.device)
    
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

    def process_batch_output(self, bounded_model, lower_bnd, 
                                    A_dict, final_lower_bnd=None, final_upper_bnd=None,
                                    refined_lower_bnd=None):
        # Check the lower bounds are computed for a batch.
        assert len(lower_bnd.shape) > 1
        batch_size = lower_bnd.shape[0]
        lbs_dict, ubs_dict = {}, {}
        lA_list, lbias_list = [], []
        self.baseline_result_list = []
        # iterate over and populate the results
        for i in range(batch_size):
            lbs_dict[i] = []
            ubs_dict[i] = []
            lA_list.append(A_dict[self.last_name][self.input_name]['lA'][i].squeeze().reshape(self.number_of_class-1, -1))
            lbias_list.append(A_dict[self.last_name][self.input_name]['lbias'][i].squeeze())

            for node_name, node in bounded_model._modules.items():
                if node_name is self.last_name:
                    continue
                if type(node) in self.supported_operators:
                    if type(node) != BoundRelu:
                        assert node.lower is not None
                        assert node.upper is not None                    
                        lbs_dict[i].append(node.lower[i].reshape(-1))
                        ubs_dict[i].append(node.upper[i].reshape(-1))
                    else:
                        assert min(len(lbs_dict[i]), len(ubs_dict[i])) > 0
                        lbs_dict[i].append(torch.zeros(lbs_dict[i][-1].shape, device=lbs_dict[i][-1].device).reshape(-1))
                        ubs_dict[i].append(torch.max(ubs_dict[i][-1], torch.zeros_like(ubs_dict[i][-1], device=ubs_dict[i][-1].device)).reshape(-1))

            lbs_dict[i].append(final_lower_bnd[i].squeeze().reshape(-1))
            ubs_dict[i].append(final_upper_bnd[i].squeeze().reshape(-1))
            # Put the ilb and iub
            lbs_dict[i].append(self.ilb_list[i].squeeze().reshape(-1))
            ubs_dict[i].append(self.iub_list[i].squeeze().reshape(-1))

            self.baseline_result_list.append(BaselineVerifierRes(input=self.input_list[i].reshape(-1), layer_lbs=lbs_dict[i], layer_ubs=ubs_dict[i], final_lb=lower_bnd[i].squeeze(), 
                                        final_ub = None, lb_bias=lbias_list[-1], lb_coef=lA_list[-1], 
                                        eps=self.eps, last_conv_diff_struct=None,
                                        refined_lower_bnd= refined_lower_bnd[i] if refined_lower_bnd[i] is not None else None))


            
    def process_bound_propagation_output(self, bounded_model, lower_bnd, 
                                         A_dict, final_lower_bnd=None, final_upper_bnd=None, 
                                         refined_lower_bnd=None):
        i = 0
        for node_name, node in bounded_model._modules.items():
            if node_name is self.last_name:
                continue
            if type(node) in self.supported_operators:
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
                                        eps=self.eps, last_conv_diff_struct=None,
                                        refined_lower_bnd= refined_lower_bnd if refined_lower_bnd is not None else None)

    def populate_layer_names(self, model):
        for node_name, node in model._modules.items():
            if type(node) in [BoundLinear, BoundConv]:
                self.layer_names.append(node_name)

    # Handles a list of batches.
    def handle_prop_list(self, bounded_model):
        input_list = []
        ilb_list = []
        iub_list = []
        constraint_matrices = []
        for prop in self.prop_list:
            ilb_list.append(prop.input_lb.view(*self.shape))
            iub_list.append(prop.input_ub.view(*self.shape))
            input_list.append(prop.input.view(*self.shape))
            constraint_matrices.append(prop.output_constr_mat().T)
        
        input_list = torch.stack(input_list)
        self.input_list = input_list
        input_list = input_list.to(bounded_model.device) 
        self.ilb_list = torch.stack(ilb_list)
        ilb_list = self.ilb_list.to(bounded_model.device)
        self.iub_list = torch.stack(iub_list)
        iub_list = self.iub_list.to(bounded_model.device)
        constraint_matrices = torch.stack(constraint_matrices).to(bounded_model.device)
        self.get_output_layer_name(bounded_model=bounded_model)
        ptb = PerturbationLpNorm(norm = np.inf, x_L=ilb_list, x_U=iub_list)
        bounded_images = BoundedTensor(input_list, ptb)
        coef_dict = {self.last_name: [self.input_name]}
        # import pdb; pdb.set_trace()
        result = bounded_model.compute_bounds(x=(bounded_images,), method=self.base_method_io_formulation, C=constraint_matrices,
                                        bound_upper=False, return_A=True, needed_A_dict=coef_dict)
        
        lower_bnd, _, A_dict = result

        # first extract upper and lower bound of final layer
        result = bounded_model.compute_bounds(x=(bounded_images,), method=self.raven_domain, C=None,
                                        bound_upper=True, return_A=True, needed_A_dict=coef_dict)
        final_lower_bnd, final_upper_bnd, _ = result
        if len(final_lower_bnd.shape) > 1:
            final_lower_bnd = final_lower_bnd.squeeze()
        if len(final_upper_bnd.shape) > 1:
            final_upper_bnd = final_upper_bnd.squeeze()
        if self.raven_domain == 'CROWN-Optimized':
            # Update the internal bounds with alpha crown which are used by RaVeN
            refined_lower_bnd, _, _ = bounded_model.compute_bounds(x=(bounded_images,), method=self.raven_domain, C=constraint_matrices,
                                bound_upper=False, return_A=True, needed_A_dict=coef_dict)
        self.process_batch_output(bounded_model=bounded_model, lower_bnd=lower_bnd, A_dict=A_dict,
                                              final_lower_bnd=final_lower_bnd, final_upper_bnd=final_upper_bnd, 
                                              refined_lower_bnd=refined_lower_bnd)



    def handle_prop(self, net):
        torch_net = get_pytorch_net(model=net, remove_last_layer=False, all_linear=False)
        # shift the network to the specified device
        torch_net = torch_net.to(self.device)
        rand_input = torch.rand(self.shape)
        if len(rand_input.shape) < 4:
            rand_input = rand_input.view(1, *rand_input.shape)
        bounded_model = BoundedModule(torch_net, rand_input, bound_opts={})
        if self.args.enable_batch_processing and self.prop_list is not None:
            self.handle_prop_list(bounded_model=bounded_model)
            return self

        self.input = self.input.view(-1, *self.shape)
        self.ilb = self.ilb.view(-1, *self.shape)
        self.iub = self.iub.view(-1, *self.shape)
        self.get_output_layer_name(bounded_model=bounded_model)
        ptb = PerturbationLpNorm(norm = np.inf, x_L=self.ilb, x_U=self.iub)
        bounded_images = BoundedTensor(self.input, ptb)
        coef_dict = { self.last_name: [self.input_name]}
        constraint_matrix = self.prop.output_constr_mat().T
        # shift the constraint matrix to the specified device
        constraint_matrix = constraint_matrix.to(self.device)
        if len(constraint_matrix.shape) < 3:
            constraint_matrix = constraint_matrix.reshape(1, *constraint_matrix.shape)        
        result = bounded_model.compute_bounds(x=(bounded_images,), method=self.base_method_io_formulation, C=constraint_matrix,
                                        bound_upper=False, return_A=True, needed_A_dict=coef_dict)
        lower_bnd, upper, A_dict = result

        # first extract upper and lower bound of final layer
        result = bounded_model.compute_bounds(x=(bounded_images,), method=self.raven_domain, C=None,
                                        bound_upper=True, return_A=True, needed_A_dict=coef_dict)
        final_lower_bnd, final_upper_bnd, _ = result
        if len(final_lower_bnd.shape) > 1:
            final_lower_bnd = final_lower_bnd.squeeze()
        if len(final_upper_bnd.shape) > 1:
            final_upper_bnd = final_upper_bnd.squeeze()
        if self.raven_domain == 'CROWN-Optimized':
            # Update the internal bounds with alpha crown which are used by RaVeN
            refined_lower_bnd, _, _ = bounded_model.compute_bounds(x=(bounded_images,), method=self.raven_domain, C=constraint_matrix,
                                bound_upper=False, return_A=True, needed_A_dict=coef_dict)
        self.process_bound_propagation_output(bounded_model=bounded_model, lower_bnd=lower_bnd, A_dict=A_dict,
                                              final_lower_bnd=final_lower_bnd, final_upper_bnd=final_upper_bnd, 
                                              refined_lower_bnd=refined_lower_bnd)
        return self
    
    def populate_result_list(self):
        assert self.baseline_result_list is not None
        return self.baseline_result_list


    def populate_baseline_verifier_result(self, args=None):
        assert self.baseline_result is not None
        return self.baseline_result
