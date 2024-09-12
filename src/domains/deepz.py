import torch
from torch.nn import functional as F
from src.baseline_res import BaselineVerifierRes

from src.specs.out_spec import create_out_constr_matrix

device = 'cpu'


"""
This implementation taken from https://github.com/eth-sri/eran.
"""
class ZonoTransformer:
    def __init__(self, prop, cof_constrain=None, bias_constrain=None, complete=False, args=None, prop_list=None):
        """
        ilb: the lower bound for the input variables
        iub: the upper bound for the input variables
        """
        self.size = prop.get_input_size()
        #print(self.size)
        self.prop = prop
        self.prop_list = prop_list
        self.ilb = prop.input_lb
        self.iub = prop.input_ub
        self.eps = torch.max(self.iub - self.ilb) / 2.0
        self.complete = complete
        # A map that keeps tracks of the scaling factor of the perturbation
        # bound for each index. Currently the perturbation bound is only defined
        # for the final two layers.
        self.perturbation_scaling = {}
        self.args = args

        # Following fields are used for complete verification
        self.complete = complete
        self.map_for_noise_indices = {}
        
        if self.size == 784:
            self.shape = (1, 28, 28)
        elif self.size == 3072:
            self.shape = (3, 32, 32)
        elif self.size == 2:
            # For debug network
            self.shape = (1, 1, 2)


        self.ilb = self.ilb.to(device)
        self.iub = self.iub.to(device)

        center = (self.ilb + self.iub) / 2
        self.unstable_relus = []
        self.active_relus = []
        self.inactive_relus = []        
        self.noise_ind = self.get_noise_indices()
        cof = ((self.iub - self.ilb) / 2 * torch.eye(self.size))[self.noise_ind]
        #print(self.size, self.iub.shape, cof.shape, noise_ind)
        self.centers = []
        self.cofs = []
        self.linear_centers = []
        self.linear_coefs = []

        # In this case we don't multiply the constr mat with
        # the final layer weight matrix
        self.final_layer_without_constr_center = None
        self.final_layer_without_constr_coef = None
        
        self.targeted = prop.targeted
        self.targeted_centers = None
        self.targeted_coef = None

        self.set_zono(center, cof)


    
    def populate_baseline_verifier_result(self, args=None):
        final_lb = self.compute_lb()
        final_ub = self.compute_ub()
        if self.targeted:
            target_ubs = self.compute_target_ubs()
        else:
            target_ubs = None
        if args is None or args.all_layer_sub is False:
            layer_lbs, layer_ubs = self.get_all_linear_bounds_wt_constraints()            
        else:
            layer_lbs, layer_ubs = self.get_all_bounds_wt_constraints()    

        layer_lbs.append(self.ilb)
        layer_ubs.append(self.iub)
        
        # print(f'layer lbs: {layer_lbs[2]}')
        # print(f'layer ubs: {layer_ubs[2]}')
        # print(self.final_layer_without_constr_center)
        # print(self.final_layer_without_constr_coef)
        # print(torch.sum(torch.abs(self.final_layer_without_constr_coef), dim=0))
        # print(self.get_all_bounds()[0])
        # print(self.get_all_bounds()[1])
        # print(f'final centers: {self.centers[-1]}')
        # print(f'final coefs: {self.cofs[-1]}')
        # print(f'final coefs sum: {torch.sum(torch.abs(self.cofs[-1]), dim=0)}')
        coef, center = self.final_coef_center()
        return BaselineVerifierRes(input=self.prop.input, layer_lbs=layer_lbs, layer_ubs=layer_ubs, final_lb=final_lb, final_ub = final_ub,
                                   zono_center=center, zono_coef=coef, target_ubs=target_ubs, target_centers = self.targeted_centers, 
                                   target_coefs = self.targeted_coef, noise_ind = self.noise_ind, eps = self.eps)


    def get_noise_indices(self):
        num_eps = 1e-7
        noise_ind = torch.where(self.iub > (self.ilb + num_eps))
        if noise_ind[0].size() == 0:
            # add one dummy index in case there is no perturbation
            noise_ind = torch.tensor([0]).to(device)
        for i in range(len(noise_ind[0])):
            self.map_for_noise_indices[i] = noise_ind[0][i].item()
        return noise_ind

    def final_coef_center(self):
        center = self.centers[-1]
        coef = self.cofs[-1]
        return coef, center

    def compute_target_ubs(self):
        target_ubs = []
        for i in range(10):
            cof = self.targeted_coef[i]
            center = self.targeted_centers[i]
            cof_abs = torch.sum(torch.abs(cof), dim=0)
            ub = center + cof_abs
            target_ubs.append(ub)
        return target_ubs

    def compute_lb(self, adv_label=None, complete=False, center=None, cof=None):
        """
        return the lower bound for the variables of the current layer
        """
        if center is None or cof is None:
            center = self.centers[-1]
            cof = self.cofs[-1]
        
        if complete:
            cof = cof[:, adv_label]
            cof_abs = torch.sum(torch.abs(cof), dim=0)
            lb = center[adv_label] - cof_abs
            sz = len(self.ilb)
            signs = (cof[:sz] > 0).to(device)
            if self.prop.is_conjunctive():
                lb = torch.min(lb)
            else:
                lb = torch.max(lb)
            return lb, True, None
        else:
            cof_abs = torch.sum(torch.abs(cof), dim=0)
            #print(center, cof_abs)
            lb = center - cof_abs
            return lb

    def compute_ub(self, test=True):
        """
        return the upper bound for the variables of the current layer
        """
        center = self.centers[-1]
        cof = self.cofs[-1]

        cof_abs = torch.sum(torch.abs(cof), dim=0)

        ub = center + cof_abs

        return ub

    def bound(self):
        # This can be little faster by reusing the computation
        center = self.centers[-1]
        cof = self.cofs[-1]

        cof_abs = torch.sum(torch.abs(cof), dim=0)

        lb = center - cof_abs
        ub = center + cof_abs

        return lb, ub

    def get_zono(self):
        return self.centers[-1], self.cofs[-1]

    def set_zono(self, center, cof):
        self.centers.append(center)
        self.cofs.append(cof)

    def set_linear_zono(self, center, coef):
        self.linear_centers.append(center)
        self.linear_coefs.append(coef)

    def get_all_linear_bounds_wt_constraints(self):
        lbs = []
        ubs = []

        for i in range(len(self.linear_centers)):
            center = self.linear_centers[i]
            coef = self.linear_coefs[i]

            coef_abs = torch.sum(torch.abs(coef), dim=0)

            lb = center - coef_abs
            ub = center + coef_abs

            lbs.append(lb)
            ubs.append(ub)

        # update bounds without constraints.
        lbs.pop()
        ubs.pop()
        coef_abs = torch.sum(torch.abs(self.final_layer_without_constr_coef), dim=0)
        lb = self.final_layer_without_constr_center - coef_abs
        ub = self.final_layer_without_constr_center + coef_abs
        lbs.append(lb)
        ubs.append(ub)
        return lbs, ubs
    
    def get_all_bounds_wt_constraints(self):
        lbs, ubs = self.get_all_bounds()
        lbs, ubs = lbs[1:], ubs[1:]
        lbs.pop()
        ubs.pop()
        coef_abs = torch.sum(torch.abs(self.final_layer_without_constr_coef), dim=0)
        lb = self.final_layer_without_constr_center - coef_abs
        ub = self.final_layer_without_constr_center + coef_abs
        lbs.append(lb)
        ubs.append(ub)
        return lbs, ubs


    def get_all_bounds(self):
        lbs = []
        ubs = []

        for i in range(len(self.centers)):
            center = self.centers[i]
            cof = self.cofs[i]

            cof_abs = torch.sum(torch.abs(cof), dim=0)

            lb = center - cof_abs
            ub = center + cof_abs

            lbs.append(lb)
            ubs.append(ub)

        return lbs, ubs
    


    def get_layer_bound(self, index):
        lbs, ubs = self.get_all_bounds()
        try:
            return lbs[index], ubs[index]
        except:
            raise ValueError("Index out of bound")

    def get_active_relu_list(self):
        return self.active_relus

    def get_inactive_relu_list(self):
        return self.inactive_relus

    # Find the scaling factor to scale perturbation bound.
    def get_perturbation_scaling(self, layer_index):
        if layer_index not in [-1, -2]:
            raise ValueError("Perturbation scaling is not implemented for any layer other than last two layers")
        if layer_index not in self.perturbation_scaling.keys():
            return None
        else:
            return self.perturbation_scaling[layer_index]


    def verify_property_with_pruned_layer(self, pruned_final_layer, adv_label, complete):
        prev_center, prev_coefficent = self.centers[-2], self.cofs[-2]
        weight = pruned_final_layer.weight.T
        bias = pruned_final_layer.bias
        weight = weight @ self.prop.output_constr_mat()
        bias = bias @ self.prop.output_constr_mat() + self.prop.output_constr_const()
        center = prev_center @ weight + bias
        cof = prev_coefficent @ weight
        lb, _, _ = self.compute_lb(adv_label=adv_label, complete=complete, center=center, cof=cof)
        if lb is not None and torch.all(lb >= 0):
            return True
        else:
            return False


    def handle_normalization(self, layer):
        '''
        only change the lower/upper bound of the input variables
        '''
        return
        # mean = layer.mean.view((1))
        # sigma = layer.sigma.view((1))
        #
        # prev_cent, prev_cof = self.get_zono()
        #
        # center = (prev_cent - mean) / sigma
        # cof = prev_cof / sigma
        #
        # self.set_zono(center, cof)
        #
        # return self

    def handle_addition(self, layer, last_layer=False):
        """
        handle addition layer
        """
        bias = layer.bias
        if last_layer:
            bias = bias @ self.prop.output_constr_mat()

        prev_cent, prev_cof = self.get_zono()

        center = prev_cent + bias
        cof = prev_cof

        self.set_zono(center, cof)
        return self

    def populate_result_list(self):
        raise NotImplementedError(f"The DeepPoly implementation does not support batches")

    # Populate the scaling factor for perturbation for different
    # index.
    def populate_perturbation_scaling_factor(self, last_layer_wt, output_specification_mat):
        if output_specification_mat is None:
            self.perturbation_scaling[-1] = None
        else:
            # self.perturbation_scaling[-1] = torch.max(torch.norm(output_specification_mat, dim=0))
            self.perturbation_scaling[-1] = 1.0
        if last_layer_wt is None:
            self.perturbation_scaling[-2] = None
        else:
            self.perturbation_scaling[-2] = torch.max(torch.norm(last_layer_wt, dim=0))

    def handle_linear(self, layer, last_layer=False):
        """
        handle linear layer
        """
        weight = layer.weight.T
        bias = layer.bias
        if last_layer:
            org_weight = weight
            org_bias = bias
            #print(self.prop.output_constr_mat())
            weight = weight @ self.prop.output_constr_mat()
            #print(f'original {weight.shape}')
            bias = bias @ self.prop.output_constr_mat() + self.prop.output_constr_const()
            #print(self.prop.output_constr_const())
            self.populate_perturbation_scaling_factor(weight, self.prop.output_constr_mat())
            # print("output bias", bias)
        self.shape = (1, weight.shape[1])
        self.size = weight.shape[1]

        prev_cent, prev_cof = self.get_zono()
        if self.targeted:
            if last_layer:
                self.targeted_centers = []
                self.targeted_coef = []
                for i in range(10):
                    constr_mat_i = create_out_constr_matrix(torch.tensor(i))
                    tar_weight = org_weight @ constr_mat_i
                    #print(tar_weight.shape)
                    tar_bias = org_bias @ constr_mat_i
                    tar_center = prev_cent @ tar_weight + tar_bias
                    tar_cof = prev_cof @ tar_weight
                    #print(f'test_center: {tar_center.shape}')
                    #print(f'test_coef: {tar_cof.shape}')
                    self.targeted_centers.append(tar_center)
                    self.targeted_coef.append(tar_cof)
        
        center = prev_cent @ weight + bias
        cof = prev_cof @ weight


        if last_layer:
            self.final_layer_without_constr_center = prev_cent @ org_weight + org_bias
            self.final_layer_without_constr_coef = prev_cof @ org_weight

        self.set_zono(center, cof)
        self.set_linear_zono(center=center, coef=cof)
        return self

    def handle_conv2d(self, layer):
        """
        handle conv2d layer
        first transform it to linear matrix
        then use absmul func
        """
        weight = layer.weight
        bias = layer.bias
        num_kernel = weight.shape[0]

        k_h, k_w = layer.kernel_size
        s_h, s_w = layer.stride
        p_h, p_w = layer.padding

        shape = self.shape

        input_h, input_w = shape[1:]

        ### ref. https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d ###
        output_h = int((input_h + 2 * p_h - k_h) / s_h + 1)
        output_w = int((input_w + 2 * p_w - k_w) / s_w + 1)

        self.shape = (num_kernel, output_h, output_w)
        self.size = num_kernel * output_h * output_w

        prev_cent, prev_cof = self.get_zono()

        prev_cent = prev_cent.reshape(1, shape[0], input_h, input_w)
        prev_cof = prev_cof.reshape(-1, shape[0], input_h, input_w)

        center = F.conv2d(prev_cent, weight, padding=layer.padding, stride=layer.stride, bias=bias).flatten()

        num_eps = prev_cof.shape[0]
        cof = F.conv2d(prev_cof, weight, padding=layer.padding, stride=layer.stride).reshape(num_eps, -1)

        self.set_zono(center, cof)
        self.set_linear_zono(center=center, coef=cof)

        return self

    def handle_relu(self, layer, optimize=True, relu_mask=None):
        """
        handle relu func
        """
        size = self.size

        prev_cent, prev_cof = self.get_zono()
        lb, ub = self.bound()

        layer_no = len(self.unstable_relus)
        self.unstable_relus.append(torch.where(torch.logical_and(ub >= 0, lb <= 0))[0].tolist())

        num_eps = 1e-7
        lmbda = torch.div(ub, ub - lb + num_eps)
        mu = -(lb / 2) * lmbda

        active_relus = (lb > 0)
        passive_relus = (ub <= 0)
        ambiguous_relus = (~active_relus) & (~passive_relus)

        self.active_relus.append(torch.where(active_relus)[0].tolist())
        self.inactive_relus.append(torch.where(passive_relus)[0].tolist())

        if self.complete:
            # Store the map from (unstable relu index -> index of the added noise)
            prev_error_terms = prev_cof.shape[0]
            unstable_relu_indices = torch.where(ambiguous_relus)[0]

            for i, index in enumerate(unstable_relu_indices):
                index_of_unstable_relu = prev_error_terms + i
                self.map_for_noise_indices[index_of_unstable_relu] = (layer_no, index.item())

            # Figure out how these should be used
            c1_decision = torch.zeros(size, dtype=torch.bool)
            c2_decision = torch.zeros(size, dtype=torch.bool)

            if relu_mask is not None:
                for relu in relu_mask.keys():
                    if relu[0] == layer_no:
                        if ambiguous_relus[relu[1]]:
                            if relu_mask[relu] == 1:
                                c1_decision[relu[1]] = 1
                            elif relu_mask[relu] == -1:
                                c2_decision[relu[1]] = 1

            ambiguous_relus = ambiguous_relus & (~c1_decision) & (~c2_decision)
            c1_mu = c1_decision*ub/2
            c2_mu = c2_decision*lb/2

        mult_fact = torch.ones(size, dtype=torch.bool)
        # mult_fact should have 1 at active relus, 0 at passive relus and lambda at ambiguous_relus
        mult_fact = mult_fact * (active_relus + ambiguous_relus * lmbda)

        if self.complete:
            new_noise_cofs = torch.diag(mu * ambiguous_relus + c1_mu + c2_mu)
        else:
            new_noise_cofs = torch.diag(mu * ambiguous_relus)

        non_empty_mask = new_noise_cofs.abs().sum(dim=0).bool()
        new_noise_cofs = new_noise_cofs[non_empty_mask, :]
        cof = torch.cat([mult_fact * prev_cof, new_noise_cofs])
        if self.complete:
            center = prev_cent * active_relus + (lmbda * prev_cent + mu) * ambiguous_relus + c1_mu + c2_mu
        else:
            center = prev_cent * active_relus + (lmbda * prev_cent + mu) * ambiguous_relus

        self.set_zono(center, cof)
        #print(f'test: {center, cof}')
        return self
    
    def handle_sigmoid(self, layer):
        raise NotImplementedError('Sigmoid is not implemented for DeepZ')

    def handle_tanh(self, layer):
        raise NotImplementedError('Tanh is not implemented for DeepZ')

    def verify_robustness(self, y, true_label):
        pass
