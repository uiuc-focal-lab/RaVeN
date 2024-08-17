import torch
import torch.nn.functional as F
from src.common.network import LayerType, Layer
from src.baseline_res import BaselineVerifierRes


class DeepPolyStruct:
    def __init__(self, lb_coef, lb_bias, ub_coef, ub_bias) -> None:
        self.lb_coef = lb_coef
        self.lb_bias = lb_bias
        self.ub_coef = ub_coef
        self.ub_bias = ub_bias
    
    def populate(self, lb_coef, lb_bias, ub_coef, ub_bias):
        self.lb_coef = lb_coef
        self.lb_bias = lb_bias
        self.ub_coef = ub_coef
        self.ub_bias = ub_bias

"""
This implementation taken from https://github.com/eth-sri/eran.
"""
class DeepPolyTransformerOptimized:
    def __init__(self, prop, complete=False, device=None, args=None):
        self.lbs = []
        self.ubs = []
        self.prop = prop
        self.layers = []
        self.complete = complete
        # keep track of the final lb coef and bias
        # this will be used in baseline LP formulation.
        self.final_lb_coef = None
        self.final_lb_bias = None
        # Track input bounds and input size.
        self.size = prop.get_input_size()
        self.prop = prop
        self.ilb = prop.input_lb
        self.iub = prop.input_ub
        self.input_shape = None
        self.eps = torch.max(self.iub - self.ilb) / 2.0
        self.last_conv_diff_struct = None
        self.args = args
        # Tracking shpes for supporting conv layers.
        self.shapes = []
        if self.size == 784:
            self.shape = (1, 28, 28)
        elif self.size == 3072:
            self.shape = (3, 32, 32)
        elif self.size == 2:
            # For debug network
            self.shape = (1, 1, 2)
        elif self.size == 87:
            self.shape = (1, 1, 87)
        self.shapes.append(self.shape)

        self.device = device if device is not None else 'cpu'

    def pos_neg_weight_decomposition(self, coef):
        neg_comp = torch.where(coef < 0, coef, torch.zeros_like(coef, device=self.device))
        pos_comp = torch.where(coef >= 0, coef, torch.zeros_like(coef, device=self.device))
        return neg_comp, pos_comp
    
    def concrete_substitution(self, diff_struct, lb_layer, ub_layer):
        neg_comp_lb, pos_comp_lb = self.pos_neg_weight_decomposition(diff_struct.lb_coef)
        neg_comp_ub, pos_comp_ub = self.pos_neg_weight_decomposition(diff_struct.ub_coef)
        lb = neg_comp_lb @ ub_layer + pos_comp_lb @ lb_layer + diff_struct.lb_bias
        ub = neg_comp_ub @ lb_layer + pos_comp_ub @ ub_layer + diff_struct.ub_bias 
        #print(ub - lb)      
        assert torch.all(lb <= ub + 1e-6)
        return lb, ub    
    
    def analyze_linear(self, diff_struct, layer):
        # layer = self.layers[-1]
        lb_coef = diff_struct.lb_coef.matmul(layer.weight)
        lb_bias = diff_struct.lb_bias + diff_struct.lb_coef.matmul(layer.bias)
        ub_coef = diff_struct.ub_coef.matmul(layer.weight)
        ub_bias = diff_struct.ub_bias + diff_struct.ub_coef.matmul(layer.bias)

        diff_struct = DeepPolyStruct(lb_bias=lb_bias, lb_coef=lb_coef, 
                                     ub_bias=ub_bias, ub_coef=ub_coef)
        return diff_struct
    
    def analyze_conv(self, diff_struct, layer, layer_idx):
        # layer = self.layers[-1]
        conv_weight=layer.weight
        conv_bias=layer.bias
        preconv_shape= self.shapes[layer_idx]
        postconv_shape= self.shapes[layer_idx + 1]
        stride= layer.stride 
        padding= layer.padding 
        groups=1 
        dilation=(1, 1)
        kernel_hw = conv_weight.shape[-2:]

        h_padding = (preconv_shape[1] + 2 * padding[0] - 1 - dilation[0] * (kernel_hw[0] - 1)) % stride[0]
        w_padding = (preconv_shape[2] + 2 * padding[1] - 1 - dilation[1] * (kernel_hw[1] - 1)) % stride[1]
        output_padding = (h_padding, w_padding)

        coef_shape = diff_struct.lb_coef.shape
        lb_coef = diff_struct.lb_coef.view((coef_shape[0], *postconv_shape))
        ub_coef = diff_struct.ub_coef.view((coef_shape[0], *postconv_shape))
        lb_bias = diff_struct.lb_bias + (lb_coef.sum((2, 3)) * conv_bias).sum(1)
        ub_bias = diff_struct.ub_bias + (ub_coef.sum((2, 3)) * conv_bias).sum(1)

        lb_coef = F.conv_transpose2d(lb_coef, conv_weight, None, stride, padding,
                            output_padding, groups, dilation)
        ub_coef = F.conv_transpose2d(ub_coef, conv_weight, None, stride, padding,
                            output_padding, groups, dilation)
        
        lb_coef = lb_coef.view((coef_shape[0], -1))
        ub_coef = ub_coef.view((coef_shape[0], -1))
        
        diff_struct = DeepPolyStruct(lb_bias=lb_bias, lb_coef=lb_coef, ub_bias=ub_bias, ub_coef=ub_coef)
        return diff_struct

    def analyze_relu(self, diff_struct, layer_idx):
        lb_layer, ub_layer = self.lbs[layer_idx-1], self.ubs[layer_idx-1]
        active = (lb_layer >= 0)
        passive = (ub_layer <= 0)
        unsettled = ~(active) & ~(passive)

        lambda_lb = torch.zeros(lb_layer.size(), device=self.device)
        lambda_ub = torch.zeros(lb_layer.size(), device=self.device)
        mu_ub = torch.zeros(lb_layer.size(), device=self.device) 

        # input is active
        lambda_lb = torch.where(active, torch.ones(lb_layer.size(), device=self.device), lambda_lb)
        lambda_ub = torch.where(active, torch.ones(lb_layer.size(), device=self.device), lambda_ub)
        # input is unsettled
        temp = torch.where(ub_layer < -lb_layer, torch.zeros(lb_layer.size(), device=self.device), torch.ones(lb_layer.size(), device=self.device))
        lambda_lb = torch.where(unsettled, temp, lambda_lb)
        lambda_ub = torch.where(unsettled, ub_layer/(ub_layer - lb_layer + 1e-15), lambda_ub)
        mu_ub = torch.where(unsettled, -(ub_layer * lb_layer) / (ub_layer - lb_layer + 1e-15), mu_ub)        

        neg_comp_lb, pos_comp_lb = self.pos_neg_weight_decomposition(diff_struct.lb_coef)
        neg_comp_ub, pos_comp_ub = self.pos_neg_weight_decomposition(diff_struct.ub_coef)
        
        lb_coef = neg_comp_lb * lambda_ub + pos_comp_lb * lambda_lb
        ub_coef = neg_comp_ub * lambda_lb + pos_comp_ub * lambda_ub
        lb_bias = diff_struct.lb_bias + neg_comp_lb @ mu_ub 
        ub_bias = diff_struct.ub_bias + pos_comp_ub @ mu_ub
        diff_struct = DeepPolyStruct(lb_bias=lb_bias, lb_coef=lb_coef, ub_bias=ub_bias, ub_coef=ub_coef)
        return diff_struct
    
    def analyze_sigmoid(self, diff_struct, layer_idx):
        lb_layer, ub_layer = self.lbs[layer_idx-1], self.ubs[layer_idx-1]
        sigmoid_lb, sigmoid_ub = torch.sigmoid(lb_layer), torch.sigmoid(ub_layer)
        lmbda = torch.where(lb_layer < ub_layer, (sigmoid_ub - sigmoid_lb) / (ub_layer - lb_layer + 1e-15),  sigmoid_lb * (1 - sigmoid_lb))
        lmbda_ = torch.min(sigmoid_ub * (1 - sigmoid_ub), sigmoid_lb * (1 - sigmoid_lb))
        lambda_lb = torch.where(lb_layer > 0, lmbda, lmbda_)
        mu_lb = torch.where(lb_layer > 0, sigmoid_lb - torch.mul(lmbda, lb_layer),  sigmoid_lb - torch.mul(lmbda_, lb_layer))
        lambda_ub = torch.where(ub_layer < 0, lmbda, lmbda_)
        mu_ub =  torch.where(ub_layer < 0, sigmoid_ub - torch.mul(lmbda, ub_layer),  sigmoid_ub - torch.mul(lmbda_, lb_layer))

        neg_comp_lb, pos_comp_lb = self.pos_neg_weight_decomposition(diff_struct.lb_coef)
        neg_comp_ub, pos_comp_ub = self.pos_neg_weight_decomposition(diff_struct.ub_coef)
        
        lb_coef = neg_comp_lb * lambda_ub + pos_comp_lb * lambda_lb
        ub_coef = neg_comp_ub * lambda_lb + pos_comp_ub * lambda_ub
        lb_bias = diff_struct.lb_bias + neg_comp_lb @ mu_ub + pos_comp_lb @ mu_lb
        ub_bias = diff_struct.ub_bias + pos_comp_ub @ mu_ub + neg_comp_ub @ mu_lb
        diff_struct = DeepPolyStruct(lb_bias=lb_bias, lb_coef=lb_coef, ub_bias=ub_bias, ub_coef=ub_coef)

        return diff_struct
    
    def analyze_tanh(self, diff_struct, layer_idx):
        lb_layer, ub_layer = self.lbs[layer_idx-1], self.ubs[layer_idx-1]

        # lmb = (su - sl) / (u - l) if l < u else 1 - sl * sl
        # lmb_ = torch.min(1 - su * su, 1 - sl * sl)
        tanh_lb, tanh_ub = torch.tanh(lb_layer), torch.tanh(ub_layer)
        lmbda = torch.where(lb_layer < ub_layer, (tanh_ub - tanh_lb) / (ub_layer - lb_layer + 1e-15),  1 - tanh_lb * tanh_lb)
        lmbda_ = torch.min(1 - tanh_ub * tanh_ub, 1 - tanh_lb * tanh_lb)

        lambda_lb = torch.where(lb_layer > 0, lmbda, lmbda_)
        mu_lb = torch.where(lb_layer > 0, tanh_lb - torch.mul(lmbda, lb_layer),  tanh_lb - torch.mul(lmbda_, lb_layer))
        
        lambda_ub = torch.where(ub_layer < 0, lmbda, lmbda_)
        mu_ub =  torch.where(ub_layer < 0, tanh_ub - torch.mul(lmbda, ub_layer),  tanh_ub - torch.mul(lmbda_, lb_layer))

        neg_comp_lb, pos_comp_lb = self.pos_neg_weight_decomposition(diff_struct.lb_coef)
        neg_comp_ub, pos_comp_ub = self.pos_neg_weight_decomposition(diff_struct.ub_coef)
        
        lb_coef = neg_comp_lb * lambda_ub + pos_comp_lb * lambda_lb
        ub_coef = neg_comp_ub * lambda_lb + pos_comp_ub * lambda_ub
        lb_bias = diff_struct.lb_bias + neg_comp_lb @ mu_ub + pos_comp_lb @ mu_lb
        ub_bias = diff_struct.ub_bias + pos_comp_ub @ mu_ub + neg_comp_ub @ mu_lb
        diff_struct = DeepPolyStruct(lb_bias=lb_bias, lb_coef=lb_coef, ub_bias=ub_bias, ub_coef=ub_coef)

        return diff_struct


    def get_layer_size(self, layer_index):
        layer = self.layers[layer_index]
        shapes = self.shapes[layer_index + 1]
        if isinstance(shapes, int):
            return shapes
        else:
            return (shapes[0] * shapes[1] * shapes[2]) 
                    

    # Tracks the shape after the transform corresponding to the layer
    # is applied.
    def update_shape(self, layer):
        if layer.type is LayerType.Linear:
            if len(self.shapes) == 1:
                in_shape = self.shapes.pop()
                self.shapes.append(in_shape[0] * in_shape[1] * in_shape[2])
            self.shapes.append(layer.weight.shape[0])
        elif layer.type is LayerType.Conv2D:
            weight = layer.weight
            num_kernel = weight.shape[0]

            k_h, k_w = layer.kernel_size
            s_h, s_w = layer.stride
            p_h, p_w = layer.padding

            shape = self.shapes[-1]
            input_h, input_w = shape[1:]
            ### ref. https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d ###
            output_h = int((input_h + 2 * p_h - k_h) / s_h + 1)
            output_w = int((input_w + 2 * p_w - k_w) / s_w + 1)
            self.shapes.append((num_kernel, output_h, output_w))
        elif layer.type is LayerType.ReLU:
            if len(self.shapes) == 0:
                raise ValueError("Relu layer should not come at first")
            self.shapes.append(self.shapes[-1])
        elif layer.type is LayerType.Sigmoid:
            if len(self.shapes) == 0:
                raise ValueError("Sigmoid layer should not come at first")
            self.shapes.append(self.shapes[-1])
        elif layer.type is LayerType.TanH:
            if len(self.shapes) == 0:
                raise ValueError("TanH layer should not come at first")
            self.shapes.append(self.shapes[-1])


    def print_shapes(self, diff_struct):
        print(f'lb coef shape {diff_struct.lb_coef.shape}')
        print(f'lb bias shpae {diff_struct.lb_bias.shape}')
        print(f'ub coef shape {diff_struct.ub_coef.shape}')
        print(f'ub bias shpae {diff_struct.ub_bias.shape}')

    def back_propagation(self):
        layers_length = len(self.layers)
        diff_struct = None
        lb = None
        ub = None
        for i in reversed(range(layers_length)):
            layer = self.layers[i]
            # concretize the bounds.
            linear_types = [LayerType.Linear, LayerType.Conv2D]
            
            if diff_struct is not None and layer.type in linear_types:
                curr_lb, curr_ub = self.concrete_substitution(diff_struct=diff_struct, 
                                                              lb_layer=self.lbs[i], ub_layer=self.ubs[i])
                lb = curr_lb if lb is None else torch.max(lb, curr_lb)
                ub = curr_ub if ub is None else torch.min(ub, curr_ub)
            
            if diff_struct is None:
                layer_size = self.get_layer_size(layer_index=i)
                lb_coef = torch.eye(n=layer_size, device=self.device)
                lb_bias = torch.zeros(layer_size, device=self.device)
                ub_coef = torch.eye(n=layer_size, device=self.device)
                ub_bias = torch.zeros(layer_size, device=self.device)
                diff_struct = DeepPolyStruct(lb_bias=lb_bias, lb_coef=lb_coef,
                                             ub_bias=ub_bias, ub_coef=ub_coef)
            
            if layer.type is LayerType.Linear:
                diff_struct = self.analyze_linear(diff_struct=diff_struct, layer=layer)
            elif layer.type is LayerType.Conv2D:
                diff_struct = self.analyze_conv(diff_struct=diff_struct, layer=layer, layer_idx=i)
            elif layer.type is LayerType.ReLU:
                diff_struct = self.analyze_relu(diff_struct=diff_struct, layer_idx=i)
            elif layer.type is LayerType.Sigmoid:
                diff_struct = self.analyze_sigmoid(diff_struct=diff_struct, layer_idx=i)
            elif layer.type is LayerType.TanH:
                diff_struct = self.analyze_tanh(diff_struct=diff_struct, layer_idx=i)
            else:
                raise ValueError(f'Unsupported Layer {layer.type}')
 
        curr_lb, curr_ub = self.concrete_substitution(diff_struct=diff_struct, 
                                                    lb_layer=self.ilb, ub_layer=self.iub)
        if self.args is not None and self.args.fold_conv_layers is True:
            if self.layers[-1].type is LayerType.Conv2D:
                self.last_conv_diff_struct = diff_struct

        lb = curr_lb if lb is None else torch.max(lb, curr_lb)
        ub = curr_ub if ub is None else torch.min(ub, curr_ub)
        self.lbs.append(lb)
        self.ubs.append(ub)
        return diff_struct
    
    def handle_linear(self, layer, last_layer=False):
        self.layers.append(layer)
        self.update_shape(layer=layer)
        return self.back_propagation()


    def handle_conv2d(self, layer):
        self.layers.append(layer)
        self.update_shape(layer=layer)
        return self.back_propagation()

    def handle_relu(self, layer):
        self.layers.append(layer)
        self.update_shape(layer=layer)
        return self.back_propagation()

    def handle_sigmoid(self, layer):
        self.layers.append(layer)
        self.update_shape(layer=layer)
        return self.back_propagation()
        
    def handle_tanh(self, layer):
        self.layers.append(layer)
        self.update_shape(layer=layer)
        return self.back_propagation()


    def remove_non_affine_bounds(self):
        new_lbs = []
        new_ubs = []
        for i, layer in self.layers:
            if layer.type in [LayerType.Conv2D, LayerType.Linear]:
                new_lbs.append(self.lbs[i])
                new_ubs.append(self.ubs[i])
        self.lbs = new_lbs
        self.ubs = new_ubs
                

    def populate_baseline_verifier_result(self, args=None):
        weight = self.prop.output_constr_mat().T
        bias = self.prop.output_constr_const()
        if isinstance(bias, int):
            bias = torch.tensor([float(bias)], device=self.device)
        if bias.shape[0] != weight.shape[0]:
            bias = torch.zeros(weight.shape[0], device=self.device)
        layer = Layer(weight=weight, bias=bias, type=LayerType.Linear)
        diff_struct = self.handle_linear(layer=layer)
        if args is not None and args.all_layer_sub is False: 
            self.remove_non_affine_bounds()
        final_lb = self.lbs.pop()
        final_ub = self.ubs.pop()

        self.lbs.append(self.ilb)
        self.ubs.append(self.iub)
        # print(f'lbs: {self.lbs[-1]}')
        # print(f'ubs: {self.ubs[-1]}')
        # print(f'diff: {self.ubs[-1] - self.lbs[-1]}')

        lb_debug, _ = self.concrete_substitution(diff_struct=diff_struct, lb_layer=self.ilb, ub_layer=self.iub)
        return BaselineVerifierRes(input=self.prop.input, layer_lbs=self.lbs, layer_ubs=self.ubs, final_lb=final_lb, 
                                   final_ub = final_ub, lb_bias=diff_struct.lb_bias, lb_coef=diff_struct.lb_coef, 
                                   eps=self.eps, last_conv_diff_struct=self.last_conv_diff_struct)



# 1. Implement DeepPoly - with tanh and sigmoid.
# 2. Implement conv layer supressions.
# 3. Get good results on CIFAR-10 with diffpoly. 
# 4. Rotation - X