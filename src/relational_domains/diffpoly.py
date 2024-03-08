import torch
import torch.nn.functional as F
from src.common.network import LayerType
from src.util import compute_input_shapes


# Basic deep poly struct.  
class BasicDeepPolyStruct:
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

# DiffPoly struct storing 
class DiffPropStruct:
    def __init__(self) -> None:
        self.delta_lb_coef = None
        self.delta_lb_bias = None
        self.delta_ub_coef = None
        self.delta_ub_bias = None
        self.delta_lb_input1_coef = None
        self.delta_ub_input1_coef = None
        self.delta_lb_input2_coef = None
        self.delta_ub_input2_coef = None
        self.lb_coef_input1 = None
        self.lb_coef_input2 = None
        self.lb_bias_input1 = None
        self.lb_bias_input2 = None
        self.ub_coef_input1 = None
        self.ub_coef_input2 = None
        self.ub_bias_input1 = None
        self.ub_bias_input2 = None        
    
    def populate(self, delta_lb_coef, delta_lb_bias, delta_ub_coef, delta_ub_bias,
                 delta_lb_input1_coef, delta_ub_input1_coef, delta_lb_input2_coef,
                 delta_ub_input2_coef, lb_coef_input1, lb_coef_input2, lb_bias_input1,
                 lb_bias_input2, ub_coef_input1, ub_coef_input2, ub_bias_input1, ub_bias_input2) -> None:
        self.delta_lb_coef = delta_lb_coef
        self.delta_lb_bias = delta_lb_bias
        self.delta_ub_coef = delta_ub_coef
        self.delta_ub_bias = delta_ub_bias
        self.delta_lb_input1_coef = delta_lb_input1_coef
        self.delta_ub_input1_coef = delta_ub_input1_coef
        self.delta_lb_input2_coef = delta_lb_input2_coef
        self.delta_ub_input2_coef = delta_ub_input2_coef
        self.lb_coef_input1 = lb_coef_input1
        self.lb_coef_input2 = lb_coef_input2
        self.lb_bias_input1 = lb_bias_input1
        self.lb_bias_input2 = lb_bias_input2
        self.ub_coef_input1 = ub_coef_input1
        self.ub_coef_input2 = ub_coef_input2
        self.ub_bias_input1 = ub_bias_input1
        self.ub_bias_input2 = ub_bias_input2
 

class DiffPoly:
    def __init__(self, input1, input2, net, lb_input1, ub_input1,
                lb_input2, ub_input2, device='', noise_ind = None,
                eps = None, monotone = False, monotone_prop = 0, use_all_layers=False, lightweight_diffpoly=False) -> None:
        self.input1 = input1
        self.input2 = input2
        if self.input1.shape[0] == 784:
            self.input_shape = (1, 28, 28)
        elif self.input1.shape[0] == 3072:
            self.input_shape = (3, 32, 32)
        elif self.input1.shape[0] == 2:
            self.input_shape = (1, 1, 2)
        elif self.input1.shape[0] == 12:
            self.input_shape = (1, 1, 12)
        elif self.input1.shape[0] == 87:
            self.input_shape = (1, 1, 87)
        else:
            raise ValueError(f"Unrecognised input shape {self.input_shape}")
        self.net = net
        self.lb_input1 = lb_input1
        self.ub_input1 = ub_input1
        self.lb_input2 = lb_input2
        self.ub_input2 = ub_input2
        self.shapes = compute_input_shapes(net=self.net, input_shape=self.input_shape)
        self.diff = input1 - input2
        self.linear_conv_layer_indices = []
        self.device = device
        self.noise_ind  = noise_ind
        self.eps = eps
        self.monotone = monotone
        self.use_all_layers = use_all_layers
        self.lightweight_diffpoly = lightweight_diffpoly
        self.monotone_prop = monotone_prop
        #print(self.monotone_prop)

    # Bias cancels out (Ax + b - Ay - b) = A(x - y) = A * delta 
    def handle_linear(self, linear_wt, bias, back_prop_struct):
        if back_prop_struct.delta_lb_input1_coef is not None:
            delta_lb_bias = back_prop_struct.delta_lb_bias + back_prop_struct.delta_lb_input1_coef.matmul(bias)
            delta_lb_bias = delta_lb_bias + back_prop_struct.delta_lb_input2_coef.matmul(bias)
            delta_ub_bias = back_prop_struct.delta_ub_bias + back_prop_struct.delta_ub_input1_coef.matmul(bias)
            delta_ub_bias = delta_ub_bias + back_prop_struct.delta_ub_input2_coef.matmul(bias)
        else:
            delta_lb_bias = back_prop_struct.delta_lb_bias
            delta_ub_bias = back_prop_struct.delta_ub_bias

        if back_prop_struct.lb_coef_input1 is not None:       
            lb_bias_input1 = back_prop_struct.lb_bias_input1 + back_prop_struct.lb_coef_input1.matmul(bias)
            lb_bias_input2 = back_prop_struct.lb_bias_input2 + back_prop_struct.lb_coef_input2.matmul(bias)
            ub_bias_input1 = back_prop_struct.ub_bias_input1 + back_prop_struct.ub_coef_input1.matmul(bias)
            ub_bias_input2 = back_prop_struct.ub_bias_input2 + back_prop_struct.ub_coef_input2.matmul(bias)
        else:
            lb_bias_input1 = None
            lb_bias_input2 = None
            ub_bias_input1 = None
            ub_bias_input2 = None



        delta_lb_coef = back_prop_struct.delta_lb_coef.matmul(linear_wt)
        delta_ub_coef = back_prop_struct.delta_ub_coef.matmul(linear_wt)
        if back_prop_struct.delta_lb_input1_coef is not None:
            delta_lb_input1_coef = back_prop_struct.delta_lb_input1_coef.matmul(linear_wt)
            delta_ub_input1_coef = back_prop_struct.delta_ub_input1_coef.matmul(linear_wt)
            delta_lb_input2_coef = back_prop_struct.delta_lb_input2_coef.matmul(linear_wt)
            delta_ub_input2_coef = back_prop_struct.delta_ub_input2_coef.matmul(linear_wt)
        else:            
            delta_lb_input1_coef = None
            delta_ub_input1_coef = None
            delta_lb_input2_coef = None
            delta_ub_input2_coef = None
        if back_prop_struct.lb_coef_input1 is not None:
            lb_coef_input1 = back_prop_struct.lb_coef_input1.matmul(linear_wt)
            lb_coef_input2 = back_prop_struct.lb_coef_input2.matmul(linear_wt)
            ub_coef_input1 = back_prop_struct.ub_coef_input1.matmul(linear_wt)
            ub_coef_input2 = back_prop_struct.ub_coef_input2.matmul(linear_wt)
        else:
            lb_coef_input1 = None
            lb_coef_input2 = None
            ub_coef_input1 = None
            ub_coef_input2 = None

        back_prop_struct.populate(delta_lb_coef=delta_lb_coef, delta_lb_bias=delta_lb_bias, 
                                    delta_ub_coef=delta_ub_coef, delta_ub_bias=delta_ub_bias,
                                    delta_lb_input1_coef=delta_lb_input1_coef, 
                                    delta_ub_input1_coef=delta_ub_input1_coef,
                                    delta_lb_input2_coef=delta_lb_input2_coef,
                                    delta_ub_input2_coef=delta_ub_input2_coef,
                                    lb_coef_input1=lb_coef_input1, lb_coef_input2=lb_coef_input2,
                                    ub_coef_input1=ub_coef_input1, ub_coef_input2=ub_coef_input2,
                                    lb_bias_input1=lb_bias_input1, lb_bias_input2=lb_bias_input2,
                                    ub_bias_input1=ub_bias_input1, ub_bias_input2=ub_bias_input2)

        return back_prop_struct

    # preconv shape is the shape before the convolution is applied.
    # postconv shape is the shape after the convolution is applied.
    # while back prop the delta coef shape [rows, postconv shape after flattening].
    def handle_conv(self, conv_weight, conv_bias, back_prop_struct, preconv_shape, postconv_shape,
                    stride, padding, groups=1, dilation=(1, 1)):
        kernel_hw = conv_weight.shape[-2:]
        h_padding = (preconv_shape[1] + 2 * padding[0] - 1 - dilation[0] * (kernel_hw[0] - 1)) % stride[0]
        w_padding = (preconv_shape[2] + 2 * padding[1] - 1 - dilation[1] * (kernel_hw[1] - 1)) % stride[1]
        output_padding = (h_padding, w_padding)

        coef_shape = back_prop_struct.delta_lb_coef.shape
        delta_lb_coef = back_prop_struct.delta_lb_coef.view((coef_shape[0], *postconv_shape))
        delta_ub_coef = back_prop_struct.delta_ub_coef.view((coef_shape[0], *postconv_shape))

        if back_prop_struct.delta_lb_input1_coef is not None:
            delta_lb_input1_coef = back_prop_struct.delta_lb_input1_coef.view((coef_shape[0], *postconv_shape))
            delta_ub_input1_coef = back_prop_struct.delta_ub_input1_coef.view((coef_shape[0], *postconv_shape))
            delta_lb_input2_coef = back_prop_struct.delta_lb_input2_coef.view((coef_shape[0], *postconv_shape))
            delta_ub_input2_coef = back_prop_struct.delta_ub_input2_coef.view((coef_shape[0], *postconv_shape))                        

        if back_prop_struct.lb_coef_input1 is not None:
            lb_coef_input1 = back_prop_struct.lb_coef_input1.view((coef_shape[0], *postconv_shape))
            lb_coef_input2 = back_prop_struct.lb_coef_input2.view((coef_shape[0], *postconv_shape))
            ub_coef_input1 = back_prop_struct.ub_coef_input1.view((coef_shape[0], *postconv_shape))
            ub_coef_input2 = back_prop_struct.ub_coef_input2.view((coef_shape[0], *postconv_shape))        
        
        if back_prop_struct.delta_lb_input1_coef is not None:
            delta_lb_bias = back_prop_struct.delta_lb_bias + (delta_lb_input1_coef.sum((2, 3)) * conv_bias).sum(1)
            delta_lb_bias = delta_lb_bias + (delta_lb_input2_coef.sum((2, 3)) * conv_bias).sum(1)
            delta_ub_bias = back_prop_struct.delta_ub_bias + (delta_ub_input1_coef.sum((2, 3)) * conv_bias).sum(1)
            delta_ub_bias = delta_ub_bias + (delta_ub_input2_coef.sum((2, 3)) * conv_bias).sum(1)
        else:
            delta_lb_bias = back_prop_struct.delta_lb_bias
            delta_ub_bias = back_prop_struct.delta_ub_bias

        if back_prop_struct.lb_coef_input1 is not None:
            lb_bias_input1 = back_prop_struct.lb_bias_input1 + (lb_coef_input1.sum((2, 3)) * conv_bias).sum(1)
            lb_bias_input2 = back_prop_struct.lb_bias_input2 + (lb_coef_input2.sum((2, 3)) * conv_bias).sum(1)
            ub_bias_input1 = back_prop_struct.ub_bias_input1 + (ub_coef_input1.sum((2, 3)) * conv_bias).sum(1)
            ub_bias_input2 = back_prop_struct.ub_bias_input2 + (ub_coef_input2.sum((2, 3)) * conv_bias).sum(1)



        new_delta_lb_coef = F.conv_transpose2d(delta_lb_coef, conv_weight, None, stride, padding,
                                           output_padding, groups, dilation)
        new_delta_ub_coef = F.conv_transpose2d(delta_ub_coef, conv_weight, None, stride, padding,
                                           output_padding, groups, dilation)
        if back_prop_struct.delta_lb_input1_coef is not None:
            new_delta_lb_input1_coef = F.conv_transpose2d(delta_lb_input1_coef, conv_weight, None, stride, padding,
                                            output_padding, groups, dilation)
            new_delta_ub_input1_coef = F.conv_transpose2d(delta_ub_input1_coef, conv_weight, None, stride, padding,
                                        output_padding, groups, dilation)
            new_delta_lb_input2_coef = F.conv_transpose2d(delta_lb_input2_coef, conv_weight, None, stride, padding,
                                            output_padding, groups, dilation)
            new_delta_ub_input2_coef = F.conv_transpose2d(delta_ub_input2_coef, conv_weight, None, stride, padding,
                                        output_padding, groups, dilation)
        else:
            new_delta_lb_input1_coef = None            
            new_delta_ub_input1_coef = None
            new_delta_lb_input2_coef = None
            new_delta_ub_input2_coef = None

        if back_prop_struct.lb_coef_input1 is not None:        
            lb_coef_input1 = F.conv_transpose2d(lb_coef_input1, conv_weight, None, stride, padding,
                                        output_padding, groups, dilation)
            lb_coef_input2 = F.conv_transpose2d(lb_coef_input2, conv_weight, None, stride, padding,
                                        output_padding, groups, dilation)
            ub_coef_input1 = F.conv_transpose2d(ub_coef_input1, conv_weight, None, stride, padding,
                                        output_padding, groups, dilation)
            ub_coef_input2 = F.conv_transpose2d(ub_coef_input2, conv_weight, None, stride, padding,
                                    output_padding, groups, dilation)



        new_delta_lb_coef = new_delta_lb_coef.view((coef_shape[0], -1))
        new_delta_ub_coef = new_delta_ub_coef.view((coef_shape[0], -1))

        if new_delta_lb_input1_coef is not None:
            new_delta_lb_input1_coef = new_delta_lb_input1_coef.view((coef_shape[0], -1))
            new_delta_ub_input1_coef = new_delta_ub_input1_coef.view((coef_shape[0], -1))
            new_delta_lb_input2_coef = new_delta_lb_input2_coef.view((coef_shape[0], -1))
            new_delta_ub_input2_coef = new_delta_ub_input2_coef.view((coef_shape[0], -1))
        
        if back_prop_struct.lb_coef_input1 is not None:
            lb_coef_input1 = lb_coef_input1.view((coef_shape[0], -1))
            lb_coef_input2 = lb_coef_input2.view((coef_shape[0], -1))
            ub_coef_input1 = ub_coef_input1.view((coef_shape[0], -1))
            ub_coef_input2 = ub_coef_input2.view((coef_shape[0], -1))
        else:
            lb_coef_input1 = None
            lb_coef_input2 = None
            ub_coef_input1 = None
            ub_coef_input2 = None
            lb_bias_input1 = None
            lb_bias_input2 = None
            ub_bias_input1 = None
            ub_bias_input2 = None



        back_prop_struct.populate(delta_lb_coef=new_delta_lb_coef, delta_lb_bias=delta_lb_bias, 
                            delta_ub_coef=new_delta_ub_coef, delta_ub_bias=delta_ub_bias,
                            delta_lb_input1_coef=new_delta_lb_input1_coef, 
                            delta_ub_input1_coef=new_delta_ub_input1_coef,
                            delta_lb_input2_coef=new_delta_lb_input2_coef,
                            delta_ub_input2_coef=new_delta_ub_input2_coef,
                            lb_coef_input1=lb_coef_input1, lb_coef_input2=lb_coef_input2,
                            ub_coef_input1=ub_coef_input1, ub_coef_input2=ub_coef_input2,
                            lb_bias_input1=lb_bias_input1, lb_bias_input2=lb_bias_input2,
                            ub_bias_input1=ub_bias_input1, ub_bias_input2=ub_bias_input2)
        return back_prop_struct

    def pos_neg_weight_decomposition(self, coef):
        neg_comp = torch.where(coef < 0, coef, torch.zeros_like(coef, device=self.device))
        pos_comp = torch.where(coef >= 0, coef, torch.zeros_like(coef, device=self.device))
        return neg_comp, pos_comp

    def refine_diff_bounds(self, back_prop_struct, lb_input1_layer, ub_input1_layer, 
                           lb_input2_layer, ub_input2_layer, layer_idx=None):
        neg_comp_lb_input1, pos_comp_lb_input1 = self.pos_neg_weight_decomposition(back_prop_struct.lb_coef_input1)
        neg_comp_ub_input1, pos_comp_ub_input1 = self.pos_neg_weight_decomposition(back_prop_struct.ub_coef_input1)
        neg_comp_lb_input2, pos_comp_lb_input2 = self.pos_neg_weight_decomposition(back_prop_struct.lb_coef_input2)
        neg_comp_ub_input2, pos_comp_ub_input2 = self.pos_neg_weight_decomposition(back_prop_struct.ub_coef_input2)

        lb_input1 = neg_comp_lb_input1 @ ub_input1_layer + pos_comp_lb_input1 @ lb_input1_layer + back_prop_struct.lb_bias_input1
        ub_input1 = neg_comp_ub_input1 @ lb_input1_layer + pos_comp_ub_input1 @ ub_input1_layer + back_prop_struct.ub_bias_input1 
        lb_input2 = neg_comp_lb_input2 @ ub_input2_layer + pos_comp_lb_input2 @ lb_input2_layer + back_prop_struct.lb_bias_input2
        ub_input2 = neg_comp_ub_input2 @ lb_input2_layer + pos_comp_ub_input2 @ ub_input2_layer + back_prop_struct.ub_bias_input2
        if layer_idx is not None:
            self.lb_input1[layer_idx] = torch.max(lb_input1, self.lb_input1[layer_idx])
            self.ub_input1[layer_idx] = torch.min(ub_input1, self.ub_input1[layer_idx]) 
            self.lb_input2[layer_idx] = torch.max(lb_input2, self.lb_input2[layer_idx])
            self.ub_input2[layer_idx] = torch.min(ub_input2, self.ub_input2[layer_idx])
        assert torch.all(lb_input1 <= ub_input1 + 1e-6)
        assert torch.all(lb_input2 <= ub_input2 + 1e-6)

        return (lb_input1 - ub_input2), (ub_input1 - lb_input2)      

    def concretize_bounds(self, back_prop_struct, delta_lb_layer, delta_ub_layer,
                          lb_input1_layer, ub_input1_layer, lb_input2_layer, ub_input2_layer, layer_idx=None):
        neg_comp_lb, pos_comp_lb = self.pos_neg_weight_decomposition(back_prop_struct.delta_lb_coef)
        # print(f"Delta coef {back_prop_struct.delta_lb_coef.shape}")
        neg_comp_ub, pos_comp_ub = self.pos_neg_weight_decomposition(back_prop_struct.delta_ub_coef)
        if back_prop_struct.delta_lb_input1_coef is not None:
            neg_comp_lb_input1, pos_comp_lb_input1 = self.pos_neg_weight_decomposition(back_prop_struct.delta_lb_input1_coef)
            neg_comp_ub_input1, pos_comp_ub_input1 = self.pos_neg_weight_decomposition(back_prop_struct.delta_ub_input1_coef)
            neg_comp_lb_input2, pos_comp_lb_input2 = self.pos_neg_weight_decomposition(back_prop_struct.delta_lb_input2_coef)
            neg_comp_ub_input2, pos_comp_ub_input2 = self.pos_neg_weight_decomposition(back_prop_struct.delta_ub_input2_coef)

        lb = neg_comp_lb @ delta_ub_layer + pos_comp_lb @ delta_lb_layer + back_prop_struct.delta_lb_bias
        if back_prop_struct.delta_lb_input1_coef is not None:
            lb = lb + neg_comp_lb_input1 @ ub_input1_layer + pos_comp_lb_input1 @ lb_input1_layer    
            lb = lb + neg_comp_lb_input2 @ ub_input2_layer + pos_comp_lb_input2 @ lb_input2_layer

        ub = neg_comp_ub @ delta_lb_layer + pos_comp_ub @ delta_ub_layer + back_prop_struct.delta_ub_bias
        if back_prop_struct.delta_lb_input1_coef is not None:
            ub = ub + neg_comp_ub_input1 @ lb_input1_layer + pos_comp_ub_input1 @ ub_input1_layer
            ub = ub + neg_comp_ub_input2 @ lb_input2_layer + pos_comp_ub_input2 @ ub_input2_layer

        if back_prop_struct.lb_coef_input1 is not None:
            lb_new, ub_new = self.refine_diff_bounds(back_prop_struct=back_prop_struct, lb_input1_layer=lb_input1_layer,
                                                    ub_input1_layer=ub_input1_layer, lb_input2_layer=lb_input2_layer,
                                                    ub_input2_layer=ub_input2_layer, layer_idx=layer_idx)
        else:
            lb_new, ub_new = lb, ub
        self.check_lb_ub_correctness(lb=lb_new, ub=ub_new)
        self.check_lb_ub_correctness(lb=lb, ub=ub)
        return torch.max(lb, lb_new), torch.min(ub, ub_new)

    # Consider cases based on the state of the relu for different propagation.
    def handle_relu(self, back_prop_struct, 
                    lb_input1_layer, ub_input1_layer, 
                    lb_input2_layer, ub_input2_layer,
                    delta_lb_layer, delta_ub_layer):

        input1_active = (lb_input1_layer >= 0)
        input1_passive = (ub_input1_layer <= 0)
        input1_unsettled = ~(input1_active) & ~(input1_passive)

        input2_active = (lb_input2_layer >= 0)
        input2_passive = (ub_input2_layer <= 0)
        input2_unsettled = ~(input2_active) & ~(input2_passive)

        delta_active = (delta_lb_layer >= 0)
        delta_passive = (delta_ub_layer <= 0)
        delta_unsettled = ~(delta_active) & ~(delta_passive)

        lambda_lb = torch.zeros(lb_input1_layer.size(), device=self.device)
        lambda_ub = torch.zeros(lb_input1_layer.size(), device=self.device)
        lambda_lb_input1 = torch.zeros(lb_input1_layer.size(), device=self.device)
        lambda_ub_input1 = torch.zeros(lb_input1_layer.size(), device=self.device)
        lambda_lb_input2 = torch.zeros(lb_input1_layer.size(), device=self.device)
        lambda_ub_input2 = torch.zeros(lb_input1_layer.size(), device=self.device)

        lambda_lb_input1_prop = torch.zeros(lb_input1_layer.size(), device=self.device)
        lambda_ub_input1_prop = torch.zeros(lb_input1_layer.size(), device=self.device)
        mu_ub_input1_prop = torch.zeros(lb_input1_layer.size(), device=self.device) 
        lambda_lb_input2_prop = torch.zeros(lb_input1_layer.size(), device=self.device)
        lambda_ub_input2_prop = torch.zeros(lb_input1_layer.size(), device=self.device)
        mu_ub_input2_prop = torch.zeros(lb_input1_layer.size(), device=self.device)

        # input1 is active
        lambda_lb_input1_prop = torch.where(input1_active, torch.ones(lb_input1_layer.size(), device=self.device), lambda_lb_input1_prop)
        lambda_ub_input1_prop = torch.where(input1_active, torch.ones(lb_input1_layer.size(), device=self.device), lambda_ub_input1_prop)
        # input1 is unsettled
        temp = torch.where(ub_input1_layer < -lb_input1_layer, torch.zeros(lb_input1_layer.size(), device=self.device), torch.ones(lb_input1_layer.size(), device=self.device))
        lambda_lb_input1_prop = torch.where(input1_unsettled, temp, lambda_lb_input1_prop)
        lambda_ub_input1_prop = torch.where(input1_unsettled, ub_input1_layer/(ub_input1_layer - lb_input1_layer + 1e-15), lambda_ub_input1_prop)
        mu_ub_input1_prop = torch.where(input1_unsettled, -(ub_input1_layer * lb_input1_layer) / (ub_input1_layer - lb_input1_layer + 1e-15), mu_ub_input1_prop)        

        # input2 is active
        lambda_lb_input2_prop = torch.where(input2_active, torch.ones(lb_input2_layer.size(), device=self.device), lambda_lb_input2_prop)
        lambda_ub_input2_prop = torch.where(input2_active, torch.ones(lb_input2_layer.size(), device=self.device), lambda_ub_input2_prop)
        # input2 is unsettled
        temp = torch.where(ub_input2_layer < -lb_input2_layer, torch.zeros(lb_input2_layer.size(), device=self.device), torch.ones(lb_input2_layer.size(), device=self.device))
        lambda_lb_input2_prop = torch.where(input2_unsettled, temp, lambda_lb_input2_prop)
        lambda_ub_input2_prop = torch.where(input2_unsettled, ub_input2_layer/(ub_input2_layer - lb_input2_layer + 1e-15), lambda_ub_input2_prop)
        mu_ub_input2_prop = torch.where(input2_unsettled, -(ub_input2_layer * lb_input2_layer) / (ub_input2_layer - lb_input2_layer + 1e-15), mu_ub_input2_prop)        

        # Checked 
        mu_lb = torch.zeros(lb_input1_layer.size(), device=self.device)
        mu_ub = torch.zeros(lb_input1_layer.size(), device=self.device)
        # case 1 x.ub <= 0 and y.ub <= 0
        # case 2 x.lb >=  0 and y.lb >= 0
        lambda_lb = torch.where(input1_active & input2_active, torch.ones(lb_input1_layer.size(), device=self.device), lambda_lb)
        lambda_ub = torch.where(input1_active & input2_active, torch.ones(lb_input1_layer.size(), device=self.device), lambda_ub)
        # case 3 x.lb >= 0 and y.ub <= 0
        # lambda_lb = torch.where(input1_active & input2_passive, torch.zeros(lb_input1_layer.size(), device=self.device), lambda_lb)
        # lambda_ub = torch.where(input1_active & input2_passive, torch.zeros(lb_input1_layer.size(), device=self.device), lambda_ub)
        # mu_lb = torch.where(input1_active & input2_passive, lb_input1_layer, mu_lb)
        # mu_ub = torch.where(input1_active & input2_passive, ub_input1_layer, mu_ub)
        lambda_lb_input1 = torch.where(input1_active & input2_passive, torch.ones(lb_input1_layer.size(), device=self.device), lambda_lb_input1)
        lambda_ub_input1 = torch.where(input1_active & input2_passive, torch.ones(lb_input1_layer.size(), device=self.device), lambda_ub_input1)



        #case 4 (x.lb < 0 and x.ub > 0) and y.ub <= 0
        # lambda_lb = torch.where(input1_unsettled & input2_passive, torch.zeros(lb_input1_layer.size(), device=self.device), lambda_lb)
        # lambda_ub = torch.where(input1_unsettled & input2_passive, torch.zeros(lb_input1_layer.size(), device=self.device), lambda_ub)
        # mu_lb = torch.where(input1_unsettled & input2_passive, torch.zeros(lb_input1_layer.size(), device=self.device), mu_lb)
        # mu_ub = torch.where(input1_unsettled & input2_passive, ub_input1_layer, mu_ub)
        lambda_lb_input1 = torch.where(input1_unsettled & input2_passive, lambda_lb_input1_prop, lambda_lb_input1)
        lambda_ub_input1 = torch.where(input1_unsettled & input2_passive, lambda_ub_input1_prop, lambda_ub_input1)
        mu_ub = torch.where(input1_unsettled & input2_passive, mu_ub_input1_prop, mu_ub)

        #case 5 (x.ub <= 0) and y.lb >= 0
        lambda_lb_input2 = torch.where(input1_passive & input2_active, -torch.ones(lb_input1_layer.size(), device=self.device), lambda_lb_input2)
        lambda_ub_input2 = torch.where(input1_passive & input2_active, -torch.ones(lb_input1_layer.size(), device=self.device), lambda_ub_input2)

        # case 6 (x.ub <= 0) and (y.lb < 0 and y.ub > 0)
        lambda_lb_input2 = torch.where(input1_passive & input2_unsettled, -lambda_ub_input2_prop, lambda_lb_input2)
        lambda_ub_input2 = torch.where(input1_passive & input2_unsettled, -lambda_lb_input2_prop, lambda_ub_input2)
        mu_lb = torch.where(input1_passive & input2_unsettled, -mu_ub_input2_prop, mu_lb)


        # case 7 (x.lb >= 0) and (y.lb < 0 and y.ub > 0)
        # lambda_lb = torch.where(input1_active & input2_unsettled, torch.zeros(lb_input1_layer.size(), device=self.device), lambda_lb)
        lambda_ub = torch.where(input1_active & input2_unsettled, torch.ones(lb_input1_layer.size(), device=self.device), lambda_ub)
        # mu_lb = torch.where(input1_active & input2_unsettled, torch.min(lb_input1_layer, delta_lb_layer), mu_lb)
        mu_ub = torch.where(input1_active & input2_unsettled, torch.zeros(lb_input1_layer.size(), device=self.device), mu_ub)
        lambda_lb_input1 = torch.where(input1_active & input2_unsettled, torch.ones(lb_input1_layer.size(), device=self.device), lambda_lb_input1)
        # lambda_ub_input1 = torch.where(input1_active & input2_unsettled, torch.ones(lb_input1_layer.size(), device=self.device), lambda_ub_input1)
        lambda_lb_input2 = torch.where(input1_active & input2_unsettled, -lambda_ub_input2_prop, lambda_lb_input2)
        # lambda_ub_input2 = torch.where(input1_active & input2_unsettled, -lambda_lb_input2_prop, lambda_ub_input2)
        mu_lb = torch.where(input1_active & input2_unsettled, -mu_ub_input2_prop, mu_lb)

        # case 8 (x.lb < 0 and x.ub > 0) and (y.lb >= 0)
        lambda_lb = torch.where(input1_unsettled & input2_active, torch.ones(lb_input1_layer.size(), device=self.device), lambda_lb)
        # lambda_ub = torch.where(input1_unsettled & input2_active, torch.zeros(lb_input1_layer.size(), device=self.device), lambda_ub)
        mu_lb = torch.where(input1_unsettled & input2_active, torch.zeros(lb_input1_layer.size(), device=self.device), mu_lb)
        # mu_ub = torch.where(input1_unsettled & input2_active, torch.max(delta_ub_layer, -lb_input2_layer), mu_ub)
        # lambda_lb_input1 = torch.where(input1_unsettled & input2_active, lambda_lb_input1_prop, lambda_lb_input1)
        lambda_ub_input1 = torch.where(input1_unsettled & input2_active, lambda_ub_input1_prop, lambda_ub_input1)
        mu_ub = torch.where(input1_unsettled & input2_active, mu_ub_input1_prop, mu_ub)
        # lambda_lb_input2 = torch.where(input1_unsettled & input2_active, -torch.ones(lb_input1_layer.size(), device=self.device), lambda_lb_input2)
        lambda_ub_input2 = torch.where(input1_unsettled & input2_active, -torch.ones(lb_input1_layer.size(), device=self.device), lambda_ub_input2)


        # case 9 (x.lb < 0 and x.ub > 0) and (y.lb < 0 and y.ub > 0) and (delta_lb >= 0)
        lambda_lb = torch.where(input1_unsettled & input2_unsettled & delta_active, torch.zeros(lb_input1_layer.size(), device=self.device), lambda_lb)
        mu_lb = torch.where(input1_unsettled & input2_unsettled & delta_active, torch.zeros(lb_input1_layer.size(), device=self.device), mu_lb)
        mu_ub = torch.where(input1_unsettled & input2_unsettled & delta_active, torch.zeros(lb_input1_layer.size(), device=self.device), mu_ub)
        # lambda_ub = torch.where(input1_unsettled & input2_unsettled & delta_active, torch.ones(lb_input1_layer.size(), device=self.device), lambda_ub)
        # if lambda.ub <= x.ub then delta_ub = delta
        # else delta_ub = x.ub
        case_9 = (input1_unsettled & input2_unsettled & delta_active)
        temp_lambda = torch.where((ub_input1_layer < delta_ub_layer) & case_9, lambda_ub_input1_prop, torch.ones(lb_input1_layer.size(), device=self.device))
        lambda_ub = torch.where(case_9 & (ub_input1_layer >= delta_ub_layer), temp_lambda, lambda_ub)
        lambda_ub_input1 = torch.where(case_9 & (ub_input1_layer < delta_ub_layer), temp_lambda, lambda_ub_input1)
        mu_ub = torch.where(case_9 & (ub_input1_layer < delta_ub_layer), mu_ub_input1_prop, mu_ub)
        
        # Checked        
        # case 10 (x.lb < 0 and x.ub > 0) and (y.lb < 0 and y.ub > 0) and (delta_ub <= 0)
        lambda_ub = torch.where(input1_unsettled & input2_unsettled & delta_passive, torch.zeros(lb_input1_layer.size(), device=self.device), lambda_ub)
        mu_lb = torch.where(input1_unsettled & input2_unsettled & delta_passive, torch.zeros(lb_input1_layer.size(), device=self.device), mu_lb)
        mu_ub = torch.where(input1_unsettled & input2_unsettled & delta_passive, torch.zeros(lb_input1_layer.size(), device=self.device), mu_ub)
        case_10 = (input1_unsettled & input2_unsettled & delta_passive)
        temp_lambda = torch.where((ub_input2_layer < delta_ub_layer) & case_10, -lambda_ub_input2_prop, torch.ones(lb_input1_layer.size(), device=self.device))
        lambda_lb = torch.where((ub_input2_layer >= delta_ub_layer) & case_10, temp_lambda, lambda_lb)
        lambda_lb_input2 = torch.where((ub_input2_layer < delta_ub_layer) & case_10, temp_lambda, lambda_lb_input2)
        mu_lb = torch.where((ub_input2_layer < delta_ub_layer) & case_10, -mu_ub_input2_prop, mu_lb)

        # case 11 (x.lb < 0 and x.ub > 0) and (y.lb < 0 and y.ub > 0) and (delta_lb < 0 and delta_ub > 0)
        temp_mu = (delta_lb_layer * delta_ub_layer) / (delta_ub_layer - delta_lb_layer + 1e-15)
        temp_lambda_lb = (-delta_lb_layer) / (delta_ub_layer - delta_lb_layer + 1e-15)
        temp_lambda_ub = delta_ub_layer / (delta_ub_layer - delta_lb_layer + 1e-15)
        use_delta_lb = (torch.abs(temp_mu) == torch.abs(mu_ub_input2_prop))
        use_delta_ub = (torch.abs(temp_mu) == torch.abs(mu_ub_input1_prop))

        lambda_lb = torch.where(input1_unsettled & input2_unsettled & delta_unsettled & use_delta_lb, temp_lambda_lb, lambda_lb)
        lambda_ub = torch.where(input1_unsettled & input2_unsettled & delta_unsettled & use_delta_ub, temp_lambda_ub, lambda_ub)
        mu_lb = torch.where(input1_unsettled & input2_unsettled & delta_unsettled & use_delta_lb, temp_mu, mu_lb)
        mu_ub = torch.where(input1_unsettled & input2_unsettled & delta_unsettled & use_delta_ub, -temp_mu, mu_ub)
        
        lambda_lb_input1 = torch.where(input1_unsettled & input2_unsettled & delta_unsettled & ~use_delta_lb, lambda_lb_input1_prop, lambda_lb_input1)
        lambda_ub_input1 = torch.where(input1_unsettled & input2_unsettled & delta_unsettled & ~use_delta_ub, lambda_ub_input1_prop, lambda_ub_input1)
        mu_ub = torch.where(input1_unsettled & input2_unsettled & delta_unsettled & ~use_delta_ub, mu_ub_input1_prop, mu_ub)
        lambda_lb_input2 = torch.where(input1_unsettled & input2_unsettled & delta_unsettled & ~use_delta_lb, -lambda_ub_input2_prop, lambda_lb_input2)
        lambda_ub_input2 = torch.where(input1_unsettled & input2_unsettled & delta_unsettled & ~use_delta_ub, -lambda_lb_input2_prop, lambda_ub_input2)
        mu_lb = torch.where(input1_unsettled & input2_unsettled & delta_unsettled & ~use_delta_lb, -mu_ub_input2_prop, mu_lb)

        # checked 
        # Segregate the +ve and -ve components of the coefficients
        neg_comp_lb, pos_comp_lb = self.pos_neg_weight_decomposition(back_prop_struct.delta_lb_coef)
        neg_comp_ub, pos_comp_ub = self.pos_neg_weight_decomposition(back_prop_struct.delta_ub_coef)
        neg_comp_lb_input1, pos_comp_lb_input1 = self.pos_neg_weight_decomposition(back_prop_struct.delta_lb_input1_coef)
        neg_comp_ub_input1, pos_comp_ub_input1 = self.pos_neg_weight_decomposition(back_prop_struct.delta_ub_input1_coef)
        neg_comp_lb_input2, pos_comp_lb_input2 = self.pos_neg_weight_decomposition(back_prop_struct.delta_lb_input2_coef)
        neg_comp_ub_input2, pos_comp_ub_input2 = self.pos_neg_weight_decomposition(back_prop_struct.delta_ub_input2_coef)

        neg_coef_lb_input1, pos_coef_lb_input1 = self.pos_neg_weight_decomposition(back_prop_struct.lb_coef_input1)
        neg_coef_ub_input1, pos_coef_ub_input1 = self.pos_neg_weight_decomposition(back_prop_struct.ub_coef_input1)
        neg_coef_lb_input2, pos_coef_lb_input2 = self.pos_neg_weight_decomposition(back_prop_struct.lb_coef_input2)
        neg_coef_ub_input2, pos_coef_ub_input2 = self.pos_neg_weight_decomposition(back_prop_struct.ub_coef_input2)       


        delta_lb_coef = pos_comp_lb * lambda_lb + neg_comp_lb * lambda_ub
        delta_lb_input1_coef = pos_comp_lb_input1 * lambda_lb_input1_prop + neg_comp_lb_input1 * lambda_ub_input1_prop
        delta_lb_input1_coef = delta_lb_input1_coef + pos_comp_lb * lambda_lb_input1 + neg_comp_lb * lambda_ub_input1
        delta_lb_input2_coef = pos_comp_lb_input2 * lambda_lb_input2_prop + neg_comp_lb_input2 * lambda_ub_input2_prop
        delta_lb_input2_coef = delta_lb_input2_coef + pos_comp_lb * lambda_lb_input2 + neg_comp_lb * lambda_ub_input2

     
        delta_ub_coef = pos_comp_ub * lambda_ub + neg_comp_ub * lambda_lb
        delta_ub_input1_coef = pos_comp_ub_input1 * lambda_ub_input1_prop + neg_comp_ub_input1 * lambda_lb_input1_prop
        delta_ub_input1_coef = delta_ub_input1_coef + pos_comp_ub * lambda_ub_input1 + neg_comp_ub * lambda_lb_input1
        delta_ub_input2_coef = pos_comp_ub_input2 * lambda_ub_input2_prop + neg_comp_ub_input2 * lambda_lb_input2_prop
        delta_ub_input2_coef = delta_ub_input2_coef + pos_comp_ub * lambda_ub_input2 + neg_comp_ub * lambda_lb_input2


        delta_lb_bias = pos_comp_lb @ mu_lb + neg_comp_lb @ mu_ub + back_prop_struct.delta_lb_bias
        delta_lb_bias = delta_lb_bias + neg_comp_lb_input1 @ mu_ub_input1_prop + neg_comp_lb_input2 @ mu_ub_input2_prop
        delta_ub_bias = pos_comp_ub @ mu_ub + neg_comp_ub @ mu_lb + back_prop_struct.delta_ub_bias
        delta_ub_bias = delta_ub_bias + pos_comp_ub_input1 @ mu_ub_input1_prop + pos_comp_ub_input2 @ mu_ub_input2_prop

        if back_prop_struct.lb_bias_input1 is not None:
            lb_bias_input1 = back_prop_struct.lb_bias_input1 + neg_coef_lb_input1 @ mu_ub_input1_prop
            ub_bias_input1 = back_prop_struct.ub_bias_input1 + pos_coef_ub_input1 @ mu_ub_input1_prop
            lb_bias_input2 = back_prop_struct.lb_bias_input2 + neg_coef_lb_input2 @ mu_ub_input2_prop
            ub_bias_input2 = back_prop_struct.ub_bias_input2 + pos_coef_ub_input2 @ mu_ub_input2_prop

            lb_coef_input1 = neg_coef_lb_input1 * lambda_ub_input1_prop + pos_coef_lb_input1 * lambda_lb_input1_prop
            ub_coef_input1 = neg_coef_ub_input1 * lambda_lb_input1_prop + pos_coef_ub_input1 * lambda_ub_input1_prop
            lb_coef_input2 = neg_coef_lb_input2 * lambda_ub_input2_prop + pos_coef_lb_input2 * lambda_lb_input2_prop
            ub_coef_input2 = neg_coef_ub_input2 * lambda_lb_input2_prop + pos_coef_ub_input2 * lambda_ub_input2_prop
        else:
            lb_bias_input1 = None
            ub_bias_input1 = None
            lb_bias_input2 = None
            ub_bias_input2 = None

            lb_coef_input1 = None
            ub_coef_input1 = None
            lb_coef_input2 = None
            ub_coef_input2 = None


        back_prop_struct.populate(delta_lb_coef=delta_lb_coef, delta_lb_bias=delta_lb_bias, 
                                    delta_ub_coef=delta_ub_coef, delta_ub_bias=delta_ub_bias,
                                    delta_lb_input1_coef=delta_lb_input1_coef, 
                                    delta_ub_input1_coef=delta_ub_input1_coef,
                                    delta_lb_input2_coef=delta_lb_input2_coef,
                                    delta_ub_input2_coef=delta_ub_input2_coef,
                                    lb_coef_input1=lb_coef_input1, lb_coef_input2=lb_coef_input2,
                                    ub_coef_input1=ub_coef_input1, ub_coef_input2=ub_coef_input2,
                                    lb_bias_input1=lb_bias_input1, lb_bias_input2=lb_bias_input2,
                                    ub_bias_input1=ub_bias_input1, ub_bias_input2=ub_bias_input2)


        return back_prop_struct

    # Use the individual propagation for diffPoly.
    def analyze_sigmoid(self, poly_struct, lb_layer, ub_layer):
        sigmoid_lb, sigmoid_ub = torch.sigmoid(lb_layer), torch.sigmoid(ub_layer)
        lmbda = torch.where(lb_layer < ub_layer, (sigmoid_ub - sigmoid_lb) / (ub_layer - lb_layer + 1e-15),  sigmoid_lb * (1 - sigmoid_lb))
        lmbda_ = torch.min(sigmoid_ub * (1 - sigmoid_ub), sigmoid_lb * (1 - sigmoid_lb))
        lambda_lb = torch.where(lb_layer > 0, lmbda, lmbda_)
        mu_lb = torch.where(lb_layer > 0, sigmoid_lb - torch.mul(lmbda, lb_layer),  sigmoid_lb - torch.mul(lmbda_, lb_layer))
        lambda_ub = torch.where(ub_layer < 0, lmbda, lmbda_)
        mu_ub =  torch.where(ub_layer < 0, sigmoid_ub - torch.mul(lmbda, ub_layer),  sigmoid_ub - torch.mul(lmbda_, lb_layer))

        neg_comp_lb, pos_comp_lb = self.pos_neg_weight_decomposition(poly_struct.lb_coef)
        neg_comp_ub, pos_comp_ub = self.pos_neg_weight_decomposition(poly_struct.ub_coef)
        
        lb_coef = neg_comp_lb * lambda_ub + pos_comp_lb * lambda_lb
        ub_coef = neg_comp_ub * lambda_lb + pos_comp_ub * lambda_ub
        lb_bias = poly_struct.lb_bias + neg_comp_lb @ mu_ub + pos_comp_lb @ mu_lb
        ub_bias = poly_struct.ub_bias + pos_comp_ub @ mu_ub + neg_comp_ub @ mu_lb
        poly_struct = BasicDeepPolyStruct(lb_bias=lb_bias, lb_coef=lb_coef, ub_bias=ub_bias, ub_coef=ub_coef)

        return poly_struct


    def handle_sigmoid(self, back_prop_struct, lb_input1_layer, ub_input1_layer, 
                    lb_input2_layer, ub_input2_layer,
                    delta_lb_layer, delta_ub_layer):
        poly_struct1 = BasicDeepPolyStruct(lb_bias=back_prop_struct.lb_bias_input1, 
                                           lb_coef=back_prop_struct.lb_coef_input1,
                                           ub_bias=back_prop_struct.ub_bias_input1,
                                           ub_coef=back_prop_struct.ub_coef_input1)
        poly_struct1 = self.analyze_sigmoid(poly_struct=poly_struct1, lb_layer=lb_input1_layer, ub_layer=ub_input1_layer)

        poly_struct2 = BasicDeepPolyStruct(lb_bias=back_prop_struct.lb_bias_input2, 
                                           lb_coef=back_prop_struct.lb_coef_input2,
                                           ub_bias=back_prop_struct.ub_bias_input2,
                                           ub_coef=back_prop_struct.ub_coef_input2)
        poly_struct2 = self.analyze_sigmoid(poly_struct=poly_struct2, lb_layer=lb_input2_layer, ub_layer=ub_input2_layer)

        lb, ub = torch.min(lb_input1_layer, lb_input2_layer), torch.max(ub_input1_layer, ub_input2_layer)
        sigmoid_lb, sigmoid_ub = torch.sigmoid(lb), torch.sigmoid(ub)
        lambda_lower, lambda_upper = sigmoid_lb * (1.0 - sigmoid_lb), sigmoid_ub * (1.0 - sigmoid_ub)
        
        input_active = (lb >= 0)
        input_passive = (ub <= 0) 
        input_unsettled = ~(input_active) & ~(input_passive)

        deriv_min = torch.min(lambda_lower, lambda_upper)
        deriv_max = 0.25 * torch.ones(lb_input1_layer.size(), device=self.device)
        deriv_max = torch.where(~input_unsettled, torch.max(lambda_lower, lambda_upper), deriv_max)



        delta_active = (delta_lb_layer >= 0)
        delta_passive = (delta_ub_layer <= 0)
        delta_unsettled = ~(delta_active) & ~(delta_passive)

        lambda_lb = torch.zeros(lb_input1_layer.size(), device=self.device)
        lambda_ub = torch.zeros(lb_input1_layer.size(), device=self.device)

        mu_lb = torch.zeros(lb_input1_layer.size(), device=self.device)
        mu_ub = torch.zeros(lb_input1_layer.size(), device=self.device)

        # case 1 delta_lb >= 0 
        lambda_lb = torch.where(delta_active, deriv_min, lambda_lb)
        lambda_ub = torch.where(delta_active, deriv_max, lambda_ub)

        # case 2 delta_lb >= 0 
        lambda_lb = torch.where(delta_passive, deriv_max, lambda_lb)
        lambda_ub = torch.where(delta_passive, deriv_min, lambda_ub)

        # case 2 delta_lb < 0 and delta_ub > 0
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

        neg_comp_lb, pos_comp_lb = self.pos_neg_weight_decomposition(back_prop_struct.delta_lb_coef)
        neg_comp_ub, pos_comp_ub = self.pos_neg_weight_decomposition(back_prop_struct.delta_ub_coef)

        delta_lb_coef = neg_comp_lb * lambda_ub + pos_comp_lb * lambda_lb 
        delta_ub_coef = neg_comp_ub * lambda_lb + pos_comp_ub * lambda_ub

        delta_lb_bias = back_prop_struct.delta_lb_bias + neg_comp_lb @ mu_ub + pos_comp_lb @ mu_lb
        delta_ub_bias = back_prop_struct.delta_ub_bias + neg_comp_ub @ mu_lb + pos_comp_ub @ mu_ub

        back_prop_struct.populate(delta_lb_coef=delta_lb_coef, delta_lb_bias=delta_lb_bias, 
                                    delta_ub_coef=delta_ub_coef, delta_ub_bias=delta_ub_bias,
                                    delta_lb_input1_coef=None, 
                                    delta_ub_input1_coef=None,
                                    delta_lb_input2_coef=None,
                                    delta_ub_input2_coef=None,
                                    lb_coef_input1=poly_struct1.lb_coef, lb_coef_input2=poly_struct2.lb_coef,
                                    ub_coef_input1=poly_struct1.ub_coef, ub_coef_input2=poly_struct2.ub_coef,
                                    lb_bias_input1=poly_struct1.lb_bias, lb_bias_input2=poly_struct2.lb_bias,
                                    ub_bias_input1=poly_struct1.ub_bias, ub_bias_input2=poly_struct2.ub_bias)

        
        return back_prop_struct


    def analyze_tanh(self, poly_struct, lb_layer, ub_layer):
        tanh_lb, tanh_ub = torch.tanh(lb_layer), torch.tanh(ub_layer)
        lmbda = torch.where(lb_layer < ub_layer, (tanh_ub - tanh_lb) / (ub_layer - lb_layer + 1e-15),  1 - tanh_lb * tanh_lb)
        lmbda_ = torch.min(1 - tanh_ub * tanh_ub, 1 - tanh_lb * tanh_lb)

        lambda_lb = torch.where(lb_layer > 0, lmbda, lmbda_)
        mu_lb = torch.where(lb_layer > 0, tanh_lb - torch.mul(lmbda, lb_layer),  tanh_lb - torch.mul(lmbda_, lb_layer))
        
        lambda_ub = torch.where(ub_layer < 0, lmbda, lmbda_)
        mu_ub =  torch.where(ub_layer < 0, tanh_ub - torch.mul(lmbda, ub_layer),  tanh_ub - torch.mul(lmbda_, lb_layer))

        neg_comp_lb, pos_comp_lb = self.pos_neg_weight_decomposition(poly_struct.lb_coef)
        neg_comp_ub, pos_comp_ub = self.pos_neg_weight_decomposition(poly_struct.ub_coef)
        
        lb_coef = neg_comp_lb * lambda_ub + pos_comp_lb * lambda_lb
        ub_coef = neg_comp_ub * lambda_lb + pos_comp_ub * lambda_ub
        lb_bias = poly_struct.lb_bias + neg_comp_lb @ mu_ub + pos_comp_lb @ mu_lb
        ub_bias = poly_struct.ub_bias + pos_comp_ub @ mu_ub + neg_comp_ub @ mu_lb
        poly_struct = BasicDeepPolyStruct(lb_bias=lb_bias, lb_coef=lb_coef, ub_bias=ub_bias, ub_coef=ub_coef)
        return poly_struct


    def handle_tanh(self, back_prop_struct, lb_input1_layer, ub_input1_layer, 
                    lb_input2_layer, ub_input2_layer,
                    delta_lb_layer, delta_ub_layer):
        
        poly_struct1 = BasicDeepPolyStruct(lb_bias=back_prop_struct.lb_bias_input1, 
                                           lb_coef=back_prop_struct.lb_coef_input1,
                                           ub_bias=back_prop_struct.ub_bias_input1,
                                           ub_coef=back_prop_struct.ub_coef_input1)
        poly_struct1 = self.analyze_tanh(poly_struct=poly_struct1, lb_layer=lb_input1_layer, ub_layer=ub_input1_layer)

        poly_struct2 = BasicDeepPolyStruct(lb_bias=back_prop_struct.lb_bias_input2, 
                                           lb_coef=back_prop_struct.lb_coef_input2,
                                           ub_bias=back_prop_struct.ub_bias_input2,
                                           ub_coef=back_prop_struct.ub_coef_input2)
        poly_struct2 = self.analyze_tanh(poly_struct=poly_struct2, lb_layer=lb_input2_layer, ub_layer=ub_input2_layer)

        lb, ub = torch.min(lb_input1_layer, lb_input2_layer), torch.max(ub_input1_layer, ub_input2_layer)
        tanh_lb, tanh_ub = torch.tanh(lb), torch.tanh(ub)
        lambda_lower, lambda_upper = 1.0 - (tanh_lb * tanh_lb), 1.0 - (tanh_ub * tanh_ub)

        input_active = (lb >= 0)
        input_passive = (ub <= 0) 
        input_unsettled = ~(input_active) & ~(input_passive)

        deriv_min = torch.min(lambda_lower, lambda_upper)
        deriv_max = torch.ones(lb_input1_layer.size(), device=self.device)
        deriv_max = torch.where(~input_unsettled, torch.max(lambda_lower, lambda_upper), deriv_max)



        delta_active = (delta_lb_layer >= 0)
        delta_passive = (delta_ub_layer <= 0)
        delta_unsettled = ~(delta_active) & ~(delta_passive)

        lambda_lb = torch.zeros(lb_input1_layer.size(), device=self.device)
        lambda_ub = torch.zeros(lb_input1_layer.size(), device=self.device)

        mu_lb = torch.zeros(lb_input1_layer.size(), device=self.device)
        mu_ub = torch.zeros(lb_input1_layer.size(), device=self.device)

        # case 1 delta_lb >= 0 
        lambda_lb = torch.where(delta_active, deriv_min, lambda_lb)
        lambda_ub = torch.where(delta_active, deriv_max, lambda_ub)

        # case 2 delta_lb >= 0 
        lambda_lb = torch.where(delta_passive, deriv_max, lambda_lb)
        lambda_ub = torch.where(delta_passive, deriv_min, lambda_ub)

        # case 2 delta_lb < 0 and delta_ub > 0
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

        neg_comp_lb, pos_comp_lb = self.pos_neg_weight_decomposition(back_prop_struct.delta_lb_coef)
        neg_comp_ub, pos_comp_ub = self.pos_neg_weight_decomposition(back_prop_struct.delta_ub_coef)

        delta_lb_coef = neg_comp_lb * lambda_ub + pos_comp_lb * lambda_lb 
        delta_ub_coef = neg_comp_ub * lambda_lb + pos_comp_ub * lambda_ub

        delta_lb_bias = back_prop_struct.delta_lb_bias + neg_comp_lb @ mu_ub + pos_comp_lb @ mu_lb
        delta_ub_bias = back_prop_struct.delta_ub_bias + neg_comp_ub @ mu_lb + pos_comp_ub @ mu_ub

        back_prop_struct.populate(delta_lb_coef=delta_lb_coef, delta_lb_bias=delta_lb_bias, 
                                    delta_ub_coef=delta_ub_coef, delta_ub_bias=delta_ub_bias,
                                    delta_lb_input1_coef=None, 
                                    delta_ub_input1_coef=None,
                                    delta_lb_input2_coef=None,
                                    delta_ub_input2_coef=None,
                                    lb_coef_input1=poly_struct1.lb_coef, lb_coef_input2=poly_struct2.lb_coef,
                                    ub_coef_input1=poly_struct1.ub_coef, ub_coef_input2=poly_struct2.ub_coef,
                                    lb_bias_input1=poly_struct1.lb_bias, lb_bias_input2=poly_struct2.lb_bias,
                                    ub_bias_input1=poly_struct1.ub_bias, ub_bias_input2=poly_struct2.ub_bias)
        return back_prop_struct



    def get_layer_size(self, linear_layer_index):
        layer = self.net[self.linear_conv_layer_indices[linear_layer_index]]
        if layer.type is LayerType.Linear:
            shape = self.shapes[linear_layer_index + 1]
            return shape
        if layer.type is LayerType.Conv2D:
            shape = self.shapes[linear_layer_index+ 1]
            return (shape[0] * shape[1] * shape[2])
    

    def check_lb_ub_correctness(self, lb, ub):
        if not torch.all(lb <= ub + 1e-6) :
            assert torch.all(lb <= ub + 1e-6)

    def initialize_back_prop_struct(self, layer_idx):
        layer_size = self.get_layer_size(linear_layer_index=layer_idx)
        delta_lb_coef = torch.eye(n=layer_size, device=self.device)
        delta_lb_bias = torch.zeros(layer_size, device=self.device)
        delta_ub_coef = torch.eye(n=layer_size, device=self.device)
        delta_ub_bias = torch.zeros(layer_size, device=self.device)
        delta_lb_input1_coef = torch.zeros((layer_size, layer_size), device=self.device)                
        delta_ub_input1_coef = torch.zeros((layer_size, layer_size), device=self.device)
        delta_lb_input2_coef = torch.zeros((layer_size, layer_size), device=self.device)
        delta_ub_input2_coef = torch.zeros((layer_size, layer_size), device=self.device)
        if self.lightweight_diffpoly is False:
            lb_coef_input1 = torch.eye(n=layer_size, device=self.device)
            lb_coef_input2 = torch.eye(n=layer_size, device=self.device)
            ub_coef_input1 = torch.eye(n=layer_size, device=self.device)
            ub_coef_input2 = torch.eye(n=layer_size, device=self.device)

            lb_bias_input1 = torch.zeros(layer_size, device=self.device)
            lb_bias_input2 = torch.zeros(layer_size, device=self.device)
            ub_bias_input1 = torch.zeros(layer_size, device=self.device)
            ub_bias_input2 = torch.zeros(layer_size, device=self.device)
        else:
            lb_coef_input1 = None
            lb_coef_input2 = None
            ub_coef_input1 = None
            ub_coef_input2 = None

            lb_bias_input1 = None
            lb_bias_input2 = None
            ub_bias_input1 = None
            ub_bias_input2 = None


        back_prop_struct = DiffPropStruct()
        back_prop_struct.populate(delta_lb_coef=delta_lb_coef, delta_lb_bias=delta_lb_bias, 
                                    delta_ub_coef=delta_ub_coef, delta_ub_bias=delta_ub_bias,
                                    delta_lb_input1_coef=delta_lb_input1_coef, 
                                    delta_ub_input1_coef=delta_ub_input1_coef,
                                    delta_lb_input2_coef=delta_lb_input2_coef,
                                    delta_ub_input2_coef=delta_ub_input2_coef,
                                    lb_coef_input1=lb_coef_input1, lb_coef_input2=lb_coef_input2,
                                    ub_coef_input1=ub_coef_input1, ub_coef_input2=ub_coef_input2,
                                    lb_bias_input1=lb_bias_input1, lb_bias_input2=lb_bias_input2,
                                    ub_bias_input1=ub_bias_input1, ub_bias_input2=ub_bias_input2)
        return back_prop_struct

    def handle_layer(self, prop_struct, layer, linear_layer_idx, layer_idx, delta_lbs, delta_ubs):
        if layer.type is LayerType.Linear:
            back_prop_struct = self.handle_linear(linear_wt=layer.weight,
                                    bias=layer.bias, back_prop_struct=prop_struct)
        elif layer.type is LayerType.Conv2D:
            back_prop_struct = self.handle_conv(conv_weight=layer.weight, conv_bias=layer.bias, 
                                    back_prop_struct=prop_struct, 
                                    preconv_shape=self.shapes[linear_layer_idx], postconv_shape=self.shapes[linear_layer_idx + 1],
                                    stride=layer.stride, padding=layer.padding, dilation=layer.dilation)
        elif layer.type is LayerType.ReLU:
            back_prop_struct = self.handle_relu(
                                            back_prop_struct=prop_struct,
                                            lb_input1_layer=self.lb_input1[layer_idx], 
                                            ub_input1_layer=self.ub_input1[layer_idx], 
                                            lb_input2_layer=self.lb_input2[layer_idx], 
                                            ub_input2_layer=self.ub_input2[layer_idx],
                                            delta_lb_layer=delta_lbs[layer_idx], 
                                            delta_ub_layer=delta_ubs[layer_idx])
        elif layer.type is LayerType.Sigmoid:
            back_prop_struct = self.handle_sigmoid(
                                            back_prop_struct=prop_struct,
                                            lb_input1_layer=self.lb_input1[layer_idx], 
                                            ub_input1_layer=self.ub_input1[layer_idx], 
                                            lb_input2_layer=self.lb_input2[layer_idx], 
                                            ub_input2_layer=self.ub_input2[layer_idx],
                                            delta_lb_layer=delta_lbs[layer_idx], 
                                            delta_ub_layer=delta_ubs[layer_idx])
        elif layer.type is LayerType.TanH:
            back_prop_struct = self.handle_tanh(
                                            back_prop_struct=prop_struct,
                                            lb_input1_layer=self.lb_input1[layer_idx], 
                                            ub_input1_layer=self.ub_input1[layer_idx], 
                                            lb_input2_layer=self.lb_input2[layer_idx], 
                                            ub_input2_layer=self.ub_input2[layer_idx],
                                            delta_lb_layer=delta_lbs[layer_idx], 
                                            delta_ub_layer=delta_ubs[layer_idx])
        else:
            raise NotImplementedError(f'diff verifier for {layer.type} is not implemented')
        return back_prop_struct

    # layer index : index of the current layer
    # linear layer index: No of linear layers seen before the current layer.
    def back_substitution_affine_only(self, layer_index, linear_layer_index, delta_lbs, delta_ubs):
        if linear_layer_index != len(delta_lbs):
            raise ValueError("Size of lower bounds computed in previous layers don't match")

        back_prop_struct = None
        delta_lb = None
        delta_ub = None

        for i in reversed(range(layer_index + 1)):
            # Concretize the bounds for the previous layers.
            if self.net[i].type in [LayerType.Linear, LayerType.Conv2D] and back_prop_struct is not None:
                new_delta_lb, new_delta_ub = self.concretize_bounds(back_prop_struct=back_prop_struct,delta_lb_layer=delta_lbs[linear_layer_index], 
                                                                    delta_ub_layer=delta_ubs[linear_layer_index],
                                                         lb_input1_layer=self.lb_input1[linear_layer_index], 
                                                ub_input1_layer=self.ub_input1[linear_layer_index], 
                                                lb_input2_layer=self.lb_input2[linear_layer_index], 
                                                ub_input2_layer=self.ub_input2[linear_layer_index])
                self.check_lb_ub_correctness(lb=new_delta_lb, ub=new_delta_ub)
                delta_lb = (new_delta_lb if delta_lb is None else (torch.max(delta_lb, new_delta_lb)))
                delta_ub = (new_delta_ub if delta_ub is None else (torch.min(delta_ub, new_delta_ub)))

            if back_prop_struct is None:
                back_prop_struct = self.initialize_back_prop_struct(layer_idx=linear_layer_index)

            curr_layer = self.net[i]
            back_prop_struct = self.handle_layer(prop_struct=back_prop_struct, layer=curr_layer, linear_layer_idx=linear_layer_index, 
                                                 layer_idx=linear_layer_index, delta_lbs=delta_lbs, delta_ubs=delta_ubs)
            if curr_layer.type in [LayerType.Linear, LayerType.Conv2D]:
                linear_layer_index -= 1
        
        if self.monotone:         
            new_delta_lb, new_delta_ub = self.concretize_bounds(back_prop_struct=back_prop_struct,
                                                                #delta_lb_layer=torch.zeros(12),#-self.eps * torch.nn.functional.one_hot(self.noise_ind[0], 12).flatten(),
                                                                delta_lb_layer = torch.zeros(87),
                                                                delta_ub_layer = self.eps * torch.nn.functional.one_hot(torch.tensor(self.monotone_prop), 87).flatten(),
                                                                lb_input1_layer=self.lb_input1[-1],
                                                                ub_input1_layer=self.ub_input1[-1],
                                                                lb_input2_layer=self.lb_input2[-1],
                                                                ub_input2_layer=self.ub_input2[-1])
        else:
            new_delta_lb, new_delta_ub = self.concretize_bounds(back_prop_struct=back_prop_struct,
                                                                delta_lb_layer=self.diff, 
                                                                delta_ub_layer=self.diff,
                                                                lb_input1_layer=self.lb_input1[-1],
                                                                ub_input1_layer=self.ub_input1[-1],
                                                                lb_input2_layer=self.lb_input2[-1],
                                                                ub_input2_layer=self.ub_input2[-1])

        self.check_lb_ub_correctness(lb=new_delta_lb, ub=new_delta_ub)
        delta_lb = (new_delta_lb if delta_lb is None else (torch.max(delta_lb, new_delta_lb)))
        delta_ub = (new_delta_ub if delta_ub is None else (torch.min(delta_ub, new_delta_ub)))
        return delta_lb, delta_ub

    # Run existing implementation of diffpoly that only uses backsubstitution
    # at the end of affine layers it is suboptimal.
    def run_only_affine_back_substitution(self):
        delta_lbs = []
        delta_ubs = []
        if self.net is None:
            raise ValueError("Passed network can not be none")
        
                
        if len(self.lb_input1) - 1 != len(self.linear_conv_layer_indices) or len(self.ub_input1) - 1 != len(self.linear_conv_layer_indices):
            raise ValueError("Input1 bounds do not match")
        if len(self.lb_input2) - 1 != len(self.linear_conv_layer_indices) or len(self.ub_input2) - 1 != len(self.linear_conv_layer_indices):
            raise ValueError("Input2 bounds do not match")

        for linear_layer_index, layer_index in enumerate(self.linear_conv_layer_indices):
            curr_delta_lb, curr_delta_ub = self.back_substitution_affine_only(layer_index=layer_index, 
                                                                    linear_layer_index=linear_layer_index,
                                                                   delta_lbs=delta_lbs, delta_ubs=delta_ubs)
            brute_delta_lb = self.lb_input1[linear_layer_index] - self.ub_input2[linear_layer_index]
            brute_delta_ub = self.ub_input1[linear_layer_index] - self.lb_input2[linear_layer_index]

            curr_delta_lb = torch.max(brute_delta_lb, curr_delta_lb)
            curr_delta_ub = torch.min(brute_delta_ub, curr_delta_ub)
            delta_lbs.append(curr_delta_lb)
            delta_ubs.append(curr_delta_ub)

        return delta_lbs, delta_ubs

    def back_substitution_full(self, layer_idx, delta_lbs, delta_ubs):
        if layer_idx != len(delta_lbs) or layer_idx != len(delta_ubs):
            raise ValueError("Size of lower bounds computed in previous layers don't match")
        back_prop_struct = None
        delta_lb = None
        delta_ub = None
        # Assuming the network has alternate affine and activation layer.
        linear_layer_index = layer_idx // 2
        for i in reversed(range(layer_idx + 1)):
            if back_prop_struct is not None:
                new_delta_lb, new_delta_ub = self.concretize_bounds(back_prop_struct=back_prop_struct,delta_lb_layer=delta_lbs[i], 
                                                                    delta_ub_layer=delta_ubs[i],
                                                         lb_input1_layer=self.lb_input1[i], 
                                                ub_input1_layer=self.ub_input1[i], 
                                                lb_input2_layer=self.lb_input2[i], 
                                                ub_input2_layer=self.ub_input2[i], layer_idx=layer_idx)
                self.check_lb_ub_correctness(lb=new_delta_lb, ub=new_delta_ub)
                delta_lb = (new_delta_lb if delta_lb is None else (torch.max(delta_lb, new_delta_lb)))
                delta_ub = (new_delta_ub if delta_ub is None else (torch.min(delta_ub, new_delta_ub)))

            if back_prop_struct is None:
                back_prop_struct = self.initialize_back_prop_struct(layer_idx=linear_layer_index)
            curr_layer = self.net[i]
            back_prop_struct = self.handle_layer(prop_struct=back_prop_struct, layer=curr_layer, linear_layer_idx=linear_layer_index,
                                                  layer_idx=i-1, delta_lbs=delta_lbs, delta_ubs=delta_ubs)
            if curr_layer.type in [LayerType.Linear, LayerType.Conv2D]:
                linear_layer_index -= 1
        #print(self.eps)
        if self.monotone:         
            new_delta_lb, new_delta_ub = self.concretize_bounds(back_prop_struct=back_prop_struct,
                                                                #delta_lb_layer=torch.zeros(12),#-self.eps * torch.nn.functional.one_hot(self.noise_ind[0], 12).flatten(), 
                                                                #delta_ub_layer=self.eps * torch.nn.functional.one_hot(self.noise_ind[0], 12).flatten(),
                                                                delta_lb_layer = torch.zeros(87),
                                                                delta_ub_layer = self.eps * torch.nn.functional.one_hot(torch.tensor(self.monotone_prop), 87).flatten(),
                                                                lb_input1_layer=self.lb_input1[-1],
                                                                ub_input1_layer=self.ub_input1[-1],
                                                                lb_input2_layer=self.lb_input2[-1],
                                                                ub_input2_layer=self.ub_input2[-1])
        else:
            new_delta_lb, new_delta_ub = self.concretize_bounds(back_prop_struct=back_prop_struct,
                                                                delta_lb_layer=self.diff, 
                                                                delta_ub_layer=self.diff,
                                                                lb_input1_layer=self.lb_input1[-1],
                                                                ub_input1_layer=self.ub_input1[-1],
                                                                lb_input2_layer=self.lb_input2[-1],
                                                                ub_input2_layer=self.ub_input2[-1])

        self.check_lb_ub_correctness(lb=new_delta_lb, ub=new_delta_ub)
        delta_lb = (new_delta_lb if delta_lb is None else (torch.max(delta_lb, new_delta_lb)))
        delta_ub = (new_delta_ub if delta_ub is None else (torch.min(delta_ub, new_delta_ub)))
            
        return delta_lb, delta_ub

    def run_full_back_substitution(self):

        delta_lbs = []
        delta_ubs = []

        # Check the validity of inputs.
        if self.net is None:
            raise ValueError("Passed network can not be none")
        if len(self.lb_input1) - 1 != len(self.net) or len(self.ub_input1) - 1 != len(self.net):
            raise ValueError("Input1 bounds do not match")
        if len(self.lb_input2) - 1 != len(self.net) or len(self.ub_input2) - 1 != len(self.net):
            raise ValueError("Input1 bounds do not match")
        
        for layer_idx, layer in enumerate(self.net):
            #print(layer_idx, layer.type)
            if layer.type is LayerType.ReLU:
                self.lb_input1[layer_idx] = torch.max(self.lb_input1[layer_idx], torch.zeros(self.lb_input1[layer_idx].size(),
                                                                                              device=self.device))
                self.lb_input2[layer_idx] = torch.max(self.lb_input2[layer_idx], torch.zeros(self.lb_input2[layer_idx].size(),
                                                                                              device=self.device))
            elif layer.type is LayerType.Sigmoid and layer_idx > 0:
                self.lb_input1[layer_idx] = torch.max(self.lb_input1[layer_idx], torch.sigmoid(self.lb_input1[layer_idx-1]))
                self.lb_input2[layer_idx] = torch.max(self.lb_input2[layer_idx], torch.sigmoid(self.lb_input2[layer_idx-1]))

                self.ub_input1[layer_idx] = torch.min(self.ub_input1[layer_idx], torch.sigmoid(self.ub_input1[layer_idx-1]))
                self.ub_input2[layer_idx] = torch.min(self.ub_input2[layer_idx], torch.sigmoid(self.ub_input2[layer_idx-1]))
            

            curr_delta_lb, curr_delta_ub = self.back_substitution_full(layer_idx=layer_idx, delta_lbs=delta_lbs,
                                                                       delta_ubs=delta_ubs)
        
        # Code for full back substitution.
            brute_delta_lb = self.lb_input1[layer_idx] - self.ub_input2[layer_idx]
            brute_delta_ub = self.ub_input1[layer_idx] - self.lb_input2[layer_idx]
            
            curr_delta_lb = torch.max(brute_delta_lb, curr_delta_lb)
            curr_delta_ub = torch.min(brute_delta_ub, curr_delta_ub)
            delta_lbs.append(curr_delta_lb)
            delta_ubs.append(curr_delta_ub)
            if self.lightweight_diffpoly is True:
                break

        return delta_lbs, delta_ubs

    def run(self):
        with torch.no_grad():
            for ind, layer in enumerate(self.net):
                if layer.type in [LayerType.Linear, LayerType.Conv2D]:
                    self.linear_conv_layer_indices.append(ind)
            
            if self.use_all_layers is True:
                return self.run_full_back_substitution()
            else:
                return self.run_only_affine_back_substitution()

