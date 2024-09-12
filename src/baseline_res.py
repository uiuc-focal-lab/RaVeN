# Stores result of verifiers designed for verifying linf
# property.
class BaselineVerifierRes:
    def __init__(self, input, layer_lbs, layer_ubs, final_lb, final_ub, time=None, zono_center=None, 
                 zono_coef = None, lb_coef=None, lb_bias=None, target_ubs = None, target_centers = None, target_coefs = None, 
                 noise_ind = None, eps = None, last_conv_diff_struct=None, refined_lower_bnd=None) -> None:
        self.input = input
        self.layer_lbs = layer_lbs
        self.layer_ubs = layer_ubs
        self.final_lb = final_lb
        self.final_ub = final_ub
        self.time = time
        self.noise_ind = noise_ind
        # Populated if underlying verifier is zonotope
        self.zono_center = zono_center
        self.zono_coef = zono_coef
        # Populated if underlying verifier is auto_lirpa crown/deeppoly etc.
        self.lb_coef = lb_coef
        self.lb_bias = lb_bias
        self.target_ubs = target_ubs
        self.target_centers = target_centers 
        self.target_coefs = target_coefs
        self.eps = eps
        self.last_conv_diff_struct = last_conv_diff_struct
        self.refined_lower_bnd = refined_lower_bnd

    # Returns the refined bound if available else returns
    # existing lower bound.
    def get_refined_bnd(self):
        return self.refined_lower_bnd if self.refined_lower_bnd is not None else self.final_lb