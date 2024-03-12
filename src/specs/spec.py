import torch

from src.specs.properties.acasxu import get_acas_spec
from src.specs.property import Property, InputSpecType, OutSpecType
from src.specs.out_spec import Constraint
from src.util import prepare_data, get_net
from src.common import Status
from src.common.dataset import Dataset
from src.network_converters.network_conversion_helper import convert_model, is_linear
import pandas as pd
import numpy as np
from torchvision.transforms import Normalize as norm
from src.config import is_linear_model
import sklearn.preprocessing as preprocessing

'''
Specification holds upper bound and lower bound on ranges for each dimension.
In future, it can be extended to handle other specs such as those for rotation 
or even the relu stability could be part of specification.
'''


class Spec:
    def __init__(self, input_spec, relu_spec=None, parent=None, status=Status.UNKNOWN):
        self.input_spec = input_spec
        self.relu_spec = relu_spec
        self.children = []
        self.status = status
        self.lb = 0
        self.active_relus = []
        self.inactive_relus = []
        self.last_feature_lb = []
        self.last_feature_ub = []
        self.chosen_split = None
        self.parent = parent
        self.eta_norm = None
        if parent is not None and parent.status == Status.VERIFIED:
            self.status = Status.VERIFIED

    # Custom comparator between sepecs
    def __lt__(self, other):
        if self.eta_norm is None or other.eta_norm is None:
            return True
        elif self.eta_norm >= other.eta_norm:
           return True
        else:
            return True 

    def update_feature_bounds(self, lb, ub):
        self.last_feature_lb = lb
        self.last_feature_ub = ub
    
    def get_feature_bounds(self):
        return self.last_feature_lb, self.last_feature_ub

    def update_status(self, status, lb, eta_norm=None, 
                active_relus=None, inactive_relus=None):
        self.status = status
        if lb is None:
            self.lb = 0
        else:
            self.lb = lb
        if eta_norm is not None:
            self.eta_norm = eta_norm
        if active_relus is not None:
            self.active_relus = active_relus
        if inactive_relus is not None:
            self.inactive_relus = inactive_relus
    
    def get_perturbation_bound(self):
        if self.eta_norm is None or self.lb < 0:
            return None
        else:
            return self.lb / self.eta_norm


    def reset_status(self):
        self.status = Status.UNKNOWN
        self.lb = 0




class SpecList(list):
    def check_perturbation_bound(self, spec, perturbation_bound=None):
        spec_perturbation_bound = spec.get_perturbation_bound()
        if perturbation_bound is None or spec_perturbation_bound is None:
            return False
        if perturbation_bound is None or spec_perturbation_bound is None:
            return False
        else:
            if (spec_perturbation_bound < perturbation_bound):
                return True
            else:
                return False

    def prune(self, split, split_score=None, inp_template=None, args=None, net=None, perturbation_bound=None):
        new_spec_list = SpecList()
        verified_specs = SpecList()


        for spec in self:

            if spec.status == Status.UNKNOWN or self.check_perturbation_bound(spec, perturbation_bound=perturbation_bound):
                add_spec = spec.split_spec(split, split_score=split_score,
                                           inp_template=inp_template,
                                           args=args, net=net)
                if add_spec is None:
                    return None, None
                # if spec.status != Status.UNKNOWN:
                #     print("Status:", spec.status)
                new_spec_list += SpecList(add_spec)
            else:
                verified_specs.append(spec)
        return new_spec_list, verified_specs


def create_relu_spec(unstable_relus):
    relu_mask = {}

    for layer in range(len(unstable_relus)):
        for id in unstable_relus[layer]:
            relu_mask[(layer, id)] = 0

    return Reluspec(relu_mask)


def score_relu_grad(spec, prop, net=None):
    """
    Gives a score to each relu based on its gradient. Higher score indicates higher preference while splitting.
    """
    relu_spec = spec.relu_spec
    relu_mask = relu_spec.relu_mask

    # Collect all relus that are not already split
    relu_spec.relu_score = {}

    # TODO: support CIFAR10
    ilb = prop.input_lb
    inp = ilb.reshape(1, 1, 28, 28)

    # Add all relu layers for which we need gradients
    layers = {}
    for relu in relu_mask.keys():
        layers[relu[0]] = True

    grad_map = {}

    # use ilb and net to get the grad for each neuron
    for layer in layers.keys():
        x = net[:layer * 2 + 2](inp).detach()
        x.requires_grad = True

        y = net[layer * 2 + 2:](x)
        y.mean().backward()

        grad_map[layer] = x.grad[0]

    for relu in relu_mask.keys():
        relu_spec.relu_score[relu] = abs(grad_map[relu[0]][relu[1]])

    return relu_spec.relu_score


def score_relu_esip(zono_transformer):
    """
    The relu score here is similar to the direct score defined in DeepSplit paper
    https://www.ijcai.org/proceedings/2021/0351.pdf
    """
    center = zono_transformer.centers[-1]
    cof = zono_transformer.cofs[-1]
    cof_abs = torch.sum(torch.abs(cof), dim=0)
    lb = center - cof_abs

    adv_index = torch.argmin(lb)
    relu_score = {}

    for noise_index, relu_index in zono_transformer.map_for_noise_indices.items():
        # Score relu based on effect on one label
        relu_score[relu_index] = torch.abs(cof[noise_index, adv_index])

        # Score relu based on effect on all label
        # relu_score[relu_index] = torch.sum(torch.abs(cof[noise_index, :]))

    return relu_score

def process_input_for_target_label(inputs, labels, target_label, target_count=0):
    new_inputs = []
    new_labels = []
    count = 0
    if target_label is None:
        return inputs, labels
    for i in range(len(inputs)):
        if labels[i].item() is target_label and count < target_count:
            new_inputs.append(inputs[i])
            new_labels.append(labels[i])
            count += 1
    new_inputs = torch.stack(new_inputs)
    new_labels = torch.stack(new_labels)
    return new_inputs, new_labels

def process_input_for_sink_label(inputs, labels, sink_label, target_count=0):
    new_inputs = []
    new_labels = []
    count = 0
    for i in range(len(inputs)):
        if labels[i] is not sink_label and count < target_count:
            new_inputs.append(inputs[i])
            new_labels.append(labels[i])
            count += 1
    new_inputs = torch.stack(new_inputs)
    new_labels = torch.stack(new_labels)
    return new_inputs, new_labels


def process_input_for_binary(inputs, labels, target_count=0):
    new_inputs = []
    new_labels = []
    count = 0
    binary_label = [0, 1]
    for i in range(len(inputs)):
        if labels[i] in binary_label and count < target_count:
            new_inputs.append(inputs[i])
            new_labels.append(labels[i])
            count += 1
    new_inputs = torch.stack(new_inputs)
    new_labels = torch.stack(new_labels)
    return new_inputs, new_labels

def remove_unclassified_images(inputs, labels, dataset, net_name):
    if net_name == '':
        return inputs, labels

    model = get_net(net_name, dataset)
    try:
        with torch.no_grad():
            only_net_name = net_name.split('/')[-1]
            converted_model = convert_model(model, remove_last_layer=False, all_linear=is_linear(net_name=only_net_name))
            mean, std = get_mean_std(dataset=dataset, net_name=net_name)
            norm_transform = norm(mean, std) 
            inputs_normalised = norm_transform(inputs)
            outputs = converted_model(inputs_normalised)
            output_labels = torch.max(outputs, axis=1)[1]
            # print(f'matching tensor {output_labels == labels}')
            inputs = inputs[output_labels == labels]
            labels = labels[output_labels == labels]
            return inputs, labels
    except:
        return inputs, labels

def get_specs(dataset, spec_type=InputSpecType.LINF, eps=0.01, count=None, 
              sink_label=None, debug_mode=False, monotone_prop = None, 
              monotone_inv = False, monotone_splits = 1, try_input_smoothing=False, count_per_prop=None,
              net_name=''):
    if debug_mode == True:
        return generate_debug_specs(count=count, eps=eps)
    if dataset == Dataset.MNIST or dataset == Dataset.CIFAR10:
        if spec_type == InputSpecType.LINF:
            if count is None:
                count = 100
            testloader = prepare_data(dataset, batch_size=count)
            inputs, labels = next(iter(testloader))
            inputs, labels = process_input_for_target_label(inputs=inputs, labels=labels, 
                                                          target_label=sink_label, target_count=count)            
            props = get_linf_spec(inputs, labels, eps, dataset)
        elif spec_type == InputSpecType.PATCH:
            if count is None:
                count = 10
            testloader = prepare_data(dataset, batch_size=count)
            inputs, labels = next(iter(testloader))
            props = get_patch_specs(inputs, labels, eps, dataset, p_width=2, p_length=2)
            width = inputs.shape[2] - 2 + 1
            length = inputs.shape[3] - 2 + 1
            pos_patch_count = width * length
            specs_per_patch = pos_patch_count
            # labels = labels.unsqueeze(1).repeat(1, pos_patch_count).flatten()
        elif spec_type == InputSpecType.UAP:
            if try_input_smoothing is True:
                count //= count_per_prop
            testloader = prepare_data(dataset, batch_size=5*count)
            inputs, labels = next(iter(testloader))
            inputs, labels = remove_unclassified_images(inputs, labels, dataset, net_name)
            inputs, labels = inputs[:count], labels[:count]
            if try_input_smoothing is True:
                torch.manual_seed(1000)
                inputs = inputs.repeat_interleave(count_per_prop, dim=0)
                inputs += torch.rand(inputs.size()) * (eps / 20.0) 
                labels = labels.repeat_interleave(count_per_prop, dim=0)
            props = get_linf_spec(inputs, labels, eps, dataset, net_name=net_name)
        elif spec_type == InputSpecType.UAP_TARGETED:
            if try_input_smoothing is True:
                count //= count_per_prop
            testloader = prepare_data(dataset, batch_size=6*count)
            inputs, labels = next(iter(testloader))
            if try_input_smoothing is True:
                torch.manual_seed(1000)
                inputs = inputs.repeat_interleave(count_per_prop, dim=0)
                inputs += torch.rand(inputs.size()) * (eps / 4.5) 
                labels = labels.repeat_interleave(count_per_prop, dim=0)
            # inputs, labels = process_input_for_sink_label(inputs=inputs, labels=labels, 
            #                                               sink_label=sink_label, target_count=count)
            inputs, labels = remove_unclassified_images(inputs, labels, dataset, net_name)
            inputs, labels = inputs[:count], labels[:count]
            props = get_targeted_UAP_spec(inputs, labels, eps, dataset, net_name=net_name)
        elif spec_type == InputSpecType.UAP_BINARY:
            testloader = prepare_data(dataset, batch_size=20*count)
            inputs, labels = next(iter(testloader))
            inputs, labels = process_input_for_binary(inputs=inputs, labels=labels, target_count=count)
            inputs, labels = remove_unclassified_images(inputs, labels, dataset, net_name)            
            props = get_binary_uap_spec(inputs=inputs, labels=labels, eps=eps, dataset=dataset, net_name=net_name)   
        return props, inputs
    elif dataset == Dataset.HOUSING:
        test_dataset = pd.read_csv('./data/testing_data.csv', index_col=0)

        test_labels = test_dataset.pop('HousePrice')
        test_dataset, test_labels = torch.tensor(np.array(test_dataset), dtype = torch.float32), torch.tensor(np.array(test_labels), dtype = torch.float32)
        test_dataset, test_labels = test_dataset[:count].reshape(count, 1, 12), test_labels[:count]
        props = get_monotone_spec(test_dataset, test_labels, eps, dataset, monotone_prop = monotone_prop, monotone_inv = monotone_inv, monotone_splits = monotone_splits)
        #props = get_linf_spec_test(test_dataset, test_labels, eps, dataset)
        #props = get_linf_spec_test(test_dataset, test_labels, eps, dataset)
        return props, test_dataset
        #return props, torch.cat((test_dataset, test_dataset), dim = 0)
    elif dataset == Dataset.ADULT:
        features = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
                "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                "Hours per week", "Country", "Target"]

        # This will download 3.8M
        original_train = pd.read_csv(filepath_or_buffer="data/adult.data", names=features, sep=r'\s*,\s*',
                                    engine='python', na_values="?")
        # This will download 1.9M
        original_test = pd.read_csv(filepath_or_buffer="data/adult.test", names=features, sep=r'\s*,\s*',
                                    engine='python', na_values="?", skiprows=1)

        num_train = len(original_train)
        original = pd.concat([original_train, original_test])
        roc_original = original
        labels = original['Target']
        labels = labels.replace('<=50K', 0).replace('>50K', 1)
        labels = labels.replace('<=50K.', 0).replace('>50K.', 1)

        # Redundant column
        del original["Education"]

        # Remove target variable
        del original["Target"]


        def data_transform(df):
            binary_data = pd.get_dummies(df)
            feature_cols = binary_data[binary_data.columns[:-2]]
            scaler = preprocessing.StandardScaler()
            scaled_data = scaler.fit_transform(feature_cols)

            continuous_columns = ["Age", "Education-Num", "Capital Gain", "Capital Loss", "Hours per week"]
            continuous_col_indices = [feature_cols.columns.get_loc(c) for c in continuous_columns if c in feature_cols.columns]

            continuous_means = scaler.mean_[continuous_col_indices]
            continuous_stdev = scaler.scale_[continuous_col_indices]

            scaled_data_df = pd.DataFrame(scaled_data, columns=feature_cols.columns)

            for (col, mean, std) in zip(continuous_columns, continuous_means, continuous_stdev):

                within_range_count = np.sum((scaled_data_df[col] >= -0.4) & (scaled_data_df[col] <= 0.4))
                total_count = len(scaled_data_df[col])
                proportion = within_range_count / total_count

            mask = np.all(np.logical_and(scaled_data[:, continuous_col_indices] >= -0.4,
                                        scaled_data[:, continuous_col_indices] <= 0.4), axis=1)

            proportion = np.mean(mask)

            data = pd.DataFrame(scaled_data, columns=feature_cols.columns)
            return data


        data = data_transform(original)

        #train_data = data[:num_train]
        #train_labels = labels.iloc[:num_train]
        test_data = data[num_train:]
        test_labels = labels.iloc[num_train:]
        test_data, test_labels = torch.tensor(np.array(test_data), dtype = torch.float32), torch.tensor(np.array(test_labels), dtype = torch.float32)
        test_data, test_labels = test_data[:count].reshape(count, 1, 87), test_labels[:count]
        props = get_monotone_spec(test_data, test_labels, eps, dataset, monotone_prop = monotone_prop, monotone_inv = monotone_inv, monotone_splits = monotone_splits)
        return props, test_data

    elif dataset == Dataset.ACAS:
        return get_acas_props(count), None
    else:
        raise ValueError("Unsupported specification dataset!")


def get_acas_props(count):
    props = []
    if count is None:
        count = 10
    for i in range(1, count + 1):
        props.append(get_acas_spec(i))
    return props

def generate_debug_specs(count=2, eps=1.0):
    inputs = []
    props = []
    for i in range(count):
        t = torch.zeros(2)
        if i % 2 == 0:
            t += torch.tensor([14, 11])
        else:
            t += torch.tensor([11, 14])
        inputs.append(t)
    for i, input in enumerate(inputs):
        ilb = input - eps
        iub = input + eps
        ilb = ilb.reshape(-1)
        iub = iub.reshape(-1)
        out_constr = Constraint(OutSpecType.LOCAL_ROBUST, label=torch.tensor([i%2]), debug_mode=True)
        props.append(Property(ilb, iub, InputSpecType.LINF, out_constr, dataset='dataset', input=input))
    return props, inputs

def get_monotone_spec(inputs, labels, eps, dataset, monotone_prop = None, monotone_inv = False, monotone_splits = 1):
    properties = []

    for i in range(len(inputs)):
        image = inputs[i]

        #ilb = image - eps
        #iub = image + eps
        ilb = image.clone()
        iub = image.clone()
        # iub[:, monotone_prop] += eps
        iub[:, monotone_prop] += eps
        # [0, 2, 3, 4, 5]
        # ilb[:, monotone_prop] -= eps
        # mean, std = get_mean_std(dataset)
        # ilb = (ilb - mean) / std
        # iub = (iub - mean) / std
        
        # image = (image - mean) / std
        # if monotone_inv:
        #     ilb = torch.cat((ilb, ilb_upper), dim = 0)
        #     iub = torch.cat((iub_lower, iub), dim = 0)
        # else:
        #     ilb = torch.cat((ilb_upper, ilb), dim = 0)
        #     iub = torch.cat((iub, iub_lower), dim = 0)
        # image = torch.cat((image, image), dim = 0)
        # mean, std = get_mean_std(dataset)
        # ilb = (ilb - mean) / std
        # iub = (iub - mean) / std
        
        # image = (image - mean) / std
        ilb = ilb.reshape(-1)
        iub = iub.reshape(-1)
        base = image.reshape(-1)
        out_constr = Constraint(OutSpecType.MONOTONE, label = labels[i])
        #print(monotone_inv)

        
        for i in range(monotone_splits):
            iub = ilb.clone()
            iub[monotone_prop] += eps/ monotone_splits
            properties.append(Property(ilb, iub, InputSpecType.LINF, out_constr, dataset, input=image, monotone = True, monotone_prop = monotone_prop))
            properties.append(Property(ilb, iub, InputSpecType.LINF, out_constr, dataset, input=image, monotone = True, monotone_prop = monotone_prop))
            ilb = iub.clone()
    

    return properties

def get_linf_spec_test(inputs, labels, eps, dataset):
    properties = []

    for i in range(len(inputs)):
        image = inputs[i]

        ilb = image - eps
        iub = image + eps

        mean, std = get_mean_std(dataset)
        ilb = (ilb - mean) / std
        iub = (iub - mean) / std
        
        image = (image - mean) / std

        ilb = ilb.reshape(-1)
        iub = iub.reshape(-1)
        out_constr = Constraint(OutSpecType.MONOTONE, label = labels[i])
        properties.append(Property(ilb, iub, InputSpecType.LINF, out_constr, dataset, input=image))

    return properties

# Get the specification for local linf robusteness.
# Untargeted uap are exactly same for local linf specs.
def get_linf_spec(inputs, labels, eps, dataset, net_name=''):
    properties = []

    for i in range(len(inputs)):
        image = inputs[i]

        ilb = image - eps
        iub = image + eps

        mean, std = get_mean_std(dataset, net_name=net_name)
        ilb = (ilb - mean) / std
        iub = (iub - mean) / std
        
        image = (image - mean) / std

        ilb = ilb.reshape(-1)
        iub = iub.reshape(-1)
        out_constr = Constraint(OutSpecType.LOCAL_ROBUST, label=labels[i])
        properties.append(Property(ilb, iub, InputSpecType.LINF, out_constr, dataset, input=image))

    return properties

def get_binary_uap_spec(inputs, labels, eps, dataset, net_name=''):
    properties = []

    for i in range(len(inputs)):
        image = inputs[i]

        ilb = image - eps
        iub = image + eps

        mean, std = get_mean_std(dataset, net_name=net_name)

        ilb = (ilb - mean) / std
        iub = (iub - mean) / std
        image = (image - mean) / std

        ilb = ilb.reshape(-1)
        iub = iub.reshape(-1)

        out_constr = Constraint(OutSpecType.LOCAL_ROBUST, label=labels[i], is_binary=True)
        properties.append(Property(ilb, iub, InputSpecType.LINF, out_constr, dataset, input=image))

    return properties


def get_targeted_UAP_spec(inputs, labels, eps, dataset, net_name=''):
    properties = []

    for i in range(len(inputs)):
        image = inputs[i]

        ilb = image - eps
        iub = image + eps

        mean, std = get_mean_std(dataset, net_name=net_name)

        ilb = (ilb - mean) / std
        iub = (iub - mean) / std
        image = (image - mean) / std

        ilb = ilb.reshape(-1)
        iub = iub.reshape(-1)

        out_constr = Constraint(OutSpecType.LOCAL_ROBUST, label=labels[i])
        properties.append(Property(ilb, iub, InputSpecType.UAP, out_constr, dataset, input=image, targeted = True))

    return properties



def get_patch_specs(inputs, labels, eps, dataset, p_width=2, p_length=2):
    width = inputs.shape[2] - p_width + 1
    length = inputs.shape[3] - p_length + 1
    pos_patch_count = width * length
    final_bound_count = pos_patch_count

    patch_idx = torch.arange(pos_patch_count, dtype=torch.long)[None, :]

    x_cord = torch.zeros((1, pos_patch_count), dtype=torch.long, requires_grad=False)
    y_cord = torch.zeros((1, pos_patch_count), dtype=torch.long, requires_grad=False)
    idx = 0
    for w in range(width):
        for l in range(length):
            x_cord[0, idx] = w
            y_cord[0, idx] = l
            idx = idx + 1

    # expand the list to include coordinates from the complete patch
    patch_idx = [patch_idx.flatten()]
    x_cord = [x_cord.flatten()]
    y_cord = [y_cord.flatten()]
    for w in range(p_width):
        for l in range(p_length):
            patch_idx.append(patch_idx[0])
            x_cord.append(x_cord[0] + w)
            y_cord.append(y_cord[0] + l)

    patch_idx = torch.cat(patch_idx, dim=0)
    x_cord = torch.cat(x_cord, dim=0)
    y_cord = torch.cat(y_cord, dim=0)

    # create masks for each data point
    mask = torch.zeros([1, pos_patch_count, inputs.shape[2], inputs.shape[3]],
                       dtype=torch.uint8)
    mask[:, patch_idx, x_cord, y_cord] = 1
    mask = mask[:, :, None, :, :]
    mask = mask.cpu()

    iubs = torch.clip(inputs + 1, min=0., max=1.)
    ilbs = torch.clip(inputs - 1, min=0., max=1.)

    iubs = torch.where(mask, iubs[:, None, :, :, :], inputs[:, None, :, :, :])
    ilbs = torch.where(mask, ilbs[:, None, :, :, :], inputs[:, None, :, :, :])

    mean, stds = get_mean_std(dataset)

    iubs = (iubs - mean) / stds
    ilbs = (ilbs - mean) / stds

    # (data, patches, spec)
    iubs = iubs.view(iubs.shape[0], iubs.shape[1], -1)
    ilbs = ilbs.view(ilbs.shape[0], ilbs.shape[1], -1)

    props = []

    for i in range(ilbs.shape[0]):
        out_constr = Constraint(OutSpecType.LOCAL_ROBUST, label=labels[i])
        props.append(Property(ilbs[i], iubs[i], InputSpecType.PATCH, out_constr, dataset, input=(inputs[i]-mean)/stds))
    return props




def get_mean_std(dataset, net_name=''):
    if dataset == Dataset.MNIST:
        if 'crown' in net_name or is_linear_model(net_name) :
            means = [0.0]
            stds = [1.0]
        else:
            means = [0.1307]
            stds = [0.3081]
    elif dataset == Dataset.CIFAR10:
        # For the model that is loaded from cert def this normalization was
        # used
        stds = [0.2023, 0.1994, 0.2010]
        means = [0.4914, 0.4822, 0.4465]
        # means = [0.0, 0.0, 0.0]
        # stds = [1, 1, 1]
    elif dataset == Dataset.ACAS:
        means = [19791.091, 0.0, 0.0, 650.0, 600.0]
        stds = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]
    elif dataset == Dataset.HOUSING:
        means = 0
        stds = 1
    else:
        raise ValueError("Unsupported Dataset!")
    return torch.tensor(means).reshape(-1, 1, 1), torch.tensor(stds).reshape(-1, 1, 1)
