import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from args import args as parser_args

DenseConv = nn.Conv2d

def sparseFunction(x, s, activation=torch.relu, f=torch.sigmoid):
    return torch.sign(x)*activation(torch.abs(x)-f(s))

def initialize_sInit():

    if parser_args.sInit_type == "constant":
        return parser_args.sInit_value*torch.ones([1, 1])

class PseudoRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output
        return grad_x, None

pseudoRelu = PseudoRelu.apply

def softThresholdinv(x, s):
    return torch.sign(x) * (torch.abs(x) + s)

def softThresholdmod(x, s):
    return torch.sign(x) * pseudoRelu(torch.abs(x)-s)

class STRConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #self.activation = pseudoRelu
        with torch.no_grad():
            if parser_args.sparse_function == 'identity':
                self.mapping = lambda x: x
            elif parser_args.sparse_function == 'stmod':
                if parser_args.gradual is None:
                    self.mapping = lambda x: softThresholdmod(x, parser_args.flat_width)
                else:
                    self.mapping = lambda x: x
            
            if parser_args.sparse_function == 'stmod' and parser_args.gradual is None:
                self.weight.data = softThresholdinv(self.weight.data, parser_args.flat_width)
    
    def forward(self, x):
        # In case STR is not training for the hyperparameters given in the paper, change sparseWeight to self.sparseWeight if it is a problem of backprop.
        # However, that should not be the case according to graph computation.
        
        sparseWeight = self.mapping(self.weight)
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def getSparsity(self): #, f=torch.sigmoid):
        #sparseWeight = sparseFunction(self.weight, self.sparseThreshold,  self.activation, self.f)
        sparseWeight = self.mapping(self.weight)
        temp = sparseWeight.detach().cpu()
        return (temp == 0).sum(), temp.numel()#, f(self.sparseThreshold).item()

    @torch.no_grad()
    def getSparseWeight(self):
        return self.mapping(self.weight)
        #return sparseFunction(self.weight, self.sparseThreshold, self.activation, self.f)
    
    @torch.no_grad()
    def setFlatWidth(self, width):
        if parser_args.sparse_function == 'stmod':
            self.mapping = lambda x: softThresholdmod(x, width)

class ChooseEdges(autograd.Function):
    @staticmethod
    def forward(ctx, weight, prune_rate):
        output = weight.clone()
        _, idx = weight.flatten().abs().sort()
        p = int(prune_rate * weight.numel())
        # flat_oup and output access the same memory.
        flat_oup = output.flatten()
        flat_oup[idx[:p]] = 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class DNWConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        print(f"=> Setting prune rate to {prune_rate}")

    def forward(self, x):
        w = ChooseEdges.apply(self.weight, self.prune_rate)

        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return x

def GMPChooseEdges(weight, prune_rate):
    output = weight.clone()
    _, idx = weight.flatten().abs().sort()
    p = int(prune_rate * weight.numel())
    # flat_oup and output access the same memory.
    flat_oup = output.flatten()
    flat_oup[idx[:p]] = 0
    return output

class GMPConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        self.curr_prune_rate = 0.0
        print(f"=> Setting prune rate to {prune_rate}")

    def set_curr_prune_rate(self, curr_prune_rate):
        self.curr_prune_rate = curr_prune_rate

    def forward(self, x):
        w = GMPChooseEdges(self.weight, self.curr_prune_rate)
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return x
