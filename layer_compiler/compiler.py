from enum import Enum

from layer_compiler.layer import Layer, Container, Type

class Op(Enum):
    LOAD_TILE = 1
    GEMM_OP = 2
    CONV_OP = 3
    VECTOR_OP = 4
    STORE_TILE = 5

def compile(layer: Layer):
    inst = []
    if layer.layer_type == Type.FC:
        inst.append(f"LOAD_TILE {layer.in_dim}") # for input activation
        weight_size = layer.in_dim * layer.out_dim
        inst.append(f"LOAD_TILE {weight_size}") # for weight 
        inst.append(f"GEMM_OP   ")


class Inst:
    def __init__(self, inst_type, size=None):
        self.inst_type = inst_type


        if inst_type == Op.LOAD_TILE :
            if all((size)):
                self.size = size
                self.base = -1
                self.buffer = -1
            else:
                print("Missing Argument!")
    


class NN:
    def __init__(self, inst):
        self.inst = []
        self.inst.extend(inst)
        






