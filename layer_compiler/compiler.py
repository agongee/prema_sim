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
        


class Inst:
    def __init__(self, inst_type, size=None, buffer=None, in_row=None, in_col=None, out_row=None, out_col=None, op_type=None):
        self.inst_type = inst_type

        if inst_type == Op.LOAD_TILE:
            if all((size, buffer)):
                self.size = size
                self.base = -1
                self.buffer = -1
            else:
                print("Missing Argument!")
        elif inst_type == Op.GEMM_OP:
            if all((in_row, in_col, out_row, out_col)):
                self.in_row = in_row
                self.in_col = in_col
                self.out_row = out_row
                self.out_col = out_col
            else:
                print("Missing Argument!")
        elif inst_type == Op.VECTOR_OP:
            if all((size, op_type)):
                self.size = size
                self.op_type = op_type
        elif inst_type == Op.STORE_TILE:
            if all((size, buffer)):
                self.size = size
                self.base = -1
                self.buffer = -1
        '''
        elif inst_type = OP.CONV_OP:
            if all()
        '''

# PCB for the given NN
class NN:
    def __init__(self, inst, priority, layer_type, nnid):
        self.inst = []
        self.inst.extend(inst)
        self.priority = priority
        self.layer_type = layer_type
        self.nnid = nnid
        self.pc = 0






