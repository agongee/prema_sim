from layer_compiler.layer import Layer, Container
from unit import Mmunit, Vecunit
from enum_def import Type, Op, Buf


def compile(layer: Layer, mmunit: Mmunit):
    inst = []
    if layer.layer_type == Type.FC:
        '''
        Input: ACC * SH (m * k)
        Weight: SH * SW (k * n)
        '''
        m = layer.batch
        k = layer.in_dim
        n = layer.out_dim

        fit_m = int(m/mmunit.width)
        fit_k = int(k/mmunit.height)
        fit_n = int(n/mmunit.depth)
        
        left_m = m - fit_m*mmunit.width 
        left_k = k - fit_k*mmunit.height
        left_n = n - fit_n*mmunit.depth

        outer_m = 0
        outer_n = 0
        if left_m > 0:
            outer_m = 1
        if left_n > 0:
            outer_n = 1

        for mm in range(fit_m + outer_m):
            for nn in range(fit_n + outer_n):
                # single tile for output matrix
                for mmm in range(fit_m):
                    input_load = Inst(Op.LOAD_TILE, size=m*k, buf=Buf.UBUF)
                    weight_load = Inst(Op.LOAD_TILE, size=k*n, buf=Buf.WBUF)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth)
                    inst.append(input_load)
                    inst.append(weight_load)
                    inst.append(gemm_op)
                if outer_m == 1:
                    input_load = Inst(Op.LOAD_TILE, size=mmunit.height*left_m, buf=Buf.UBUF)
                    weight_load = Inst(Op.LOAD_TILE, size=, buf=Buf.WBUF)
                    gemm_op = Inst(Op.GEMM_OP)











        input_size = layer.in_dim*layer.batch
        weight_size = layer.in_dim*layer.out_dim

        # load input and weight
        input_load = Inst(Op.LOAD_TILE, size=input_size, buf=Buf.UBUF)
        weight_load = Inst(Op.LOAD_TILE, size=weight_size, buf=Buf.WBUF)
        inst.append(input_load)
        inst.append(weight_load)

        # matrix tiling
        m = layer.batch
        k = layer.in_dim
        n = layer.out_dim

        fit_m = int(m/mmunit.width)
        fit_k = int(k/mmunit.height)
        fit_n = int(n/mmunit.depth)

        for i in range(fit_m*fit_k*fit_n):
            input_base = fit_m

        
        


class Inst:
    def __init__(self, inst_type, size=None, buf=None, M=None, K=None, N=None, op_type=None):
        self.inst_type = inst_type

        if inst_type == Op.LOAD_TILE:
            if all((size, buf)):
                self.size = size
                self.base = -1
                self.buf = -1
            else:
                print("Missing Argument!")
        elif inst_type == Op.GEMM_OP:
            if all((M, N, K)):
                self.M = M
                self.N = N
                self.K = K
            else:
                print("Missing Argument!")
        elif inst_type == Op.VECTOR_OP:
            if all((size, op_type)):
                self.size = size
                self.op_type = op_type
        elif inst_type == Op.STORE_TILE:
            if all((size, buf)):
                self.size = size
                self.base = -1
                self.buf = -1
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






