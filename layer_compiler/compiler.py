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

        for mm in range(fit_m):
            for nn in range(fit_n):
                # single tile for output matrix
                for kk in range(fit_k):
                    input_load = Inst(Op.LOAD_TILE, size=mmunit.width*mmunit.height, buf=Buf.UBUF)
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                    inst.append(input_load)
                    inst.append(weight_load)
                    inst.append(gemm_op)
                if outer_m == 1:
                    input_load = Inst(Op.LOAD_TILE, size=mmunit.width*left_k, buf=Buf.UBUF)
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.depth*left_k, buf=Buf.WBUF)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=left_k, N=mmunit.depth, depend=[input_load, weight_load])
                    inst.append(input_load)
                    inst.append(weight_load)
                    inst.append(gemm_op)
                vect_op = Inst(Op.VECTOR_OP, size=mmunit.depth, depend=[gemm_op])
                inst.append(vect_op)

        if outer_m == 1:
            for nn in range(fit_n):
                for kk in range(fit_k):
                    input_load = Inst(Op.LOAD_TILE, size=left_m*mmunit.height, buf=Buf.UBUF)
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    gemm_op = Inst(Op.GEMM_OP, M=left_m, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                    inst.append(input_load)
                    inst.append(weight_load)
                    inst.append(gemm_op)
                if outer_m == 1:
                    input_load = Inst(Op.LOAD_TILE, size=left_m*left_k, buf=Buf.UBUF)
                    weight_load = Inst(Op.LOAD_TILE, size=left_k*mmunit.depth, buf=Buf.WBUF)
                    gemm_op = Inst(Op.GEMM_OP, M=left_m, K=left_k, N=mmunit.depth, depend=[input_load, weight_load])
                    inst.append(input_load)
                    inst.append(weight_load)
                    inst.append(gemm_op)
                vect_op = Inst(Op.VECTOR_OP, size=mmunit.depth, depend=[gemm_op])
                inst.append(vect_op)

        if outer_n == 1:
            for mm in range(fit_m):
                for kk in range(fit_k):
                    input_load = Inst(Op.LOAD_TILE, size=mmunit.width*mmunit.height, buf=Buf.UBUF)
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*left_n, buf=Buf.WBUF)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=left_n, depend=[input_load, weight_load])
                    inst.append(input_load)
                    inst.append(weight_load)
                    inst.append(gemm_op)
                if outer_m == 1:
                    input_load = Inst(Op.LOAD_TILE, size=mmunit.width*left_k, buf=Buf.UBUF)
                    weight_load = Inst(Op.LOAD_TILE, size=left_k*left_n, buf=Buf.WBUF)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=left_k, N=left_n, depend=[input_load, weight_load])
                    inst.append(input_load)
                    inst.append(weight_load)
                    inst.append(gemm_op)
                vect_op = Inst(Op.VECTOR_OP, size=left_n, depend=[gemm_op])
                inst.append(vect_op)

        if outer_m == 1 and outer_n == 1:
            for kk in range(fit_k):
                input_load = Inst(Op.LOAD_TILE, size=left_m*mmunit.height, buf=Buf.UBUF)
                weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*left_n, buf=Buf.WBUF)
                gemm_op = Inst(Op.GEMM_OP, M=left_m, K=mmunit.height, N=left_n, depend=[input_load, weight_load])
                inst.append(input_load)
                inst.append(weight_load)
                inst.append(gemm_op)
            if outer_m == 1:
                input_load = Inst(Op.LOAD_TILE, size=left_m*left_k, buf=Buf.UBUF)
                weight_load = Inst(Op.LOAD_TILE, size=left_k*left_n, buf=Buf.WBUF)
                gemm_op = Inst(Op.GEMM_OP, M=left_m, K=left_k, N=left_n, depend=[input_load, weight_load])
                inst.append(input_load)
                inst.append(weight_load)
                inst.append(gemm_op)
            vect_op = Inst(Op.VECTOR_OP, size=left_n, depend=[gemm_op])
            inst.append(vect_op)

    if layer.layer_type == Type.LSTM:
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
        


class Inst:
    def __init__(self, inst_type, size=None, buf=None, M=None, K=None, N=None, depend=None):
        self.inst_type = inst_type
        self.done = False
        self.depend = []
        
        if depend != None:
            self.depend.extend(depend)

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
            if all((size)):
                self.size = size
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

        self.estimated = None





