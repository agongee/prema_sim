from layer_compiler.layer import Layer, Container
from unit import Mmunit, Vecunit
from layer_compiler.enum_def import Type, Op, Buf


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

        # print(fit_m, fit_k, fit_n, left_m, left_k, left_n, outer_m, outer_n)
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
                for i in range(mmunit.depth):
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op)
                store_op = Inst(Op.STORE_TILE, size=mmunit.width*mmunit.depth, buf=Buf.UBUF, depend=[vect_op])

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
                for i in range(mmunit.depth):
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)
                store_op = Inst(Op.STORE_TILE, size=left_m*mmunit.depth, buf=Buf.UBUF, depend=[vect_op])
                inst.append(store_op)

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
                for i in range(left_n):
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op)
                store_op = Inst(Op.STORE_TILE, size=left_n*mmunit.width, buf=Buf.UBUF, depend=[vect_op])
                inst.append(store_op)

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
            for i in range(left_n):
                vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                inst.append(vect_op)
                
        store_op = Inst(Op.STORE_TILE, size=left_m*left_n, buf=Buf.UBUF, depend=[vect_op])
        inst.append(store_op)

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

        input_load = None
        hidden_load = None

    
                

        

    # Debug
    '''
    for i in inst:
        print(i)
    '''
    return inst
        


class Inst:
    def __init__(self, inst_type, size=None, buf=None, M=None, K=None, N=None, depend=None):
        self.inst_type = inst_type
        self.done = False
        self.depend = []

        self.size = None
        self.base = None
        self.buf = None
        self.M = None
        self.N = None
        self.K = None
        
        if depend != None:
            self.depend.extend(depend)

        if inst_type == Op.LOAD_TILE:
            if all((size, buf)):
                self.size = size
                self.base = -1
                self.buf = buf
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
            if size != None:
                self.size = size
        elif inst_type == Op.STORE_TILE:
            if all((size, buf)):
                self.size = size
                self.base = -1
                self.buf = buf
        '''
        elif inst_type = OP.CONV_OP:
            if all()
        '''

    def fetchable(self):
        return all(self.depend)

    def __bool__(self):
        return self.done

    def __str__(self):
        if self.inst_type == Op.LOAD_TILE:
            res = f"LOAD_TILE\t{self.size}"
            if self.buf == Buf.UBUF:
                return res + "\tUBUF"
            elif self.buf == Buf.WBUF:
                return res + "\tWBUF"
            return "DEBUG"
        elif self.inst_type == Op.STORE_TILE:
            res = f"STORE_TILE\t{self.size}"
            if self.buf == Buf.UBUF:
                return res + "\tUBUF"
            elif self.buf == Buf.WBUF:
                return res + "\tWBUF"
            return "DEBUG"
        elif self.inst_type == Op.GEMM_OP:
            return f"GEMM_OP\t{self.M}\t{self.K}\t{self.N}"
        elif self.inst_type == Op.VECTOR_OP:
            return f"VECTOR_OP\t{self.size}"

        return "DEBUG"


# PCB for the given NN
class NN:
    def __init__(self, priority, nnid, mmunit, dispatch_time):
        self.inst = []
        self.container = None
        self.priority = priority
        self.token = 0
        if self.priority == 0:
            self.token = 1
        elif self.priority == 1:
            self.token = 3
        elif self.priority == 2:
            self.token = 9
        self.nnid = nnid
        self.mmunit = mmunit
        self.pc = 0
        self.done = False
        self.running = False
        self.dispatched = False
        self.dispatch_time = dispatch_time
        if dispatch_time == 0:
            self.dispatched = True
        self.estimated = 0
        self.waited = 0
        self.runned = 0
        self.remaining = 0
        self.context = None

    def container_to_inst(self, container: Container):
        self.container = container
        for i in container.container:
            print(i)
            self.inst.extend(compile(i, self.mmunit))
        self.estimated = self.container.estimate(self.mmunit.height, self.mmunit.width, self.mmunit.depth)
        print("DEBUG: ", self.estimated)

    def fetch1(self):
        return self.inst[self.pc]

    def fetch2(self):
        temp_pc = self.pc
        self.pc += 1
        if self.pc >= len(self.inst):
            self.done = True
            self.running = False
        return self.inst[temp_pc]

    def dispatch_nn(self):
        if self.dispatch_time == 1:
            self.dispatch_time = 0
            self.dispatched = True
        elif self.dispatch_time > 0:
            self.dispatch_time -= 1
            self.dispatch = False
        else:
            self.dispatch_time = 0
            self.dispatch = False

    def inst_str(self):
        res = f"  NNID: {self.nnid}\n"

        for i in self.inst:
            res += str(i)
            res += '\n'

        res += '\n'

        return res

    def str_pre(self):
        res = f"  NNID: {self.nnid}\n"

        if self.priority == 0:
            res += f"  Priority: low\n"
        elif self.priority == 1:
            res += f"  Priority: medium\n"
        elif self.priority == 2:
            res += f"  Priority: high\n"

        res += f"  To be Dispatched: {self.dispatch_time}\n"
        res += f"  Estimated: {self.estimated}\n"
        res += "  Container Information:\n"
        res += str(self.container)
        res += "\n"

        return res

    def str_current(self):
        res = f"  NNID: {self.nnid}\n"

        if self.priority == 0:
            res += f"  Priority: low\n"
        elif self.priority == 1:
            res += f"  Priority: medium\n"
        elif self.priority == 2:
            res += f"  Priority: high\n"
        
        res += f"  To be Dispatched: {self.dispatch_time}\n"
        res += f"  Estimated Time: {self.estimated}\n"
        if self.done:
            res += "  Done\n"
        elif self.running:
            res += f"  Running\n"
        else:
            res += f"  Wating\n"
        res += f"  Runned: {self.runned}\n"
        res += f"  Waited: {self.waited}\n"
        res += f"  Processing: {self.pc}/{len(self.inst)}\n"
        res += "\n"

        return res

    
    def __bool__(self):
        return self.done