from layer_compiler.layer import Layer, Container
from unit import Mmunit, Vecunit
from layer_compiler.enum_def import Type, Op, Buf
from sys import exit


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

        input_load = None
        '''
        if layer.previous_input:
            input_load = Inst(Op.NOP)        
        '''
        #print(fit_m, fit_k, fit_n, left_m, left_k, left_n, outer_m, outer_n, mmunit.width, mmunit.height, mmunit.depth)
        #input()
        for mm in range(fit_m):
            for nn in range(fit_n):
                # single tile for output matrix
                for kk in range(fit_k):
                    if not layer.previous_input:
                        input_load = Inst(Op.LOAD_TILE, size=mmunit.width*mmunit.height, buf=Buf.UBUF)
                        inst.append(input_load)
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    if not layer.previous_input:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load,weight_load])
                        inst.append(gemm_op)
                    else:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[ weight_load])
                        inst.append(gemm_op)                    
                if outer_m == 1:
                    if not layer.previous_input:
                        input_load = Inst(Op.LOAD_TILE, size=mmunit.width*left_k, buf=Buf.UBUF)
                        inst.append(input_load)
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load  )
                    if not layer.previous_input:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                        inst.append(gemm_op)
                    else:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op) 
                store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                inst.append(vect_op) 
                store_op = Inst(Op.STORE_TILE, size=mmunit.width*mmunit.depth, buf=Buf.UBUF, depend=[vect_op])
                inst.append(store_op)

        if outer_m == 1:
            for nn in range(fit_n):
                for kk in range(fit_k):
                    if not layer.previous_input:
                        input_load = Inst(Op.LOAD_TILE, size=left_m*mmunit.height, buf=Buf.UBUF)
                        inst.append(input_load)
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    if not layer.previous_input:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                        inst.append(gemm_op)
                    else:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                if outer_m == 1:
                    if not layer.previous_input:
                        input_load = Inst(Op.LOAD_TILE, size=left_m*left_k, buf=Buf.UBUF)
                        inst.append(input_load)
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    if not layer.previous_input:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                        inst.append(gemm_op)
                    else:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                inst.append(vect_op)
                store_op = Inst(Op.STORE_TILE, size=left_m*mmunit.depth, buf=Buf.UBUF, depend=[vect_op])
                inst.append(store_op)

        if outer_n == 1:
            for mm in range(fit_m):
                for kk in range(fit_k):
                    if not layer.previous_input:
                        input_load = Inst(Op.LOAD_TILE, size=mmunit.width*mmunit.height, buf=Buf.UBUF)
                        inst.append(input_load)
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    if not layer.previous_input:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                        inst.append(gemm_op)
                    else:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)               
                if outer_m == 1:
                    if not layer.previous_input:
                        input_load = Inst(Op.LOAD_TILE, size=mmunit.width*left_k, buf=Buf.UBUF)
                        inst.append(input_load)
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    if not layer.previous_input:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                        inst.append(gemm_op)
                    else:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                inst.append(vect_op)
                store_op = Inst(Op.STORE_TILE, size=left_n*mmunit.width, buf=Buf.UBUF, depend=[vect_op])
                inst.append(store_op)

        if outer_m == 1 and outer_n == 1:
            for kk in range(fit_k):
                if not layer.previous_input:
                    input_load = Inst(Op.LOAD_TILE, size=left_m*mmunit.height, buf=Buf.UBUF)
                    inst.append(input_load)
                weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                inst.append(weight_load)
                if not layer.previous_input:
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                    inst.append(gemm_op)
                else:
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                    inst.append(gemm_op)
            if outer_m == 1:
                if not layer.previous_input:
                    input_load = Inst(Op.LOAD_TILE, size=left_m*left_k, buf=Buf.UBUF)
                    inst.append(input_load)
                weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                inst.append(weight_load)
                if not layer.previous_input:
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                    inst.append(gemm_op)
                else:
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                    inst.append(gemm_op)
            store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
            vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
            inst.append(vect_op)           
            store_op = Inst(Op.STORE_TILE, size=left_m*left_n, buf=Buf.UBUF, depend=[vect_op])
            inst.append(store_op)

    if layer.layer_type == Type.GEMM:
        '''
        Input: ACC * SH (m * k)
        Weight: SH * SW (k * n)
        '''
        batch = layer.batch
        m = layer.gemm_m
        k = layer.gemm_k
        n = layer.gemm_n

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
        '''
        if layer.previous_input:
            input_load = Inst(Op.NOP)        
        '''
        # print(fit_m, fit_k, fit_n, left_m, left_k, left_n, outer_m, outer_n)
        for b in range(batch):
            for mm in range(fit_m):
                for nn in range(fit_n):
                    # single tile for output matrix
                    for kk in range(fit_k):
                        if not layer.previous_input:
                            input_load = Inst(Op.LOAD_TILE, size=mmunit.width*mmunit.height, buf=Buf.UBUF)
                            inst.append(input_load)
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        if not layer.previous_input:
                            gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load,weight_load])
                            inst.append(gemm_op)
                        else:
                            gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[ weight_load])
                            inst.append(gemm_op)                    
                    if outer_m == 1:
                        if not layer.previous_input:
                            input_load = Inst(Op.LOAD_TILE, size=mmunit.width*left_k, buf=Buf.UBUF)
                            inst.append(input_load)
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load  )
                        if not layer.previous_input:
                            gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                            inst.append(gemm_op)
                        else:
                            gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                            inst.append(gemm_op) 
                    store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    

            if outer_m == 1:
                for nn in range(fit_n):
                    for kk in range(fit_k):
                        if not layer.previous_input:
                            input_load = Inst(Op.LOAD_TILE, size=left_m*mmunit.height, buf=Buf.UBUF)
                            inst.append(input_load)
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        if not layer.previous_input:
                            gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                            inst.append(gemm_op)
                        else:
                            gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                            inst.append(gemm_op)
                    if outer_m == 1:
                        if not layer.previous_input:
                            input_load = Inst(Op.LOAD_TILE, size=left_m*left_k, buf=Buf.UBUF)
                            inst.append(input_load)
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        if not layer.previous_input:
                            gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                            inst.append(gemm_op)
                        else:
                            gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                            inst.append(gemm_op)
                    store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    

            if outer_n == 1:
                for mm in range(fit_m):
                    for kk in range(fit_k):
                        if not layer.previous_input:
                            input_load = Inst(Op.LOAD_TILE, size=mmunit.width*mmunit.height, buf=Buf.UBUF)
                            inst.append(input_load)
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        if not layer.previous_input:
                            gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                            inst.append(gemm_op)
                        else:
                            gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                            inst.append(gemm_op)               
                    if outer_m == 1:
                        if not layer.previous_input:
                            input_load = Inst(Op.LOAD_TILE, size=mmunit.width*left_k, buf=Buf.UBUF)
                            inst.append(input_load)
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        if not layer.previous_input:
                            gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                            inst.append(gemm_op)
                        else:
                            gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                            inst.append(gemm_op)
                    store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    

            if outer_m == 1 and outer_n == 1:
                for kk in range(fit_k):
                    if not layer.previous_input:
                        input_load = Inst(Op.LOAD_TILE, size=left_m*mmunit.height, buf=Buf.UBUF)
                        inst.append(input_load)
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    if not layer.previous_input:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                        inst.append(gemm_op)
                    else:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                if outer_m == 1:
                    if not layer.previous_input:
                        input_load = Inst(Op.LOAD_TILE, size=left_m*left_k, buf=Buf.UBUF)
                        inst.append(input_load)
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    if not layer.previous_input:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                        inst.append(gemm_op)
                    else:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    

    if layer.layer_type == Type.LSTM:
        '''
        Input: ACC * SH (m * k)
        Weight: SH * SW (k * n)
        '''
        m = layer.batch
        k_in = layer.in_dim
        k_h = layer.h_dim
        n = layer.h_dim

        fit_m = int(m/mmunit.width)
        fit_k_in = int(k_in/mmunit.height)
        fit_k_h = int(k_h/mmunit.height)
        fit_n = int(n/mmunit.depth)
        
        left_m = m - fit_m*mmunit.width 
        left_k_in = k_in - fit_k_in*mmunit.height
        left_k_h = k_h - fit_k_h*mmunit.height
        left_n = n - fit_n*mmunit.depth

        outer_m = 0
        outer_n = 0
        if left_m > 0:
            outer_m = 1
        if left_n > 0:
            outer_n = 1

        input_load = None
        hidden_load = None

        # it computation
        # 1) W_xh_i * x_t
        for mm in range(fit_m):
            for nn in range(fit_n):
                # single tile for output matrix
                for kk in range(fit_k_in):
                    if not layer.previous_input:
                        input_load = Inst(Op.LOAD_TILE, size=mmunit.width*mmunit.height, buf=Buf.UBUF)
                        inst.append(input_load)
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    if not layer.previous_input:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load,weight_load])
                        inst.append(gemm_op)
                    else:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[ weight_load])
                        inst.append(gemm_op)                    
                if outer_m == 1:
                    if not layer.previous_input:
                        input_load = Inst(Op.LOAD_TILE, size=mmunit.width*left_k_in, buf=Buf.UBUF)
                        inst.append(input_load)
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load  )
                    if not layer.previous_input:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                        inst.append(gemm_op)
                    else:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op) 
                store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])
                if layer.no_hidden:   
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op)

        if outer_m == 1:
            for nn in range(fit_n):
                for kk in range(fit_k_in):
                    if not layer.previous_input:
                        input_load = Inst(Op.LOAD_TILE, size=left_m*mmunit.height, buf=Buf.UBUF)
                        inst.append(input_load)
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    if not layer.previous_input:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                        inst.append(gemm_op)
                    else:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                if outer_m == 1:
                    if not layer.previous_input:
                        input_load = Inst(Op.LOAD_TILE, size=left_m*left_k_in, buf=Buf.UBUF)
                        inst.append(input_load)
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    if not layer.previous_input:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                        inst.append(gemm_op)
                    else:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])
                if layer.no_hidden:   
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)

        if outer_n == 1:
            for mm in range(fit_m):
                for kk in range(fit_k_in):
                    if not layer.previous_input:
                        input_load = Inst(Op.LOAD_TILE, size=mmunit.width*mmunit.height, buf=Buf.UBUF)
                        inst.append(input_load)
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    if not layer.previous_input:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                        inst.append(gemm_op)
                    else:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)               
                if outer_m == 1:
                    if not layer.previous_input:
                        input_load = Inst(Op.LOAD_TILE, size=mmunit.width*left_k_in, buf=Buf.UBUF)
                        inst.append(input_load)
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    if not layer.previous_input:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                        inst.append(gemm_op)
                    else:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])  
                if layer.no_hidden:  
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op)

        if outer_m == 1 and outer_n == 1:
            for kk in range(fit_k_in):
                if not layer.previous_input:
                    input_load = Inst(Op.LOAD_TILE, size=left_m*mmunit.height, buf=Buf.UBUF)
                    inst.append(input_load)
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    if not layer.previous_input:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                        inst.append(gemm_op)
                    else:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
            if outer_m == 1:
                if not layer.previous_input:
                    input_load = Inst(Op.LOAD_TILE, size=left_m*left_k_in, buf=Buf.UBUF)
                    inst.append(input_load)
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    if not layer.previous_input:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[input_load, weight_load])
                        inst.append(gemm_op)
                    else:
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
            store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])
            if layer.no_hidden:
                vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                inst.append(vect_op)     
                vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                inst.append(vect_op)        

        # 2) W_hh_i * h_t-1 , add all
        if not layer.no_hidden:
            for mm in range(fit_m):
                for nn in range(fit_n):
                    # single tile for output matrix
                    for kk in range(fit_k_h):
                        if not layer.previous_input:
                            input_load = Inst(Op.LOAD_TILE, size=mmunit.width*mmunit.height, buf=Buf.UBUF)
                            inst.append(input_load)
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)                    
                    if outer_m == 1:
                        if not layer.previous_input:
                            input_load = Inst(Op.LOAD_TILE, size=mmunit.width*left_k_h, buf=Buf.UBUF)
                            inst.append(input_load)
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load  )
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op) 
                    store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op) 

            if outer_m == 1:
                for nn in range(fit_n):
                    for kk in range(fit_k_h):
                        if not layer.previous_input:
                            input_load = Inst(Op.LOAD_TILE, size=left_m*mmunit.height, buf=Buf.UBUF)
                            inst.append(input_load)
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                    if outer_m == 1:
                        if not layer.previous_input:
                            input_load = Inst(Op.LOAD_TILE, size=left_m*left_k_h, buf=Buf.UBUF)
                            inst.append(input_load)
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                    store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)

            if outer_n == 1:
                for mm in range(fit_m):
                    for kk in range(fit_k_h):
                        if not layer.previous_input:
                            input_load = Inst(Op.LOAD_TILE, size=mmunit.width*mmunit.height, buf=Buf.UBUF)
                            inst.append(input_load)
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)               
                    if outer_m == 1:
                        if not layer.previous_input:
                            input_load = Inst(Op.LOAD_TILE, size=mmunit.width*left_k_h, buf=Buf.UBUF)
                            inst.append(input_load)
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                    store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)

            if outer_m == 1 and outer_n == 1:
                for kk in range(fit_k_h):
                    if not layer.previous_input:
                        input_load = Inst(Op.LOAD_TILE, size=left_m*mmunit.height, buf=Buf.UBUF)
                        inst.append(input_load)
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                if outer_m == 1:
                    if not layer.previous_input:
                        input_load = Inst(Op.LOAD_TILE, size=left_m*left_k_h, buf=Buf.UBUF)
                        inst.append(input_load)
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                inst.append(vect_op)           
                vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                inst.append(vect_op)
                vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                inst.append(vect_op)


        # ft computation
        # 1) W_xh_f * x_t
        if not layer.no_hidden:
            for mm in range(fit_m):
                for nn in range(fit_n):
                    # single tile for output matrix
                    for kk in range(fit_k_in):
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[ weight_load])
                        inst.append(gemm_op)                    
                    if outer_m == 1:
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op) 
                    store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])
                    if layer.no_hidden:   
                        vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                        inst.append(vect_op)
                        vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                        inst.append(vect_op)

            if outer_m == 1:
                for nn in range(fit_n):
                    for kk in range(fit_k_in):
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                    if outer_m == 1:
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                    store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])
                    if layer.no_hidden:   
                        vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                        inst.append(vect_op)
                        vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                        inst.append(vect_op)

            if outer_n == 1:
                for mm in range(fit_m):
                    for kk in range(fit_k_in):
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)               
                    if outer_m == 1:
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                    store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])  
                    if layer.no_hidden:  
                        vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                        inst.append(vect_op)
                        vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                        inst.append(vect_op)

            if outer_m == 1 and outer_n == 1:
                for kk in range(fit_k_in):
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                    inst.append(gemm_op)
                if outer_m == 1:
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                    inst.append(gemm_op)
                store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])
                if layer.no_hidden:
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)           
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)           


            # 2) W_hh_f * h_t-1 , add all
            if not layer.no_hidden:
                for mm in range(fit_m):
                    for nn in range(fit_n):
                        # single tile for output matrix
                        for kk in range(fit_k_h):
                            weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                            inst.append(weight_load)
                            gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                            inst.append(gemm_op)                    
                        if outer_m == 1:
                            weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                            inst.append(weight_load  )
                            gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                            inst.append(gemm_op) 
                        store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                        vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                        inst.append(vect_op)
                        vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                        inst.append(vect_op) 
                        vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                        inst.append(vect_op) 

                if outer_m == 1:
                    for nn in range(fit_n):
                        for kk in range(fit_k_h):
                            weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                            inst.append(weight_load)
                            gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                            inst.append(gemm_op)
                        if outer_m == 1:
                            weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                            inst.append(weight_load)
                            gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                            inst.append(gemm_op)
                        store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                        vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                        inst.append(vect_op)
                        vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                        inst.append(vect_op)
                        vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                        inst.append(vect_op)

                if outer_n == 1:
                    for mm in range(fit_m):
                        for kk in range(fit_k_h):
                            weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                            inst.append(weight_load)
                            gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                            inst.append(gemm_op)               
                        if outer_m == 1:
                            weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                            inst.append(weight_load)
                            gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                            inst.append(gemm_op)
                        store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                        vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                        inst.append(vect_op)
                        vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                        inst.append(vect_op)
                        vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                        inst.append(vect_op)

                if outer_m == 1 and outer_n == 1:
                    for kk in range(fit_k_h):
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                    if outer_m == 1:
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                    store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)           
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)

        # ot compuatation
         # 1) W_xh_o * x_t
        for mm in range(fit_m):
            for nn in range(fit_n):
                # single tile for output matrix
                for kk in range(fit_k_in):
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[ weight_load])
                    inst.append(gemm_op)                    
                if outer_m == 1:
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                    inst.append(gemm_op) 
                store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])
                if layer.no_hidden:   
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op)

        if outer_m == 1:
            for nn in range(fit_n):
                for kk in range(fit_k_in):
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                    inst.append(gemm_op)
                if outer_m == 1:
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                    inst.append(gemm_op)
                store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])
                if layer.no_hidden:   
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)

        if outer_n == 1:
            for mm in range(fit_m):
                for kk in range(fit_k_in):
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                    inst.append(gemm_op)               
                if outer_m == 1:
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                    inst.append(gemm_op)
                store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])  
                if layer.no_hidden:  
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op)

        if outer_m == 1 and outer_n == 1:
            for kk in range(fit_k_in):
                weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                inst.append(weight_load)
                gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                inst.append(gemm_op)
            if outer_m == 1:
                weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                inst.append(weight_load)
                gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                inst.append(gemm_op)
            store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])
            if layer.no_hidden:
                vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                inst.append(vect_op)           
                vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                inst.append(vect_op)           


        # 2) W_hh_o * h_t-1 , add all
        if not layer.no_hidden:
            for mm in range(fit_m):
                for nn in range(fit_n):
                    # single tile for output matrix
                    for kk in range(fit_k_h):
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)                    
                    if outer_m == 1:
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load  )
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op) 
                    store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op) 
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op) 

            if outer_m == 1:
                for nn in range(fit_n):
                    for kk in range(fit_k_h):
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                    if outer_m == 1:
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                    store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)

            if outer_n == 1:
                for mm in range(fit_m):
                    for kk in range(fit_k_h):
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)               
                    if outer_m == 1:
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                    store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)

            if outer_m == 1 and outer_n == 1:
                for kk in range(fit_k_h):
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                    inst.append(gemm_op)
                if outer_m == 1:
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                    inst.append(gemm_op)
                store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                inst.append(vect_op)           
                vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                inst.append(vect_op)
                vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                inst.append(vect_op)

        # gt computation
        # 1) W_xh_g * x_t
        for mm in range(fit_m):
            for nn in range(fit_n):
                # single tile for output matrix
                for kk in range(fit_k_in):
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[ weight_load])
                    inst.append(gemm_op)                    
                if outer_m == 1:
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                    inst.append(gemm_op) 
                store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])
                if layer.no_hidden:   
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op)

        if outer_m == 1:
            for nn in range(fit_n):
                for kk in range(fit_k_in):
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                    inst.append(gemm_op)
                if outer_m == 1:
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                    inst.append(gemm_op)
                store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])
                if layer.no_hidden:   
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)

        if outer_n == 1:
            for mm in range(fit_m):
                for kk in range(fit_k_in):
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                    inst.append(gemm_op)               
                if outer_m == 1:
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                    inst.append(gemm_op)
                store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])  
                if layer.no_hidden:  
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op)

        if outer_m == 1 and outer_n == 1:
            for kk in range(fit_k_in):
                weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                inst.append(weight_load)
                gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                inst.append(gemm_op)
            if outer_m == 1:
                weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                inst.append(weight_load)
                gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                inst.append(gemm_op)
            store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])
            if layer.no_hidden:
                vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                inst.append(vect_op)           
                vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                inst.append(vect_op)           


        # 2) W_hh_g * h_t-1 , add all
        if not layer.no_hidden:
            for mm in range(fit_m):
                for nn in range(fit_n):
                    # single tile for output matrix
                    for kk in range(fit_k_h):
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)                    
                    if outer_m == 1:
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load  )
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op) 
                    store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op) 
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op) 

            if outer_m == 1:
                for nn in range(fit_n):
                    for kk in range(fit_k_h):
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                    if outer_m == 1:
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                    store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)

            if outer_n == 1:
                for mm in range(fit_m):
                    for kk in range(fit_k_h):
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)               
                    if outer_m == 1:
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                    store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)

            if outer_m == 1 and outer_n == 1:
                for kk in range(fit_k_h):
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                    inst.append(gemm_op)
                if outer_m == 1:
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                    inst.append(gemm_op)
                store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                inst.append(vect_op)           
                vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                inst.append(vect_op)
                vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                inst.append(vect_op)

        # ct = ft * ct-1 + it * gt
        for mm in range(m):
            for kk in range(fit_k_in+1):
                vect_op = Inst(Op.VECTOR_OP, size=mmunit.width)
                inst.append(vect_op)
                vect_op = Inst(Op.VECTOR_OP, size=mmunit.width)
                inst.append(vect_op)
                vect_op = Inst(Op.VECTOR_OP, size=mmunit.width)
                inst.append(vect_op)

        # ht = ot * tanh(ct)
        for mm in range(m):
            for kk in range(fit_k_in+1):
                vect_op = Inst(Op.VECTOR_OP, size=mmunit.width)
                inst.append(vect_op)
                vect_op = Inst(Op.VECTOR_OP, size=mmunit.width)
                inst.append(vect_op)

    if layer.layer_type == Type.CONV:
        batch = layer.batch
        m = layer.im2col_m
        k = layer.im2col_k
        n = layer.im2col_n

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

        if not layer.previous_input:
            input_load = Inst(Op.LOAD_TILE, size=layer.in_dim[0]*layer.in_dim[1], buf=Buf.UBUF)
            inst.append(input_load)    

        #print(fit_m, fit_k, fit_n, left_m, left_k, left_n, outer_m, outer_n, mmunit.width, mmunit.height, mmunit.depth)
        #input()

        for b in range(batch):
            for mm in range(fit_m):
                for nn in range(fit_n):
                    # single tile for output matrix
                    for kk in range(fit_k):
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[ weight_load])
                        inst.append(gemm_op)                    
                    if outer_m == 1:
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load  )
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op) 
                    store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op) 
                    store_op = Inst(Op.STORE_TILE, size=mmunit.width*mmunit.depth, buf=Buf.UBUF, depend=[vect_op])
                    inst.append(store_op)

            if outer_m == 1:
                for nn in range(fit_n):
                    for kk in range(fit_k):
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                    if outer_m == 1:
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                    store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                    vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                    inst.append(vect_op)
                    store_op = Inst(Op.STORE_TILE, size=left_m*mmunit.depth, buf=Buf.UBUF, depend=[vect_op])
                    inst.append(store_op)

            if outer_n == 1:
                for mm in range(fit_m):
                    for kk in range(fit_k):
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)               
                    if outer_m == 1:
                        weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                        inst.append(weight_load)
                        gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                        inst.append(gemm_op)
                    store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                    vect_op = Inst(Op.VECTOR_OP, size=mmunit.width, depend=[gemm_op])
                    inst.append(vect_op)
                    store_op = Inst(Op.STORE_TILE, size=left_n*mmunit.width, buf=Buf.UBUF, depend=[vect_op])
                    inst.append(store_op)

            if outer_m == 1 and outer_n == 1:
                for kk in range(fit_k):
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                    inst.append(gemm_op)
                if outer_m == 1:
                    weight_load = Inst(Op.LOAD_TILE, size=mmunit.height*mmunit.depth, buf=Buf.WBUF)
                    inst.append(weight_load)
                    gemm_op = Inst(Op.GEMM_OP, M=mmunit.width, K=mmunit.height, N=mmunit.depth, depend=[weight_load])
                    inst.append(gemm_op)
                store_fake = Inst(Op.STORE_FAKE, buf=Buf.ACCQ, depend=[gemm_op])    
                vect_op = Inst(Op.VECTOR_OP, size=left_m, depend=[gemm_op])
                inst.append(vect_op)           
                store_op = Inst(Op.STORE_TILE, size=left_m*left_n, buf=Buf.UBUF, depend=[vect_op])
                inst.append(store_op)

    if layer.layer_type == Type.POOL:
        vect_op = Inst(Op.VECTOR_OP, size=layer.in_dim[0]*layer.in_dim[1]*layer.in_dim[2]*layer.stride)
        inst.append(vect_op) 

    return inst

class Inst:
    def __init__(self, inst_type, size=None, buf=None, M=None, K=None, N=None, depend=None):
        self.inst_type = inst_type
        self.done = False
        self.depend = []

        self.size = None
        self.buf = None
        self.M = None
        self.N = None
        self.K = None
        
        if depend != None:
            self.depend.extend(depend)
        
        if None in self.depend:
            print("NONE INPUT FOR DEPEND")
            # input()
            exit(0)

        if inst_type == Op.NOP:
            self.done = True
        elif inst_type == Op.LOAD_TILE:
            if size != None and buf != None:
                self.size = size
                self.buf = buf
            else:
                print(f"Missing Argument! -- LOAD_TILE / {self.size} / {self.buf}")
        elif inst_type == Op.GEMM_OP:
            if M != None and N != None and K != None:
                self.M = M
                self.N = N
                self.K = K
            else:
                print("Missing Argument! -- GEMM_OP / {self.M} / {self.N} / {self.K}")
        elif inst_type == Op.VECTOR_OP:
            if size != None:
                self.size = size
            else:
                print("Missing Argument! -- VECTOR_OP / {self.size}")
        elif inst_type == Op.STORE_TILE:
            if size != None and buf != None:
                self.size = size
                self.buf = buf
            else:
                print("Missing Argument! -- STORE_TILE / {self.size} / {self.buf}")
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
            print(f"STR BUG: LOAD_TILE --> {self.buf}")
            return "DEBUG"
        elif self.inst_type == Op.STORE_TILE:
            res = f"STORE_TILE\t{self.size}"
            if self.buf == Buf.UBUF:
                return res + "\tUBUF"
            elif self.buf == Buf.WBUF:
                return res + "\tWBUF"
            print(f"STR BUG: STORE_TILE --> {self.buf}")
            return "DEBUG"
        elif self.inst_type == Op.GEMM_OP:
            return f"GEMM_OP\t{self.M}\t{self.K}\t{self.N}"
        elif self.inst_type == Op.VECTOR_OP:
            return f"VECTOR_OP\t{self.size}"

        print(f"STR BUG: What type? --> {self.inst_type}")
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
        self.dispatch_first_time = dispatch_time
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
            #print(i)
            self.inst.extend(compile(i, self.mmunit))
        self.estimated = self.container.estimate(self.mmunit.height, self.mmunit.width, self.mmunit.depth)
        # print(f"ESTIMAGED: {self.nnid} -> {self.estimated}")
        #input()

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
            if i.inst_type.value < 6:
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
        
        res += f"  Originally Dispatched: {self.dispatch_first_time}\n"
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