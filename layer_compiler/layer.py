from layer_compiler.enum_def import Type

class Layer:
    def __init__(self, layer_type, batch=1, in_dim=None, out_dim=None, h_dim=None, window_dim=None, kernel_dim=None, kernel_num=None, \
             no_hidden=False, stride=1, padding=0, previous_input=False, gemm_m=None, gemm_k=None, gemm_n=None):
        self.layer_type = layer_type
        self.batch = batch
        self.previous_input = previous_input
        
        self.in_dim = None
        self.out_dim = None
        self.window_dim = None
        self.kernel_dim = None
        self.kernel_num = None
        self.h_dim = None
        self.no_hidden = None
        self.gemm_m = None
        self.gemm_k = None
        self.gemm_n = None
        self.stride = None
        self.padding = None
        self.im2col_m = None
        self.im2col_k = None
        self.im2col_n = None
                
        if layer_type == Type.FC :
            if all((in_dim, out_dim)):
                self.in_dim = in_dim
                self.out_dim = out_dim
            else:
                print("Missing Argument! -- FC")
        elif layer_type == Type.GEMM:
            if all((gemm_m, gemm_k, gemm_n)):
                self.gemm_m = gemm_m
                self.gemm_k = gemm_k
                self.gemm_n = gemm_n
            else:
                print("Missing Argument! -- GEMM")
        elif layer_type == Type.LSTM:
            if all((in_dim, h_dim)):
                self.in_dim = in_dim
                self.h_dim = h_dim
                self.no_hidden = no_hidden
            else:
                print("Missing Argument! -- LSTM")
        elif layer_type == Type.CONV:
            if all((in_dim, kernel_dim, kernel_num, stride)):
                self.in_dim = in_dim
                self.kernel_dim = kernel_dim
                self.kernel_num = kernel_num
                self.stride = stride
                self.padding = padding
                
                channel = self.in_dim[2]
                out = [0, 0, 0]
                out[0] = int((self.in_dim[0] - self.kernel_dim[0] + self.padding*2 + 1) / stride) + 1
                out[1] = int((self.in_dim[1] - self.kernel_dim[1] + self.padding*2 + 1) / stride) + 1
                out[2] = self.kernel_num

                self.im2col_m = out[2]
                self.im2col_k = (self.kernel_dim[0] * self.kernel_dim[1] * channel)
                self.im2col_n = out[0] * out[1] * self.batch
            else:
                print("Missing Argument! -- CONV")
        elif layer_type == Type.POOL:
            # assume square input and window
            if all((in_dim, window_dim, stride)):
                self.in_dim = in_dim
                self.window_dim = window_dim
                self.stride = stride
            else:
                print("Missing Argument! -- POOL")
        elif layer_type == Type.DEPTH:
            if all((in_dim, kernel_dim, stride)):
                self.in_dim = in_dim
                self.kernel_dim = kernel_dim
                self.kernel_num = in_dim[2]
                self.stride = stride
                self.padding = padding
                
                channel = self.in_dim[2]
                out = [0, 0, 0]
                out[0] = int((self.in_dim[0] - self.kernel_dim[0] + self.padding*2 + 1) / stride) + 1
                out[1] = int((self.in_dim[1] - self.kernel_dim[1] + self.padding*2 + 1) / stride) + 1
                out[2] = self.in_dim[2]

                self.im2col_m = self.in_dim[2]
                self.im2col_k = (self.kernel_dim[0] * self.kernel_dim[1] * channel)
                self.im2col_n = out[0] * out[1] * self.batch
            else:
                print("Missing Argument! -- DEPTH")
            

    def estimate(self, height, width, depth, bw=358*1024*1024*1024/4):
        if self.layer_type == Type.FC :
            return self.estimate_gemm(self.batch, self.in_dim, self.out_dim, height, width, depth, bw)
        
        elif self.layer_type == Type.GEMM:
            return self.batch * self.estimate_gemm(self.gemm_m, self.gemm_k, self.gemm_n, height, width, depth, bw)

        elif self.layer_type == Type.LSTM:
            i_h = self.estimate_gemm(self.batch, self.in_dim, self.h_dim, height, width, depth, bw)
            h_h = self.estimate_gemm(self.batch, self.in_dim, self.h_dim, height, width, depth, bw)
            return 4 * (i_h + h_h)
        elif self.layer_type == Type.CONV:
            return self.estimate_gemm(self.im2col_m, self.im2col_k, self.im2col_n, height, width, depth, bw)
        elif self.layer_type == Type.POOL:
            return 0
        elif self.layer_type == Type.DEPTH:
            return self.estimate_gemm(self.im2col_m, self.im2col_k, self.im2col_n, height, width, depth, bw)

    def estimate_gemm(self, m, k, n, height, width, depth, bw):
        c1 = height + 2 * width + depth
        m1 = int((height * width + height * depth) / bw)
        inner = max(c1, m1)

        c2 = n - int(n/depth) * depth + height + 2 * width
        m2 = int((height * width + height * (n - int(n/depth) * depth)) / bw)
        outer = max(c2, m2)

        case = 0
        if n % depth != 0:
            case = 1

        fit_m = int(m/width)
        fit_k = int(k/height)
        fit_n = int(n/depth)

        est = fit_m * fit_k * fit_n * inner + fit_m * fit_k * case * outer
        return est
    
    def __str__(self):
        if self.layer_type == Type.FC :
            return f"FC: In({self.in_dim}), Out({self.out_dim})"
        elif self.layer_type == Type.GEMM:
            return f"GEMM: M({self.gemm_m}), K({self.gemm_k}), N({self.gemm_n})"
        elif self.layer_type == Type.LSTM:
            return f"LSTM: In({self.in_dim}), Hidden({self.h_dim})"
        elif self.layer_type == Type.POOL:
            return f"POOL: In({self.in_dim}), Window({self.window_dim})"
        elif self.layer_type == Type.CONV:
            return f"CONV: In({self.in_dim}), Kernel({self.kernel_dim}), Kernel_Num({self.kernel_num})Stride({self.stride}), Padding({self.padding}))"
        elif self.layer_type == Type.DEPTH:
            return f"DEPTH: In({self.in_dim}), Kernel({self.kernel_dim}), Stride({self.stride}), Padding({self.padding}))"
        
class Container:
    def __init__(self, *args):
        self.container = []
        self.container.extend(args)
        self.estimated = 0
        self.estimate_computed = False
        self.isolated = 0
        self.net_name = "DEFAULT"
        
    def push_layer(self, layer):
        self.container.append(layer)

    def batch_setup(self, size):
        for i in self.container:
            i.batch = size

    def estimate(self, height, width, depth):
        if self.estimate_computed:
            return max(self.estimated, 1)
        for i in self.container:
            #print("EST in CONT: ", i)
            self.estimated += i.estimate(height, width, depth)

        self.estimate_computed = True

        return max(self.estimated, 1)

    def __str__(self):
        ret = "{\n"
        for i in self.container:
            ret += "\t"
            ret += f"{i}\n"
        ret += "}"

        return ret