from layer_compiler.enum_def import Type

class Layer:
    def __init__(self, layer_type, batch=1, in_dim=None, out_dim=None, h_dim=None, window_dim=None):
        self.layer_type = layer_type
        self.batch = batch
        
        self.in_dim = None
        self.out_dim = None
        self.window_dim = None
        self.h_dim = None

        if layer_type == Type.FC :
            if all((in_dim, out_dim)):
                self.in_dim = in_dim
                self.out_dim = out_dim
            else:
                print("Missing Argument!")
        elif layer_type == Type.RECR:
            if all((in_dim, h_dim, out_dim)):
                self.in_dim = in_dim
                self.h_dim = h_dim
                self.out_dim = out_dim
            else:
                print("Missing Argument!")
        elif layer_type == Type.LSTM:
            if all((in_dim, h_dim)):
                self.in_dim = in_dim
                self.h_dim = h_dim
            else:
                print("Missing Argument!")
        '''
        elif layer_type == Type.POOL:
            if all((in_dim, window_dim)):
                self.in_dim = in_dim
                self.window_dim = window_dim
            else:
                print("Missing Argument!")
        '''
        '''
        elif layer_type == Type.ACTV:
            if all((in_dim)):
                self.in_dim = in_dim
            else:
                print("Missing Argument!")
        '''

        '''
        elif num == Type.CONV:
            self.in_row = in_row
            if self.in_col == None:
                self.in_col = in_row
            else:
                self.in_col = in_col
            self.in_ch = in_ch
            self.k_row = k_row
            if self.k_col == None:
                self.k_col = k_row
            else:
                self.k_col = k_col
            self.k_num = k_num
        elif num == Type.DEPTH:
            pass
        elif num == Type.SEPAR:
            pass
        '''

    def estimate(self, height, width, depth, bw=358*1024*1024*1024/4):
        if self.layer_type == Type.FC :
            m = self.batch
            k = self.in_dim
            n = self.out_dim

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
            print("EST in LAYER: ", est)
            return est
        
        elif self.layer_type == Type.RECR:
            pass
        elif self.layer_type == Type.LSTM:
            pass
        '''
        elif self.layer_type == Type.POOL:
            return f"POOL: In({self.in_dim}), Window({self.window_dim})"
        '''
        
    
    def __str__(self):
        if self.layer_type == Type.FC :
            return f"FC: In({self.in_dim}), Out({self.out_dim})"
        elif self.layer_type == Type.RECR:
            return f"RECR: In({self.in_dim}), Hidden({self.h_dim}), Out({self.out_dim})"
        elif self.layer_type == Type.LSTM:
            return f"LSTM: In({self.in_dim}), Hidden({self.h_dim})"
        '''
        elif self.layer_type == Type.ACTV:
            return f"ACTV: In({self.in_dim})"
        elif self.layer_type == Type.POOL:
            return f"POOL: In({self.in_dim}), Window({self.window_dim})"
        '''

    
class Container:
    def __init__(self, *args):
        self.container = []
        self.container.extend(args)
        self.estimated = 0
        
    def push_layer(self, layer):
        self.container.append(layer)

    def batch_setup(self, size):
        for i in self.container:
            i.batch = size

    def estimate(self, height, width, depth):
        for i in self.container:
            print("EST in CONT: ", i)
            self.estimated += i.estimate(height, width, depth)

        return self.estimated

    def __str__(self):
        ret = "{\n"
        for i in self.container:
            ret += "\t"
            ret += f"{i}\n"
        ret += "}"

        return ret