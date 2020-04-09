from enum import Enum

class Type(Enum):
    FC = 1
    ACTV = 2
    POOL = 3
    CONV = 4
    DEPTH = 5
    SEPAR = 6
    RECR = 7
    LSTM = 8


class Layer:
    def __init__(self, layer_type, in_dim=None, out_dim=None, h_dim=None, w_dim=None, act_type=None):
        self.layer_type = layer_type
        if layer_type == Type.FC :
            if all((in_dim, out_dim)):
                self.in_dim = in_dim
                self.out_dim = out_dim
            else:
                print("Missing Argument!")
        elif layer_type == Type.ACTV:
            if all((in_dim, act_type)):
                self.in_dim = in_dim
                self.act_type = act_type
            else:
                print("Missing Argument!")
        elif layer_type == Type.POOL:
            if all((in_dim, w_dim)):
                self.in_dim = in_dim
                self.w_dim = w_dim
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
    
    def __str__(self):
        if self.layer_type == Type.FC :
            return f"FC: In({self.in_dim}), Out({self.out_dim})"
        elif self.layer_type == Type.ACTV:
            return f"ACTV: In({self.in_dim}), Type({self.act_type})"
        elif self.layer_type == Type.POOL:
            return f"POOL: In({self.in_dim}), Window({self.w_dim})"
        elif self.layer_type == Type.RECR:
            return f"RECR: In({self.in_dim}), Hidden({self.h_dim}), Out({self.out_dim})"
        elif self.layer_type == Type.LSTM:
            return f"LSTM: In({self.in_dim}), Hidden({self.h_dim})"
    
class Container:
    def __init__(self, *args):
        self.container = []
        self.container.extend(args)
        
    def push_layer(self, layer):
        self.container.append(layer)

    def __str__(self):
        ret = "{\n"
        for i in self.container:
            ret += "\t"
            ret += f"{i}\n"
        ret += "}"

        return ret
    

        