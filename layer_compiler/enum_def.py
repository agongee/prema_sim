from enum import Enum

class Type(Enum):
    FC = 1
    ACTV = 2
    POOL = 3
    CONV = 4
    DEPTH = 5
    SEPAR = 6
    GEMM = 7
    LSTM = 8

class Buf(Enum):
    UBUF = 1
    WBUF = 2
    ACCQ = 3

class Op(Enum):
    LOAD_TILE = 1
    GEMM_OP = 2
    CONV_OP = 3
    VECTOR_OP = 4
    STORE_TILE = 5
    STORE_FAKE = 6
    NOP = 7
    
class Sched(Enum):
    FCFS = 1
    RRB = 2
    HPF = 3
    TOKEN = 4
    SJF = 5
    PREMA = 6

class Mecha(Enum):
    STATIC = 1
    DYNAMIC = 2