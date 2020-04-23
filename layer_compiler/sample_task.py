from layer_compiler.enum_def import Type, Op, Buf, Sched, Mecha
from layer_compiler.layer import Layer, Container
from layer_compiler.compiler import NN


container_rnn_asr = Container()

def rnn_asr_init(length):
    # Assumption: output = int(input/4)
    out_length = int(length/4)