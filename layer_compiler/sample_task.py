from layer_compiler.enum_def import Type, Op, Buf, Sched, Mecha
from layer_compiler.layer import Layer, Container
from layer_compiler.compiler import NN

container_mlp_sample = Container()

container_cnn_alex = Container()
container_cnn_google = Container()
container_cnn_vgg = Container()
container_cnn_mobile = Container()

container_rnn_asr = Container()
container_rnn_mt = Container()
container_rnn_sa = Container()

def all_init(batch):
    four_mlp_init(batch)
    rnn_asr_init(batch, 40)
    cnn_alex_init(batch)


# 4-layer Sample 
def four_mlp_init(batch):
    layer1 = Layer(Type.FC, batch=batch, in_dim=100, out_dim=400)
    layer2 = Layer(Type.FC, batch=batch, in_dim=400, out_dim=400, previous_input=True)
    layer3 = Layer(Type.FC, batch=batch, in_dim=400, out_dim=400, previous_input=True)
    layer4 = Layer(Type.FC, batch=batch, in_dim=400, out_dim=10, previous_input=True)
    container_mlp_sample.push_layer(layer1)
    container_mlp_sample.push_layer(layer2)
    container_mlp_sample.push_layer(layer3)
    container_mlp_sample.push_layer(layer4)

# Alexnet
def cnn_alex_init(batch):
    conv1 = Layer(Type.CONV, batch=batch, in_dim=(224, 224, 3), kernel_dim=(11, 11), kernel_num=96, stride=4, padding=0)
    pool1 = Layer(Type.POOL, batch=batch, in_dim=(55, 55, 96), window_dim=(3, 3), stride = 1)
    conv2 = Layer(Type.CONV, batch=batch, in_dim=(27, 27, 96), kernel_dim=(5, 5), kernel_num=256, stride=1, padding=2)
    pool2 = Layer(Type.POOL, batch=batch, in_dim=(27, 27, 256), window_dim=(3, 3), stride=2)
    conv3 = Layer(Type.CONV, batch=batch, in_dim=(13, 13, 256), kernel_dim=(3, 3), kernel_num=384, stride=1, padding=1)
    conv4 = Layer(Type.CONV, batch=batch, in_dim=(13, 13, 384), kernel_dim=(3, 3), kernel_num=384, stride=1, padding=1)
    conv5 = Layer(Type.CONV, batch=batch, in_dim=(13, 13, 384), kernel_dim=(3, 3), kernel_num=256, stride=1, padding=1)
    pool5 = Layer(Type.POOL, batch=batch, in_dim=(13, 13, 256), window_dim=(3, 3), stride=2)
    fc6 = Layer(Type.FC, batch=batch, in_dim=9216, out_dim=4096)
    fc7 = Layer(Type.FC, batch=batch, in_dim=4096, out_dim=1000)

    container_cnn_alex.push_layer(conv1)
    container_cnn_alex.push_layer(pool1)
    container_cnn_alex.push_layer(conv2)
    container_cnn_alex.push_layer(pool2)
    container_cnn_alex.push_layer(conv3)
    container_cnn_alex.push_layer(conv4)
    container_cnn_alex.push_layer(conv5)
    container_cnn_alex.push_layer(pool5)
    container_cnn_alex.push_layer(fc6)
    container_cnn_alex.push_layer(fc7)

# ASR: Listen, Attend and Spell
# https://arxiv.org/pdf/1508.01211.pdf
# config: https://github.com/kaituoxu/Listen-Attend-Spell/blob/master/egs/aishell/run.sh#L21
'''
Model Architecture
[Encoder]
1) pyramidal Bi-LSTM with 3 layer (ith layer's input is it)
[Decoder]
1) 2 layer Bi-LSTM
2) Attention
3) 2-layer MLP
'''
def rnn_asr_init(batch, length):
    # Assumption: output = int(input/4)
    # Embedding layer is omitted

    # listener (encoder)
    N = batch
    Ti = length
    To = int(length/4)
    H = 256 # hidden, for single direction, i.e. 512 total
    D = 240 # input
    D_EMBED = 512
    D_HIDDEN = 512

    # encoder
    # first layer(256, bidirectional lstm)
    for i in range(Ti):
        if i == 0:
            layer_lstm1_bi1 = Layer(Type.LSTM, batch=N, in_dim=D, h_dim=H, no_hidden=True, previous_input=False)
            container_rnn_asr.push_layer(layer_lstm1_bi1)
            layer_lstm1_bi2 = Layer(Type.LSTM, batch=N, in_dim=D, h_dim=H, no_hidden=True, previous_input=False)
            container_rnn_asr.push_layer(layer_lstm1_bi2)
        else:
            layer_lstm1_bi1 = Layer(Type.LSTM, batch=N, in_dim=D, h_dim=H, no_hidden=False, previous_input=False)
            container_rnn_asr.push_layer(layer_lstm1_bi1)
            layer_lstm1_bi2 = Layer(Type.LSTM, batch=N, in_dim=D, h_dim=H, no_hidden=False, previous_input=False)
            container_rnn_asr.push_layer(layer_lstm1_bi2)
    
    # second layer
    for i in range(int(Ti/2)):
        if i == 0:
            layer_lstm1_bi1 = Layer(Type.LSTM, batch=N, in_dim=H*2, h_dim=H*2, no_hidden=True, previous_input=True)
            container_rnn_asr.push_layer(layer_lstm1_bi1)
            layer_lstm1_bi2 = Layer(Type.LSTM, batch=N, in_dim=H*2, h_dim=H*2, no_hidden=True, previous_input=True)
            container_rnn_asr.push_layer(layer_lstm1_bi2)
        else:
            layer_lstm1_bi1 = Layer(Type.LSTM, batch=N, in_dim=H*2, h_dim=H*2, no_hidden=False, previous_input=True)
            container_rnn_asr.push_layer(layer_lstm1_bi1)
            layer_lstm1_bi2 = Layer(Type.LSTM, batch=N, in_dim=H*2, h_dim=H*2, no_hidden=False, previous_input=True)
            container_rnn_asr.push_layer(layer_lstm1_bi2)
    
    # third layer
    for i in range(int(Ti/4)):
        if i == 0:
            layer_lstm1_bi1 = Layer(Type.LSTM, batch=N, in_dim=H*4, h_dim=H*4, no_hidden=True, previous_input=True)
            container_rnn_asr.push_layer(layer_lstm1_bi1)
            layer_lstm1_bi2 = Layer(Type.LSTM, batch=N, in_dim=H*4, h_dim=H*4, no_hidden=True, previous_input=True)
            container_rnn_asr.push_layer(layer_lstm1_bi2)
        else:
            layer_lstm1_bi1 = Layer(Type.LSTM, batch=N, in_dim=H*4, h_dim=H*4, no_hidden=False, previous_input=True)
            container_rnn_asr.push_layer(layer_lstm1_bi1)
            layer_lstm1_bi2 = Layer(Type.LSTM, batch=N, in_dim=H*4, h_dim=H*4, no_hidden=False, previous_input=True)
            container_rnn_asr.push_layer(layer_lstm1_bi2)
    
    # decoder
    # first layer
    for i in range(To):
        if i == 0:
            layer_lstm1_bi1 = Layer(Type.LSTM, batch=N, in_dim=D_EMBED, h_dim=D_HIDDEN, no_hidden=True, previous_input=True)
            container_rnn_asr.push_layer(layer_lstm1_bi1)
            layer_lstm1_bi2 = Layer(Type.LSTM, batch=N, in_dim=D_EMBED, h_dim=D_HIDDEN, no_hidden=True, previous_input=True)
            container_rnn_asr.push_layer(layer_lstm1_bi2)
        else:
            layer_lstm1_bi1 = Layer(Type.LSTM, batch=N, in_dim=D_EMBED, h_dim=D_HIDDEN, no_hidden=False, previous_input=True)
            container_rnn_asr.push_layer(layer_lstm1_bi1)
            layer_lstm1_bi2 = Layer(Type.LSTM, batch=N, in_dim=D_EMBED, h_dim=D_HIDDEN, no_hidden=False, previous_input=True)
            container_rnn_asr.push_layer(layer_lstm1_bi2)

    # second layer
    for i in range(To):
        if i == 0:
            layer_lstm1_bi1 = Layer(Type.LSTM, batch=N, in_dim=D_EMBED, h_dim=D_HIDDEN, no_hidden=True, previous_input=True)
            container_rnn_asr.push_layer(layer_lstm1_bi1)
            layer_lstm1_bi2 = Layer(Type.LSTM, batch=N, in_dim=D_EMBED, h_dim=D_HIDDEN, no_hidden=True, previous_input=True)
            container_rnn_asr.push_layer(layer_lstm1_bi2)
        else:
            layer_lstm1_bi1 = Layer(Type.LSTM, batch=N, in_dim=D_EMBED, h_dim=D_HIDDEN, no_hidden=False, previous_input=True)
            container_rnn_asr.push_layer(layer_lstm1_bi1)
            layer_lstm1_bi2 = Layer(Type.LSTM, batch=N, in_dim=D_EMBED, h_dim=D_HIDDEN, no_hidden=False, previous_input=True)
            container_rnn_asr.push_layer(layer_lstm1_bi2)

    # attention
    attention_score = Layer(Type.GEMM, batch=N, gemm_m=To, gemm_k=D_HIDDEN, gemm_n=Ti, previous_input=True)
    container_rnn_asr.push_layer(attention_score)
    attention_output = Layer(Type.GEMM, batch=N, gemm_m=To, gemm_k=Ti, gemm_n=D_HIDDEN, previous_input=True)
    container_rnn_asr.push_layer(attention_output)

    # mlp
    layer1 = Layer(Type.FC, batch=N, in_dim=D_HIDDEN*2, out_dim=D_HIDDEN)
    layer2 = Layer(Type.FC, batch=N, in_dim=D_HIDDEN, out_dim=D)
    container_rnn_asr.push_layer(layer1)   
    container_rnn_asr.push_layer(layer2)

