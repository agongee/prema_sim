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
    cnn_vgg_init(batch)
    cnn_google_init(batch)


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
    pool1 = Layer(Type.POOL, batch=batch, in_dim=(55, 55, 96), window_dim=(3, 3), stride = 1, previous_input=True)
    conv2 = Layer(Type.CONV, batch=batch, in_dim=(27, 27, 96), kernel_dim=(5, 5), kernel_num=256, stride=1, padding=2, previous_input=True)
    pool2 = Layer(Type.POOL, batch=batch, in_dim=(27, 27, 256), window_dim=(3, 3), stride=2, previous_input=True)
    conv3 = Layer(Type.CONV, batch=batch, in_dim=(13, 13, 256), kernel_dim=(3, 3), kernel_num=384, stride=1, padding=1, previous_input=True)
    conv4 = Layer(Type.CONV, batch=batch, in_dim=(13, 13, 384), kernel_dim=(3, 3), kernel_num=384, stride=1, padding=1, previous_input=True)
    conv5 = Layer(Type.CONV, batch=batch, in_dim=(13, 13, 384), kernel_dim=(3, 3), kernel_num=256, stride=1, padding=1, previous_input=True)
    pool5 = Layer(Type.POOL, batch=batch, in_dim=(13, 13, 256), window_dim=(3, 3), stride=2, previous_input=True)
    fc6 = Layer(Type.FC, batch=batch, in_dim=9216, out_dim=4096, previous_input=True)
    fc7 = Layer(Type.FC, batch=batch, in_dim=4096, out_dim=1000, previous_input=True)

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

# VGG16
# [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], avgpool
# N -> next channel num, kernel 3x3, stride=1, pad=1
def cnn_vgg_init(batch):
    conv1 = Layer(Type.CONV, batch=batch, in_dim=(224, 224, 3), kernel_dim=(3, 3), kernel_num=64, stride=1, padding=1)
    conv2 = Layer(Type.CONV, batch=batch, in_dim=(224, 224, 64), kernel_dim=(3, 3), kernel_num=64, stride=1, padding=1, previous_input=True)
    pool2 = Layer(Type.POOL, batch=batch, in_dim=(224, 224, 64), window_dim=(2, 2), stride=2, previous_input=True)

    container_cnn_vgg.push_layer(conv1)
    container_cnn_vgg.push_layer(conv2)
    container_cnn_vgg.push_layer(pool2)

    conv3 = Layer(Type.CONV, batch=batch, in_dim=(112, 112, 64), kernel_dim=(3, 3), kernel_num=128, stride=1, padding=1, previous_input=True)
    conv4 = Layer(Type.CONV, batch=batch, in_dim=(112, 112, 128), kernel_dim=(3, 3), kernel_num=128, stride=1, padding=1, previous_input=True)
    pool4 = Layer(Type.POOL, batch=batch, in_dim=(112, 112, 128), window_dim=(2, 2), stride=2, previous_input=True)

    container_cnn_vgg.push_layer(conv3)
    container_cnn_vgg.push_layer(conv4)
    container_cnn_vgg.push_layer(pool4)

    conv5 = Layer(Type.CONV, batch=batch, in_dim=(56, 56, 128), kernel_dim=(3, 3), kernel_num=256, stride=1, padding=1, previous_input=True)
    conv6 = Layer(Type.CONV, batch=batch, in_dim=(56, 56, 256), kernel_dim=(3, 3), kernel_num=256, stride=1, padding=1, previous_input=True)
    conv7 = Layer(Type.CONV, batch=batch, in_dim=(56, 56, 256), kernel_dim=(3, 3), kernel_num=256, stride=1, padding=1, previous_input=True)
    pool7 = Layer(Type.POOL, batch=batch, in_dim=(56, 56, 256), window_dim=(2, 2), stride=2, previous_input=True)

    container_cnn_vgg.push_layer(conv5)
    container_cnn_vgg.push_layer(conv6)
    container_cnn_vgg.push_layer(conv7)
    container_cnn_vgg.push_layer(pool7)

    conv8 = Layer(Type.CONV, batch=batch, in_dim=(28, 28, 256), kernel_dim=(3, 3), kernel_num=512, stride=1, padding=1, previous_input=True)
    conv9 = Layer(Type.CONV, batch=batch, in_dim=(28, 28, 512), kernel_dim=(3, 3), kernel_num=512, stride=1, padding=1, previous_input=True)
    conv10 = Layer(Type.CONV, batch=batch, in_dim=(28, 28, 512), kernel_dim=(3, 3), kernel_num=512, stride=1, padding=1, previous_input=True)
    pool10 = Layer(Type.POOL, batch=batch, in_dim=(28, 28, 512), window_dim=(2, 2), stride=2, previous_input=True)

    container_cnn_vgg.push_layer(conv8)
    container_cnn_vgg.push_layer(conv9)
    container_cnn_vgg.push_layer(conv10)
    container_cnn_vgg.push_layer(pool10)

    conv11 = Layer(Type.CONV, batch=batch, in_dim=(14, 14, 512), kernel_dim=(3, 3), kernel_num=512, stride=1, padding=1, previous_input=True)
    conv12 = Layer(Type.CONV, batch=batch, in_dim=(14, 14, 512), kernel_dim=(3, 3), kernel_num=512, stride=1, padding=1, previous_input=True)
    conv13 = Layer(Type.CONV, batch=batch, in_dim=(14, 14, 512), kernel_dim=(3, 3), kernel_num=512, stride=1, padding=1, previous_input=True)
    pool13 = Layer(Type.POOL, batch=batch, in_dim=(14, 14, 512), window_dim=(2, 2), stride=2, previous_input=True)
    pool14 = Layer(Type.POOL, batch=batch, in_dim=(7, 7, 512), window_dim=(2, 2), stride=2, previous_input=True)

    container_cnn_vgg.push_layer(conv11)
    container_cnn_vgg.push_layer(conv12)
    container_cnn_vgg.push_layer(conv13)
    container_cnn_vgg.push_layer(pool13)
    container_cnn_vgg.push_layer(pool14)

    fc1 = Layer(Type.FC, batch=batch, in_dim=4096, out_dim=4096, previous_input=True)
    fc2 = Layer(Type.FC, batch=batch, in_dim=4096, out_dim=4096, previous_input=True)
    fc3 = Layer(Type.FC, batch=batch, in_dim=4096, out_dim=1000, previous_input=True)

    container_cnn_vgg.push_layer(fc1)
    container_cnn_vgg.push_layer(fc2)
    container_cnn_vgg.push_layer(fc3)

# GoogLeNet
# https://arxiv.org/pdf/1409.4842.pdf
def cnn_google_init(batch):
    conv1 = Layer(Type.CONV, batch=batch, in_dim=(224, 224, 3), kernel_dim=(7, 7), kernel_num=64, stride=2, padding=3, previous_input=False)
    pool1 = Layer(Type.POOL, batch=batch, in_dim=(112, 112, 64), window_dim=(3, 3), stride=2)
    conv2 = Layer(Type.CONV, batch=batch, in_dim=(56, 56, 64), kernel_dim=(3, 3), kernel_num=192, stride=1, padding=1, previous_input=True)
    pool2 = Layer(Type.POOL, batch=batch, in_dim=(56, 56, 192), window_dim=(3, 3), stride=2)

    container_cnn_google.push_layer(conv1)
    container_cnn_google.push_layer(pool1)
    container_cnn_google.push_layer(conv2)
    container_cnn_google.push_layer(pool2)

    next_in = push_inception(batch, (28, 28, 192), (64, 96, 128, 16, 32, 32))
    next_in = push_inception(batch, next_in, (128, 128, 192, 32, 96, 64))
    
    pool3 = Layer(Type.POOL, batch=batch, in_dim=next_in, window_dim=(3, 3), stride=2)
    container_cnn_google.push_layer(pool3)

    next_in = push_inception(batch, (14, 14, 480), (192, 96, 208, 16, 48, 64))
    next_in = push_inception(batch, next_in, (160, 112, 224, 24, 64, 64))
    next_in = push_inception(batch, next_in, (128, 128, 256, 24, 64, 64))
    next_in = push_inception(batch, next_in, (112, 144, 288, 32, 64, 64))
    next_in = push_inception(batch, next_in, (256, 160, 320, 32, 128, 128))

    pool4 = Layer(Type.POOL, batch=batch, in_dim=next_in, window_dim=(3, 3), stride=2)
    container_cnn_google.push_layer(pool4)

    next_in = push_inception(batch, (7, 7, 832), (256, 160, 320, 32, 128, 128))
    next_in = push_inception(batch, next_in, (384, 192, 384, 48, 128, 128))

    pool5 = Layer(Type.POOL, batch=batch, in_dim=next_in, window_dim=(7, 7), stride=1)
    container_cnn_google.push_layer(pool5)

    fc = Layer(Type.FC, batch=batch, in_dim=1024, out_dim=1000, previous_input=True)
    container_cnn_vgg.push_layer(fc)


def push_inception(batch: int, in_dim: tuple, channel_dim: tuple):
    # 1x1
    conv1_input = Layer(Type.CONV, batch=batch, in_dim=in_dim, kernel_dim=(1, 1), kernel_num=channel_dim[0], stride=1, padding=0, previous_input=True)
    container_cnn_google.push_layer(conv1_input)

    # 1x1 -> 3x3
    conv1_no_in = Layer(Type.CONV, batch=batch, in_dim=in_dim, kernel_dim=(1, 1), kernel_num=channel_dim[1], stride=1, padding=0, previous_input=True)
    conv3 = Layer(Type.CONV, batch=batch, in_dim=(in_dim[0], in_dim[1], channel_dim[1]), kernel_dim=(3, 3), kernel_num=channel_dim[2], stride=1, padding=1, previous_input=True)
    container_cnn_google.push_layer(conv1_no_in)
    container_cnn_google.push_layer(conv3)

    # 1x1 -> 5x5
    conv1_no_in = Layer(Type.CONV, batch=batch, in_dim=in_dim, kernel_dim=(1, 1), kernel_num=channel_dim[3], stride=1, padding=0, previous_input=True)
    conv5 = Layer(Type.CONV, batch=batch, in_dim=(in_dim[0], in_dim[1], channel_dim[3]), kernel_dim=(5, 5), kernel_num=channel_dim[4], stride=1, padding=2, previous_input=True)
    container_cnn_google.push_layer(conv1_no_in)
    container_cnn_google.push_layer(conv5)

    # pool -> 1x1
    pool = Layer(Type.POOL, batch=batch, in_dim=in_dim, window_dim=(3, 3), stride=1, previous_input=True)
    conv1_no_in = Layer(Type.CONV, batch=batch, in_dim=in_dim, kernel_dim=(1, 1), kernel_num=channel_dim[5], stride=1, padding=0, previous_input=True)
    container_cnn_google.push_layer(pool)
    container_cnn_google.push_layer(conv1_no_in)

    return (in_dim[0], in_dim[1], channel_dim[0]+channel_dim[2]+channel_dim[4]+channel_dim[5])

# MobileNet
def cnn_mobile_init(batch):
    conv1 = Layer(Type.CONV, batch=batch, in_dim=(224, 224, 3), kernel_dim=(3, 3), kernel_num=32, stride=2, padding=1, previous_input=False)
    container_cnn_mobile.push_layer(conv1)

    convdw2 = Layer(Type.DEPTH, batch=batch, in_dim=(112, 112, 32), kernel_dim=(3, 3), stride=1, padding=1, previous_input=True)
    conv2 = Layer(Type.CONV, batch=batch, in_dim=(112, 112, 32), kernel_dim=(1, 1), kernel_num=64, stride=1, padding=1, previous_input=True)
    container_cnn_mobile.push_layer(convdw2)
    container_cnn_mobile.push_layer(conv2)

    convdw3 = Layer(Type.DEPTH, batch=batch, in_dim=(112, 112, 64), kernel_dim=(3, 3), stride=2, padding=1, previous_input=True)
    conv3 = Layer(Type.CONV, batch=batch, in_dim=(56, 56, 64), kernel_dim=(1, 1), kernel_num=128, stride=1, padding=1, previous_input=True)
    container_cnn_mobile.push_layer(convdw3)
    container_cnn_mobile.push_layer(conv3)

    convdw4 = Layer(Type.DEPTH, batch=batch, in_dim=(56, 56, 128), kernel_dim=(3, 3), stride=1, padding=1, previous_input=True)
    conv4 = Layer(Type.CONV, batch=batch, in_dim=(56, 56, 128), kernel_dim=(1, 1), kernel_num=128, stride=1, padding=1, previous_input=True)
    container_cnn_mobile.push_layer(convdw4)
    container_cnn_mobile.push_layer(conv4)

    convdw5= Layer(Type.DEPTH, batch=batch, in_dim=(56, 56, 128), kernel_dim=(3, 3), stride=2, padding=1, previous_input=True)
    conv5 = Layer(Type.CONV, batch=batch, in_dim=(28, 28, 128), kernel_dim=(1, 1), kernel_num=256, stride=1, padding=1, previous_input=True)
    container_cnn_mobile.push_layer(convdw5)
    container_cnn_mobile.push_layer(conv5)

    convdw6 = Layer(Type.DEPTH, batch=batch, in_dim=(28, 28, 256), kernel_dim=(3, 3), stride=2, padding=1, previous_input=True)
    conv6 = Layer(Type.CONV, batch=batch, in_dim=(14, 14, 256), kernel_dim=(1, 1), kernel_num=512, stride=1, padding=1, previous_input=True)
    container_cnn_mobile.push_layer(convdw6)
    container_cnn_mobile.push_layer(conv6)

    convdw7 = Layer(Type.DEPTH, batch=batch, in_dim=(14, 14, 512), kernel_dim=(3, 3), stride=1, padding=1, previous_input=True)
    conv7 = Layer(Type.CONV, batch=batch, in_dim=(14, 14, 512), kernel_dim=(1, 1), kernel_num=512, stride=1, padding=1, previous_input=True)
    container_cnn_mobile.push_layer(convdw7)
    container_cnn_mobile.push_layer(conv7)

    convdw8 = Layer(Type.DEPTH, batch=batch, in_dim=(14, 14, 512), kernel_dim=(3, 3), stride=1, padding=1, previous_input=True)
    conv8 = Layer(Type.CONV, batch=batch, in_dim=(14, 14, 512), kernel_dim=(1, 1), kernel_num=512, stride=1, padding=1, previous_input=True)
    container_cnn_mobile.push_layer(convdw8)
    container_cnn_mobile.push_layer(conv8)

    convdw9 = Layer(Type.DEPTH, batch=batch, in_dim=(14, 14, 512), kernel_dim=(3, 3), stride=1, padding=1, previous_input=True)
    conv9 = Layer(Type.CONV, batch=batch, in_dim=(14, 14, 512), kernel_dim=(1, 1), kernel_num=512, stride=1, padding=1, previous_input=True)
    container_cnn_mobile.push_layer(convdw9)
    container_cnn_mobile.push_layer(conv9)

    convdw10 = Layer(Type.DEPTH, batch=batch, in_dim=(14, 14, 512), kernel_dim=(3, 3), stride=1, padding=1, previous_input=True)
    conv10 = Layer(Type.CONV, batch=batch, in_dim=(14, 14, 512), kernel_dim=(1, 1), kernel_num=512, stride=1, padding=1, previous_input=True)
    container_cnn_mobile.push_layer(convdw10)
    container_cnn_mobile.push_layer(conv10)

    convdw11 = Layer(Type.DEPTH, batch=batch, in_dim=(14, 14, 512), kernel_dim=(3, 3), stride=1, padding=1, previous_input=True)
    conv11 = Layer(Type.CONV, batch=batch, in_dim=(14, 14, 512), kernel_dim=(1, 1), kernel_num=512, stride=1, padding=1, previous_input=True)
    container_cnn_mobile.push_layer(convdw11)
    container_cnn_mobile.push_layer(conv11)

    convdw12 = Layer(Type.DEPTH, batch=batch, in_dim=(14, 14, 512), kernel_dim=(3, 3), stride=2, padding=1, previous_input=True)
    conv12 = Layer(Type.CONV, batch=batch, in_dim=(7, 7, 512), kernel_dim=(1, 1), kernel_num=1024, stride=1, padding=1, previous_input=True)
    container_cnn_mobile.push_layer(convdw12)
    container_cnn_mobile.push_layer(conv12)

    convdw13 = Layer(Type.DEPTH, batch=batch, in_dim=(7, 7, 1024), kernel_dim=(3, 3), stride=1, padding=1, previous_input=True)
    conv13 = Layer(Type.CONV, batch=batch, in_dim=(7, 7, 1024), kernel_dim=(1, 1), kernel_num=1024, stride=1, padding=1, previous_input=True)
    container_cnn_mobile.push_layer(convdw13)
    container_cnn_mobile.push_layer(conv13)

    pool = Layer(Type.POOL, batch=batch, in_dim=(7, 7, 1024), window_dim=(7, 7), stride=1, previous_input=True)
    fc = Layer(Type.FC, batch=batch, in_dim=1024, out_dim=1000, previous_input=True)
    container_cnn_mobile.push_layer(pool)
    container_cnn_mobile.push_layer(fc)
    

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

