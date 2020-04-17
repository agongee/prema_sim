from layer_compiler.layer import Layer, Container
from enum_def import Type, Op, Buf





if __name__ == '__main__':
    
    # random generator
    # for sample, just all fc layer instance

    container_1 = Container()
    container_2 = Container()
    container_3 = Container()
    container_4 = Container()

    layer1 = Layer(Type.FC, batch=1, in_dim=100, out_dim=1000)
    layer2 = Layer(Type.FC, batch=1, in_dim=1000, out_dim=1000)
    layer3 = Layer(Type.FC, batch=1, in_dim=1000, out_dim=1000)
    layer4 = Layer(Type.FC, batch=1, in_dim=1000, out_dim=10)
    container_1.push_layer(layer1)
    container_1.push_layer(layer2)
    container_1.push_layer(layer3)
    container_1.push_layer(layer4)

    layer1 = Layer(Type.FC, batch=1, in_dim=100, out_dim=1000)
    layer2 = Layer(Type.FC, batch=1, in_dim=1000, out_dim=1000)
    layer3 = Layer(Type.FC, batch=1, in_dim=1000, out_dim=1000)
    layer4 = Layer(Type.FC, batch=1, in_dim=1000, out_dim=10)
    container_2.push_layer(layer1)
    container_2.push_layer(layer2)
    container_2.push_layer(layer3)
    container_2.push_layer(layer4)

    layer1 = Layer(Type.FC, batch=1, in_dim=100, out_dim=1000)
    layer2 = Layer(Type.FC, batch=1, in_dim=1000, out_dim=1000)
    layer3 = Layer(Type.FC, batch=1, in_dim=1000, out_dim=1000)
    layer4 = Layer(Type.FC, batch=1, in_dim=1000, out_dim=10)
    container_3.push_layer(layer1)
    container_3.push_layer(layer2)
    container_3.push_layer(layer3)
    container_3.push_layer(layer4)

    layer1 = Layer(Type.FC, batch=1, in_dim=100, out_dim=1000)
    layer2 = Layer(Type.FC, batch=1, in_dim=1000, out_dim=1000)
    layer3 = Layer(Type.FC, batch=1, in_dim=1000, out_dim=1000)
    layer4 = Layer(Type.FC, batch=1, in_dim=1000, out_dim=10)
    container_4.push_layer(layer1)
    container_4.push_layer(layer2)
    container_4.push_layer(layer3)
    container_4.push_layer(layer4)


    