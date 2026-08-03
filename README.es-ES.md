

# Prema Simulator

Este es un simulador para el planificador Prema (https://arxiv.org/pdf/1909.04548.pdf). Puedes simular muchas cargas de trabajo diferentes y algoritmos/mecanismos de planificación. 

Puedes determinar las siguientes opciones:

1. Número de instancias de inferencia (procesos) a ejecutar
2. Tipo de instancia a ejecutar (AlexNet, VGG16, GoogLeNet, MobileNet, LSTM para Traducción Automática, Análisis de Sentimientos y Reconocimiento Automático de Voz)
3. Tamaño de lote (Batch Size)
4. Algoritmo de Planificación (Prema, FCFS(First-Come-First-Serve), RRB(Round-Robin), HPF(High Priority First), Token(Prema sin predicción de tiempo de ejecución), SJF(Shortest-Job-First))
5. Mecanismo de Planificación (Dynamic, Static(always preempt))

## Uso

Para la simulación por defecto:

```
python simul.py
```

La configuración por defecto es:

> tamaño de lote (batch size) = aleatorio entre (1, 4, 16)
>
> número de instancias = aleatorio entre (2, 8)
>
> algoritmo de planificación = Prema
>
> mecanismo de planificación = Dynamic
>
> durante la ejecución, la terminal solo muestra cuántos ciclos han transcurrido, la tarea actual y el contador de programa de la tarea actual

Para simular manualmente, puedes configurar algunos argumentos:

```
python simul.py --algo FCFS --mecha DYNAMIC --period 1000000 --batch 4 --num 8
```

Cada argumento corresponde a:

> algo: Algoritmo de planificación: {FCFS, RRB, HPF, TOKEN, SJF, PREMA}, puedes escribirlo en minúsculas o solo la primera letra
>
> mecha: Mecanismo de planificación: {DYANAMIC, STATIC}, puedes escribirlo en minúsculas o solo la primera letra
>
> period: Mostrar información sobre cada proceso periódicamente, si <=0, configuración por defecto
>
> batch: tamaño de lote
>
> num: número de instancias

## Salida

Cuando se complete la simulación, podrás encontrar dos archivos csv en la carpeta result. Uno muestra el registro de planificación y el otro muestra la información de las instancias.

## Pruebas

Utilizando run.sh y run_static.sh puedes simular todos los algoritmos de planificación. Usando los archivos csv de la prueba, puedes ejecutar test.py para generar los gráficos de ANTT, STP y Fairness.

## Wrapper de Instrucciones

Por simplicidad, creé un wrapper para el ISA soportado por el acelerador. El ISA utilizado en esta simulación es igual al del artículo, pero como el artículo no proporcionó un ISA preciso, podría haber alguna diferencia entre el utilizado en el artículo y el utilizado por este simulador.

El ISA incluye las siguientes instrucciones:

> LOAD_TILE: Carga activaciones de entrada / pesos
>
> GEMM_OP: Multiplicación de matrices
>
> CONV_OP: Reducción de convolución + GEMM_OP
>
> VECTOR_OP: Operaciones elemento a elemento para entrada vectorial
>
> STORE_TILE: Almacena las activaciones de salida

Dado que es difícil y consume mucho tiempo implementar toda la red utilizando estas instrucciones, implementé un módulo wrapper para facilitar la implementación. 

Hay dos clases para esto: Layer y Container. Si defines algunas Layers y las agregas al container, al instanciar este container como una instancia para ejecutar, cambia automáticamente las Layers dentro del container a las instrucciones correspondientes. De hecho, las instrucciones también se manejan como instancias de clase. Pero cada clase soporta un wrapper de cadena para que puedas imprimirla fácilmente en formato de instrucción.

La interfaz de Layer es muy similar a las bibliotecas de aprendizaje profundo como PyTorch, por lo que es fácil de usar. Hay 6 tipos en la clase Layer:

> Type.FC: Capa totalmente conectada
>
> ```
> layer1 = Layer(Type.FC, batch=batch, in_dim=100, out_dim=400)
> ```
>
> Type.CONV: Capa de convolución (estándar). Si kernel_dim==(1, 1), entonces es una convolución puntual
>
> ```
> conv1 = Layer(Type.CONV, batch=batch, in_dim=(224, 224, 3), kernel_dim=(11, 11), kernel_num=96, stride=4, padding=0)
> ```
>
> Type.LSTM: Capa LSTM
>
> ```
> layer_lstm1 = Layer(Type.LSTM, batch=N, in_dim=D, h_dim=H, no_hidden=True, previous_input=False)
> ```
>
> Type.GEMM: Capa de multiplicación de matrices. Esto es para algunos tipos de capas modernas como la de atención
>
> ```
> attention_score = Layer(Type.GEMM, batch=N, gemm_m=To, gemm_k=D_HIDDEN, gemm_n=Ti, previous_input=True)
> ```
>
> Type.POOL: Capa de agrupamiento (Pooling)
>
> ```
> pool = Layer(Type.POOL, batch=batch, in_dim=in_dim, window_dim=(3, 3), stride=1, previous_input=True)
> ```
>
> Type.DEPTH: Capa de convolución depthwise. Esto es para MobileNet
>
> ```
> convdw1 = Layer(Type.DEPTH, batch=batch, in_dim=(112, 112, 32), kernel_dim=(3, 3), stride=1, padding=1, previous_input=True)
> ```

Los 7 modelos de red que se utilizan como referencia (benchmark) en el artículo (AlexNet, VGG16, GoogLeNet, MobileNet, LSTM para Traducción Automática, Análisis de Sentimientos y Reconocimiento Automático de Voz) ya están implementados como instancias de Container en sample_task.py y el simulador utiliza estos contenedores para seleccionar aleatoriamente qué red instanciar.
