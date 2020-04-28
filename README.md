# Prema Simulator

This is a simulator for Prema scheduler(https://arxiv.org/pdf/1909.04548.pdf). You can simulate many different workloads and scheduling algorithms/mechanisms. 

You can determine following options:

1. Number of inference instances(processes) to run
2. Type of instance to run(AlexNet, VGG16, GoogLeNet, MobileNet, LSTM for Machine Translation, Sentiment Analysis, and Automatic Speech Recognition)
3. Batch Size
4. Scheduling Algorithm(Prema, FCFS(First-Come-First-Serve), RRB(Round-Robin), HPF(High Priority First), Token(Prema without runtime prediction), SJF(Shortest-Job-First))
5. Scheduling Mechanism(Dynamic, Static(always preempt))

## Usage

For default simulation:

```
python simul.py
```

Default setting is:

> batch size = random among (1, 4, 16)
>
> instance number = random among (2, 8)
>
> scheduling algorithm = Prema
>
> scheduling mechanism = Dynamic
>
> during execution, shell only shows how many cycles has passed, current task and program counter of current task

To simulate manually, you can set some arguments:

```
python simul.py --algo FCFS --mecha DYNAMIC --period 1000000 --batch 4 --num 8
```

Each arguments corresponds to:

> algo: Scheduling algorithm: {FCFS, RRB, HPF, TOKEN, SJF, PREMA}, you can type in lowercase or the first letter
>
> mecha: Scheduling mechanism: {DYANAMIC, STATIC}, you can type in lowercase or the first letter
>
> period: Show information about every process periodically, if <=0, default setting
>
> batch: batch size
>
> num: number of instance

## Output

When the simulation is finished, you can find two csv file in result folder. One shows the scheduling log and one shows the instance information.

## Test

Using run.sh and run_static.sh you can simulate over every scheduling algorithms. Using csv files from the test, you can run test.py to derive ANTT, STP, Fairness graph.

## Instruction Wrapper

For the sake of simplicity, I made a wrapper for the ISA which is supported by the accelerator.  ISA used in this simulation is equal to the paper, but since paper didn't provide precise ISA there might be some difference between the one the paper used and the one this simulator used.

ISA includes following instructions:

> LOAD_TILE: Loads input activations / weight
>
> GEMM_OP: Matrix multiplication
>
> CONV_OP: Convolution lowering + GEMM_OP
>
> VECTOR_OP: Element-wise operations for vector input
>
> STORE_TILE: Stores the output activations

Since it is difficult and time-consuming to implement all the network using this instructions, I implemented wrapper module for easy implementation. 

There are two classes for this: Layer and Container. If you define some Layers and push it to the container, when you instantiate this container to instance to run, it automatically change Layers in the container to corresponding instructions. In fact, instructions are also handled as class instances. But every class supports string wrapper so you can easily print it to instruction format.

The interface of Layer is very similar to deep learning libraries like pytorch, so it is easy to use. There are 6 types in Layer class:

> Type.FC: Fully connected layer
>
> ```
> layer1 = Layer(Type.FC, batch=batch, in_dim=100, out_dim=400)
> ```
>
> Type.CONV: (Standard) Convolution layer. If kernel_dim==(1, 1), then it is pointwise convolution
>
> ```
> conv1 = Layer(Type.CONV, batch=batch, in_dim=(224, 224, 3), kernel_dim=(11, 11), kernel_num=96, stride=4, padding=0)
> ```
>
> Type.LSTM: LSTM layer
>
> ```
> layer_lstm1 = Layer(Type.LSTM, batch=N, in_dim=D, h_dim=H, no_hidden=True, previous_input=False)
> ```
>
> Type.GEMM: Matrix multiplication layer. This is for some modern type layer such as attention
>
> ```
> attention_score = Layer(Type.GEMM, batch=N, gemm_m=To, gemm_k=D_HIDDEN, gemm_n=Ti, previous_input=True)
> ```
>
> Type.POOL: Pooling layer
>
> ```
> pool = Layer(Type.POOL, batch=batch, in_dim=in_dim, window_dim=(3, 3), stride=1, previous_input=True)
> ```
>
> Type.DEPTH: Depthwise convolution layer. This is for MobileNet
>
> ```
> convdw1 = Layer(Type.DEPTH, batch=batch, in_dim=(112, 112, 32), kernel_dim=(3, 3), stride=1, padding=1, previous_input=True)
> ```

7 network models which are used as a benchmark in the paper (AlexNet, VGG16, GoogLeNet, MobileNet, LSTM for Machine Translation, Sentiment Analysis, and Automatic Speech Recognition) are already implemented as Container instance in sample_task.py and simulator use these containers to randomly select which network to instantiate.