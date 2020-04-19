import random

from layer_compiler.enum_def import Type, Op, Buf
from layer_compiler.layer import Layer, Container
from layer_compiler.compiler import NN
from unit import Mmunit, Vecunit
from scheduler import Scheduler
from buffer import Buffer

KB = 1024
MB = 1024 * 1024
GB = 1024 * 1024 * 1024


def random_priority():
    return random.randint(0, 2)
    

def random_dispatch():
    return int(random.uniform(0, 10000))


if __name__ == '__main__':

    # computation unit and buffer

    MUT = Mmunit(128, 128, 128)
    VUT = Vecunit(128)

    UBUF = Buffer(8*MB/4, 358*GB/4, 100)
    WBUF = Buffer(4*MB/4, 358*GB/4, 100)
    ACCQ = Buffer(128*128, 358*GB/4, 0)
    
    # random container generator
    # for sample, just all fc layer instance

    container_1 = Container()
    container_2 = Container()
    container_3 = Container()
    container_4 = Container()

    layer1 = Layer(Type.FC, batch=400, in_dim=1000, out_dim=1000)
    layer2 = Layer(Type.FC, batch=400, in_dim=1000, out_dim=1000)
    layer3 = Layer(Type.FC, batch=400, in_dim=1000, out_dim=1000)
    layer4 = Layer(Type.FC, batch=400, in_dim=1000, out_dim=10)
    container_1.push_layer(layer1)
    container_1.push_layer(layer2)
    container_1.push_layer(layer3)
    container_1.push_layer(layer4)

    layer1 = Layer(Type.FC, batch=400, in_dim=100, out_dim=1000)
    layer2 = Layer(Type.FC, batch=400, in_dim=1000, out_dim=1000)
    layer3 = Layer(Type.FC, batch=400, in_dim=1000, out_dim=1000)
    layer4 = Layer(Type.FC, batch=400, in_dim=1000, out_dim=10)
    container_2.push_layer(layer1)
    container_2.push_layer(layer2)
    container_2.push_layer(layer3)
    container_2.push_layer(layer4)

    layer1 = Layer(Type.FC, batch=400, in_dim=100, out_dim=1000)
    layer2 = Layer(Type.FC, batch=400, in_dim=1000, out_dim=1000)
    layer3 = Layer(Type.FC, batch=400, in_dim=1000, out_dim=1000)
    layer4 = Layer(Type.FC, batch=400, in_dim=1000, out_dim=10)
    container_3.push_layer(layer1)
    container_3.push_layer(layer2)
    container_3.push_layer(layer3)
    container_3.push_layer(layer4)

    layer1 = Layer(Type.FC, batch=400, in_dim=100, out_dim=1000)
    layer2 = Layer(Type.FC, batch=400, in_dim=1000, out_dim=1000)
    layer3 = Layer(Type.FC, batch=400, in_dim=1000, out_dim=1000)
    layer4 = Layer(Type.FC, batch=400, in_dim=1000, out_dim=10)
    container_4.push_layer(layer1)
    container_4.push_layer(layer2)
    container_4.push_layer(layer3)
    container_4.push_layer(layer4)

    
    
    # container to instruction and NN instance

    NN1 = NN(random_priority(), 1, MUT, 0)
    NN2 = NN(random_priority(), 2, MUT, random_dispatch())
    NN3 = NN(random_priority(), 3, MUT, random_dispatch())
    NN4 = NN(random_priority(), 4, MUT, random_dispatch())

    NN1.container_to_inst(container_1)
    NN2.container_to_inst(container_2)
    NN3.container_to_inst(container_3)
    NN4.container_to_inst(container_4)

    SCHED = Scheduler()
    SCHED.push_task(NN1)
    SCHED.push_task(NN2)
    SCHED.push_task(NN3)
    SCHED.push_task(NN4)

    cycle = 0

    task = None
    check_task = None
    nnid = -1

    buf_inst = None
    mm_inst = None
    vec_inst = None

    checkpoint = False
    check_task = None

    print("===== PREMA SIMULATION =====")
    print(SCHED.str_pre())    

    while not SCHED.check_done():

        if cycle != 0 and cycle % 1000 == 0:
            print(f"\n  Cycle = {cycle}")
            if task == None:
                print(f"  Current task = None\n")
            else:
                print(f"  Current task = {task.nnid}\n")
            print(SCHED.str_current())
            input()

        if checkpoint and check_task != None:
            buf_check = False
            mm_check = False
            vec_check = False
            
            if buf_inst == None:
                buf_check = True
            elif buf_inst.done:
                buf_check = True

            if mm_inst == None:
                mm_check = True
            elif mm_inst.done:
                mm_check = True

            if vec_inst == None:
                vec_check = True
            elif vec_inst.done:
                vec_check = True

            if buf_check and mm_check and vec_check:
                delay = 0

                check_nnid = check_task.nnid
                delay += UBUF.checkout(check_nnid)
                delay += ACCQ.checkout(check_nnid)

                nnid = task.nnid
                delay += UBUF.checkout(nnid)
                delay += ACCQ.checkout(nnid)

                check_task.running = False
                cycle += delay

                for i in range(delay):
                    SCHED.dispatch()
                    
                task.running = True
                continue
    
        # dispatch NN
        SCHED.dispatch()
        if SCHED.sched_check():
            check_task = SCHED.current
            SCHED.schedule()
            checkpoint = SCHED.preempt()
            task = SCHED.current
            
            if check_task == None:
                checkpoint = False
                task.running = True

            cycle += 1
            continue

        if task == None:
            print("DEBUG")
            break
        else:
            print(f"DEBUG: {task.nnid}")

        # fetch instruction
        if not checkpoint:
            temp_inst = task.fetch1()
            if temp_inst.fetchable():
                if temp_inst.inst_type in [Op.LOAD_TILE, Op.STORE_TILE]:
                    if buf_inst == None:
                        buf_inst = task.fetch2()
                    elif buf_inst.done:
                        buf_inst = task.fetch2()
                elif temp_inst.inst_type == Op.GEMM_OP:
                    if mm_inst == None:
                        mm_inst = task.fetch2()
                    elif mm_inst.done:
                        mm_inst = task.fetch2()
                elif temp_inst.inst_type == Op.VECTOR_OP:
                    if vec_inst == None:
                        vec_inst = task.fetch2()
                    elif vec_inst.done:
                        vec_inst = task.fetch2()

        nnid = task.nnid

        # buffer processing
        if buf_inst == None:
            pass
        elif buf_inst.buf == Buf.UBUF:
            op = buf_inst.inst_type
            size = buf_inst.size
            nnid = task.nnid
            UBUF.process(op, size, nnid)
        elif buf_inst.buf == Buf.UBUF:
            op = buf_inst.inst_type
            size = buf_inst.size
            nnid = task.nnid
            UBUF.process(op, size, nnid, True)
        
        if UBUF.processing == 0 and buf_inst != None:
            buf_inst.done = True

        # mmunit processing
        if mm_inst == None:
            pass
        else:
            m = mm_inst.M
            k = mm_inst.K
            n = mm_inst.N
            MUT.process(m, k, n)

        if MUT.processing == 0 and mm_inst != None:
            mm_inst.done = True
            ACCQ.save(m*n, nnid)

        # vecunit processing
        if vec_inst == None:
            pass
        else:
            size = vec_inst.size
            VUT.process(size)

        if VUT.processing == 0 and vec_inst != None:
            vec_inst.done = True
            ACCQ.save(size, nnid)

        cycle += 1


    print(f"Cycle = {cycle}")