import random
import argparse

from layer_compiler.enum_def import Type, Op, Buf, Sched, Mecha
from layer_compiler.layer import Layer, Container
from layer_compiler.compiler import NN
from unit import Mmunit, Vecunit
from scheduler import Scheduler
from buffer import Buffer

KB = 1024
MB = 1024 * 1024
GB = 1024 * 1024 * 1024

HEIGHT = 128
WIDTH = 128
DEPTH = 128


def random_priority():
    return random.randint(0, 2)
    

def random_dispatch():
    return int(random.uniform(0, 80000))

def cmd_parse():
    parser = argparse.ArgumentParser(description='Cambricon ISA Timing Simulator')
    
    parser.add_argument('--cfg', required=False, \
        help='microarchitecture and latency config file')
    parser.add_argument('--src', required=True, \
        help='nn cambricon code file')
    
    args = parser.parse_args()
    
    return args.cfg, args.src


if __name__ == '__main__':

    # computation unit and buffer

    MUT = Mmunit(HEIGHT, WIDTH, DEPTH)
    VUT = Vecunit(WIDTH)

    UBUF = Buffer(8*MB/4, 358*GB/4, 100, 'UBUF')
    WBUF = Buffer(4*MB/4, 358*GB/4, 100, 'WBUF')
    ACCQ = Buffer(WIDTH*DEPTH, 358*GB/4, 0, 'ACCQ')
    
    # random container generator
    # for sample, just all fc layer instance

    container_1 = Container()
    container_2 = Container()
    container_3 = Container()
    container_4 = Container()

    layer1 = Layer(Type.FC, batch=200, in_dim=100, out_dim=400)
    layer2 = Layer(Type.FC, batch=200, in_dim=400, out_dim=400, previous_input=True)
    layer3 = Layer(Type.FC, batch=200, in_dim=400, out_dim=400, previous_input=True)
    layer4 = Layer(Type.FC, batch=200, in_dim=400, out_dim=10, previous_input=True)
    container_1.push_layer(layer1)
    container_1.push_layer(layer2)
    container_1.push_layer(layer3)
    container_1.push_layer(layer4)

    layer1 = Layer(Type.FC, batch=200, in_dim=100, out_dim=400)
    layer2 = Layer(Type.FC, batch=200, in_dim=400, out_dim=400, previous_input=True)
    layer3 = Layer(Type.FC, batch=200, in_dim=400, out_dim=400, previous_input=True)
    layer4 = Layer(Type.FC, batch=200, in_dim=400, out_dim=10, previous_input=True)
    container_2.push_layer(layer1)
    container_2.push_layer(layer2)
    container_2.push_layer(layer3)
    container_2.push_layer(layer4)

    layer1 = Layer(Type.FC, batch=200, in_dim=100, out_dim=400)
    layer2 = Layer(Type.FC, batch=200, in_dim=400, out_dim=400, previous_input=True)
    layer3 = Layer(Type.FC, batch=200, in_dim=400, out_dim=400, previous_input=True)
    layer4 = Layer(Type.FC, batch=200, in_dim=400, out_dim=10, previous_input=True)
    container_3.push_layer(layer1)
    container_3.push_layer(layer2)
    container_3.push_layer(layer3)
    container_3.push_layer(layer4)

    layer1 = Layer(Type.FC, batch=200, in_dim=100, out_dim=400)
    layer2 = Layer(Type.FC, batch=200, in_dim=400, out_dim=400, previous_input=True)
    layer3 = Layer(Type.FC, batch=200, in_dim=400, out_dim=400, previous_input=True)
    layer4 = Layer(Type.FC, batch=200, in_dim=400, out_dim=10, previous_input=True)
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

    # NN to txt
    f1 = open("inst/1.txt", 'w')
    f2 = open("inst/2.txt", 'w')
    f3 = open("inst/3.txt", 'w')
    f4 = open("inst/4.txt", 'w')

    f1.write(NN1.inst_str())
    f2.write(NN2.inst_str())
    f3.write(NN3.inst_str())
    f4.write(NN4.inst_str())

    f1.close()
    f2.close()
    f3.close()
    f4.close()

    SCHED = Scheduler(sched_mode=Sched.PREMA)
    SCHED.push_task(NN1)
    SCHED.push_task(NN2)
    SCHED.push_task(NN3)
    SCHED.push_task(NN4)

    cycle = 0
    switch_overhead = 0

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

    runned_cycles = SCHED.cycle_info()
    compare_cycles = SCHED.cycle_info()    

    while not SCHED.check_done():

        debug_task = task

        
        if cycle != 0 and cycle % 10000 == 0:

            
            print("\n===================================\n")      

            print(f"\n  Cycle = {cycle}\n")

            compare_cycles = SCHED.cycle_info()
            something_runned = False
            for i in range(len(runned_cycles)):
                if compare_cycles[i] > runned_cycles[i]:
                    something_runned = True
            if not something_runned:
                print("\n@@@@@@@@@@ NOTHING RUNNED! @@@@@@@@@@ \n")
                print("  Buf: ", str(buf_inst), str(buf_inst.done))
                print("  MM : ", str(mm_inst), str(buf_inst.done))
                print("  Vec: ", str(vec_inst), str(buf_inst.done))
                print("  Temp: ",  str(vec_inst))
                for i in temp_inst.depend:
                    print(type(i))
                    print("\tDEPEND: ", str(i). str(i.done))
                print("\n@@@@@@@@@@ NOTHING RUNNED! @@@@@@@@@@ \n")
                input()
            runned_cycles = SCHED.cycle_info()

            print(SCHED.str_current())
            
            if task == None:
                print(f"\n  Current task = None\n")
            else:
                print(f"  Current task = {task.nnid}\n")
            
            print("  PC : ", task.pc)
            print("  Buf: ", str(buf_inst))
            print("  MM : ", str(mm_inst))
            print("  Vec: ", str(vec_inst)) 
            print("\n===================================\n")      

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
                checkout_delay = 0
                recover_delay = 0

                check_nnid = check_task.nnid
                checkout_delay += UBUF.checkout(check_nnid)
                checkout_delay += ACCQ.checkout(check_nnid)

                print(f"  Checkout for {checkout_delay} cycle")

                nnid = task.nnid
                recover_delay += UBUF.recover(nnid)
                recover_delay += ACCQ.recover(nnid)

                print(f"  Recover for {recover_delay} cycle")

                check_task.running = False
                cycle += checkout_delay
                cycle += recover_delay

                switch_overhead += checkout_delay
                switch_overhead += recover_delay

                for i in range(checkout_delay+recover_delay):
                    SCHED.dispatch()
                    
                task.running = True
                checkpoint = False
                continue
    
        # dispatch NN
        SCHED.dispatch()
        if SCHED.sched_check(cycle):
            check_task = SCHED.current
            SCHED.schedule(cycle)
            checkpoint = SCHED.preempt(cycle)
            task = SCHED.current

            if check_task != None:
                if task.nnid == check_task.nnid:
                    checkpoint = False
                elif check_task != None:
                    print(f"  Schedule: {check_task.nnid} ==> {task.nnid}")
                    if checkpoint:
                        print("  Mechanism: Checkpoint")
                    else:
                        print("  Mechanism: Drain")
            
            if check_task == None:
                checkpoint = False
                task.running = True

            cycle += 1
            continue

        '''
        if task == None:
            print("DEBUG")
            break
        else:
            print(f"DEBUG: {task.nnid}")
        '''

        nnid = task.nnid

        # fetch instruction
        if not checkpoint:
            temp_inst = task.fetch1()
            if temp_inst.fetchable():
                # store fake
                if temp_inst.inst_type == Op.STORE_FAKE:
                    if temp_inst.buf == Buf.ACCQ:
                        ACCQ.store_fake(nnid)
                    elif temp_inst.buf == Buf.UBUF:
                        UBUF.store_fake(nnid)
                elif temp_inst.inst_type in [Op.LOAD_TILE, Op.STORE_TILE]:
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

                #print(temp_inst)

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
            temp_done = mm_inst.done
            mm_inst.done = True
            if not temp_done and mm_inst.done:
                ACCQ.save(m*n, nnid)

        # vecunit processing
        if vec_inst == None:
            pass
        else:
            size = vec_inst.size
            VUT.process(size)

        if VUT.processing == 0 and vec_inst != None:
            temp_done = vec_inst.done
            vec_inst.done = True
            if not temp_done and vec_inst.done:
                UBUF.save(size, nnid)

        if debug_task.nnid != task.nnid:
            print(f"DEBUG! @ Cycle={cycle}, {debug_task.nnid} ==> {task.nnid}")

        cycle += 1        

    print("======================================\n\n")
    print(f"Cycle = {cycle}")
    print(f"Overhead = {switch_overhead}\n")
    print(SCHED.str_current())
    print("\n======================================\n")