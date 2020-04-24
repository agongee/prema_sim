import random
import argparse
from sys import exit

from layer_compiler.enum_def import Type, Op, Buf, Sched, Mecha
from layer_compiler.layer import Layer, Container
from layer_compiler.compiler import NN
from unit import Mmunit, Vecunit
from scheduler import Scheduler
from buffer_simple import SimpleBuffer
from layer_compiler.sample_task import all_init, \
    container_cnn_alex, container_cnn_google, container_cnn_vgg, container_cnn_mobile, \
         container_rnn_asr, container_rnn_mt, container_rnn_sa

KB = 1024
MB = 1024 * 1024
GB = 1024 * 1024 * 1024

HEIGHT = 128
WIDTH = 128
DEPTH = 128


def random_priority():
    return random.randint(0, 2)
    

def random_dispatch():
    return int(random.uniform(0, 5000000))

def cmd_parse():
    parser = argparse.ArgumentParser(description='Prema Scheduler Simulator')
    
    parser.add_argument('--algo', required=False, \
        help='Scheduling Algorithm Selection: {FCFS, RRB, HPF, TOKEN, SJF, PREMA}')
    parser.add_argument('--mecha', required=False, \
        help='Scheduling Mechanism Selection: {DYANAMIC, STATIC}')
    parser.add_argument('--period', required=False, type=int, \
        help='Show Procedure Periodically, if <= 0, Not Show')
    
    args = parser.parse_args()

    algo = None
    mecha = None
    period = 100000

    if args.algo == None:
        algo = Sched.PREMA
    elif args.algo in ['FCFS', 'fcfs', 'F', 'f']:
        algo = Sched.FCFS
    elif args.algo in ['RRB', 'rrb', 'R', 'r']:
        algo = Sched.RRB
    elif args.algo in ['HPF', 'hpf', 'H', 'h']:
        algo = Sched.HPF
    elif args.algo in ['TOKEN', 'token', 'T', 't']:
        algo = Sched.TOKEN
    elif args.algo in ['SJF', 'sjf', 'S', 's']:
        algo = Sched.SJF
    
    if args.mecha == None:
        mecha = Mecha.DYNAMIC
    if args.mecha in ['STATIC', 'static', 'S', 's']:
        mecha = Mecha.STATIC
    elif args.mecha in ['DYNAMIC', 'dynamic', 'D', 'd']:
        mecha = Mecha.DYNAMIC

    if args.period != None:
        period = args.period
    
    return algo, mecha, period


if __name__ == '__main__':

    algo, mecha, period = cmd_parse()

    # computation unit and buffer

    MUT = Mmunit(HEIGHT, WIDTH, DEPTH)
    VUT = Vecunit(WIDTH)

    UBUF = SimpleBuffer(8*MB/4, 358*GB/4, 100, 'UBUF')
    WBUF = SimpleBuffer(4*MB/4, 358*GB/4, 100, 'WBUF')
    ACCQ = SimpleBuffer(WIDTH*DEPTH, 358*GB/4, 0, 'ACCQ')
    
    # random container generator

    all_init(4)

    NN1 = NN(random_priority(), 1, MUT, 0)
    NN2 = NN(random_priority(), 2, MUT, random_dispatch())
    NN3 = NN(random_priority(), 3, MUT, random_dispatch())
    NN4 = NN(random_priority(), 4, MUT, random_dispatch())

    NN1.container_to_inst(container_cnn_alex)
    NN2.container_to_inst(container_cnn_alex)
    NN3.container_to_inst(container_rnn_asr)
    NN4.container_to_inst(container_rnn_asr)

    SCHED = Scheduler(sched_mode=algo, mecha_mode=mecha)
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
        
        if period > 0:
            if cycle != 0 and cycle % period == 0:

                
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
                        print("\tDEPEND: ", i, i.done)  
                    print("\n@@@@@@@@@@ NOTHING RUNNED! @@@@@@@@@@ \n")
                    #input()
                    exit(0)
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
        else:
            if cycle != 0 and cycle % 100000 == 0:
                print(f"\n  Cycle = {cycle}\n")

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
            '''
            if SCHED.current != None:
                print(f"  Before Scheduling: {SCHED.current.nnid}")
            else:
                print(f"  Before Scheduling: None")
            '''
            check_task = SCHED.current
            SCHED.schedule(cycle)
            checkpoint = SCHED.preempt(cycle)
            task = SCHED.current
            '''
            if SCHED.current != None:
                print(f"  After Scheduling: {SCHED.current.nnid}")
            else:
                print(f"  After Scheduling: {SCHED.current.nnid}")
            '''
            if check_task != None:
                if task.nnid == check_task.nnid:
                    checkpoint = False
                '''
                else:
                    print(f"  Schedule: {check_task.nnid} ==> {task.nnid}")
                    if checkpoint:
                        print("  Mechanism: Checkpoint")
                    else:
                        print("  Mechanism: Drain")
                '''
            
            if check_task == None:
                checkpoint = False
                task.running = True

            if checkpoint:
                print(f"  For check_task [{check_task.nnid}]:")
                UBUF.context_status(check_task.nnid)
                ACCQ.context_status(check_task.nnid)
                print(f"  For task [{task.nnid}]:")
                UBUF.context_status(task.nnid)
                ACCQ.context_status(task.nnid)

            cycle += 1
            continue

        nnid = task.nnid

        # fetch instruction
        if not checkpoint:
            temp_inst = task.fetch1()
            # store_fake, nop
            while temp_inst.inst_type == Op.NOP or temp_inst.inst_type == Op.STORE_FAKE:
                if not temp_inst.fetchable():
                    break
                if temp_inst.inst_type == Op.STORE_FAKE:
                    if temp_inst.buf == Buf.ACCQ:
                        ACCQ.store_fake(nnid)
                    elif temp_inst.buf == Buf.UBUF:
                        UBUF.store_fake(nnid)
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

                #print(temp_inst)

        # buffer processing
        if buf_inst == None:
            pass
        elif buf_inst.buf == Buf.UBUF:
            op = buf_inst.inst_type
            size = buf_inst.size
            nnid = task.nnid
            UBUF.process(op, size, nnid)
        elif buf_inst.buf == Buf.WBUF:
            op = buf_inst.inst_type
            size = buf_inst.size
            nnid = task.nnid
            WBUF.process(op, size, nnid, True)
        
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