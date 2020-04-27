import random
import argparse
from sys import exit
from datetime import datetime

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
    
def random_dispatch(N):
    return int(random.uniform(0, 500000*N))

def random_batch():
    index = random.randint(0, 2)
    if index == 0:
        return 1
    elif index == 1:
        return 4
    elif index == 2:
        return 16

def random_container():
    index = random.randint(0, 6)
    if index == 0:
        return container_cnn_alex
    elif index == 1:
        return container_cnn_google
    elif index == 2:
        return container_cnn_mobile
    elif index == 3:
        return container_cnn_vgg
    elif index == 4:
        return container_rnn_asr
    elif index == 5:
        return container_rnn_mt
    elif index == 6:
        return container_rnn_sa

def cmd_parse():
    parser = argparse.ArgumentParser(description='Prema Scheduler Simulator')
    
    parser.add_argument('--algo', required=False, \
        help='Scheduling Algorithm Selection: {FCFS, RRB, HPF, TOKEN, SJF, PREMA}')
    parser.add_argument('--mecha', required=False, \
        help='Scheduling Mechanism Selection: {DYANAMIC, STATIC}')
    parser.add_argument('--period', required=False, type=int, \
        help='Show Procedure Periodically, if <= 0, Not Show')
    parser.add_argument('--batch', required=False, type=int, \
        help='Batch Size (1, 4, 16 recommended)')
    parser.add_argument('--num', required=False, type=int, \
        help='Instance Number')
    
    args = parser.parse_args()

    algo = None
    mecha = None
    period = 0
    batch = -1
    num = -1

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
    elif args.algo in ['PREMA', 'prema', 'P', 'p']:
        algo = Sched.PREMA
    
    if args.mecha == None:
        mecha = Mecha.DYNAMIC
    if args.mecha in ['STATIC', 'static', 'S', 's']:
        mecha = Mecha.STATIC
    elif args.mecha in ['DYNAMIC', 'dynamic', 'D', 'd']:
        mecha = Mecha.DYNAMIC

    if args.period != None:
        period = args.period

    if args.batch != None:
        batch = args.batch
    
    if args.num != None:
        num = args.num
    
    return algo, mecha, period, batch, num


if __name__ == '__main__':

    now = datetime.now()
    filename = now.strftime("%Y-%m-%d_%H-%M-%S")

    algo, mecha, period, B, N = cmd_parse()

    # computation unit and buffer

    MUT = Mmunit(HEIGHT, WIDTH, DEPTH)
    VUT = Vecunit(WIDTH)

    UBUF = SimpleBuffer(8*MB/4, 358*GB/4, 100, 'UBUF')
    WBUF = SimpleBuffer(4*MB/4, 358*GB/4, 100, 'WBUF')
    ACCQ = SimpleBuffer(WIDTH*DEPTH, 358*GB/4, 0, 'ACCQ')
    
    # random container generator
    if B < 0:
        B = random_batch()
    if N < 0:
        N = random.randint(2, 8)

    algo_str = None
    mech_str = None

    if algo == Sched.FCFS:
        algo_str = "FCFS"
    elif algo == Sched.HPF:
        algo_str = "HPF"
    elif algo == Sched.PREMA:
        algo_str = "PREMA"
    elif algo == Sched.RRB:
        algo_str = "RRB"
    elif algo == Sched.SJF:
        algo_str = "SJF"
    elif algo == Sched.TOKEN:
        algo_str = "TOKEN"

    if mecha == Mecha.DYNAMIC:
        mecha_str = "DYNAMIC"
    elif mecha == Mecha.STATIC:
        mecha_str = "STATIC"
    
    filename = algo_str + "_" + mecha_str + "_BATCH_" + str(B) + "_NUM_" + str(N) + "_" + filename

    all_init(B, random.randint(10, 50))
    SCHED = Scheduler(sched_mode=algo, mecha_mode=mecha)
    
    for i in range(N):
        if i == 0:
            disp = 0
        else:
            disp = random_dispatch(N)
        NN_temp = NN(random_priority(), i+1, MUT, disp)
        NN_temp.container_to_inst(random_container())
        SCHED.push_task(NN_temp)

    cycle = 0
    switch_overhead = 0

    task = None
    check_task = None
    nnid = -1

    buf_inst = None
    mm_inst = None
    vec_inst = None
    temp_inst = None

    checkpoint = False
    check_task = None

    print("========== PREMA SIMULATION ==========\n")
    print(SCHED.str_pre())
    print(f"  Batch Size = {B}")
    print(f"  Instance Num = {N}\n")

    runned_cycles = SCHED.cycle_info()
    compare_cycles = SCHED.cycle_info()    


    while not SCHED.check_done():

        debug_task = task
        
        if period > 0:
            if cycle != 0 and cycle % period == 0:
                
                print("\n===================================\n")      

                print(f"\n  Cycle = {cycle}\n")
                print(SCHED.str_current())

                if task == None:
                    print(f"\n  Current task = None\n")
                else:
                    print(f"  Current task = {task.nnid}\n")
                    print("  PC : ", task.pc)
                    print("  Buf: ", str(buf_inst), str(buf_inst.done))
                    print("  MM : ", str(mm_inst), str(buf_inst.done))
                    print("  Vec: ", str(vec_inst), str(buf_inst.done))      

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
                    print("\n")
                    if temp_inst != None:
                        for i in temp_inst.depend:
                            print("  DEPEND: ", i, i.done)  
                    print("\n@@@@@@@@@@ NOTHING RUNNED! @@@@@@@@@@ \n")
                    input()
                    #exit(0)
                runned_cycles = SCHED.cycle_info()

                print("\n===================================\n")

        else:
            if cycle != 0 and cycle % 100000 == 0:
                if task != None:
                    print(f"Cycle = {cycle}, Current = {task.nnid}, PC = {task.pc}/{len(task.inst)}")
                else:
                    print(f"Cycle = {cycle}, Current = None")

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
            # print("CHECKOUT DOING")
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

                SCHED.dispatch()
                UBUF.process()
                WBUF.process()
                MUT.process()
                VUT.process()
                continue
    
        if SCHED.sched_check(cycle):
            check_task = SCHED.current
            SCHED.schedule(cycle)
            checkpoint = SCHED.preempt(cycle)
            task = SCHED.current

            if task == None:
                cycle += 1

                SCHED.dispatch()
                UBUF.process()
                WBUF.process()
                MUT.process()
                VUT.process()
                continue


            if check_task != None:
                if task.nnid == check_task.nnid:
                    checkpoint = False
            # check_task == None, initial state, nothing at first
            else:
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

            SCHED.dispatch()
            UBUF.process()
            WBUF.process()
            MUT.process()
            VUT.process()
            continue

        # dispatch NN
        SCHED.dispatch()

        nnid = task.nnid

        # fetch instruction
        if not checkpoint:
            temp_inst = task.fetch1()
            # store_fake, nop
            while temp_inst.inst_type == Op.NOP or temp_inst.inst_type == Op.STORE_FAKE:
                if not temp_inst.fetchable():
                    print(f"DEBUG: NOT FETCHABLE, NNID={task.nnid}")
                    break
                if temp_inst.inst_type == Op.STORE_FAKE:
                    if temp_inst.buf == Buf.ACCQ:
                        ACCQ.store_fake(nnid)
                    elif temp_inst.buf == Buf.UBUF:
                        UBUF.store_fake(nnid)
                print(f"\tSTORE_FAKE before cycle = {task.pc}")
                task.fetch2()
                print(f"\tSTORE_FAKE after cycle = {task.pc}")
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
            if buf_inst.buf == Buf.UBUF:
                buf_inst.done = True
        if WBUF.processing == 0 and buf_inst != None:
            if buf_inst.buf == Buf.WBUF:
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
    SCHED.scheduler_info(filename)
    SCHED.instance_info(N, filename)
    print("\n======================================\n")