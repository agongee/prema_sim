import csv

from layer_compiler.compiler import NN
from layer_compiler.enum_def import Sched, Mecha

class Scheduler:
    def __init__(self, slice=175000, sched_mode=Sched.PREMA, mecha_mode=Mecha.DYNAMIC):
        self.queue = []
        self.threshold = 3
        self.current = None
        self.candidate = None
        self.new_dispatch = False
        self.slice = slice
        self.elapsed = 0
        self.sched_mode = sched_mode
        self.mecha_mode = mecha_mode
        self.preeempted = 0
        self.preeempted_list = ["PREEMPTED"]
        self.drained = 0
        self.drained_list = ["DRAINED"]
        self.scheduled_reason_list = ["SCHEDULE REASON"]
        self.nnid_list = ["NNID"]
        self.reason = None
    
    def push_task(self, task: NN):
        if isinstance(task, list):
            self.queue.extend(task)
        else:
            self.queue.append(task)

    def schedule(self, cycle):
        if self.sched_mode == Sched.PREMA:
            self.schedule_prema(cycle)
        elif self.sched_mode == Sched.FCFS:
            self.schedule_fcfs(cycle)
        elif self.sched_mode == Sched.RRB:
            self.schedule_rrb(cycle)
        elif self.sched_mode == Sched.HPF:
            self.schedule_hpf(cycle)
        elif self.sched_mode == Sched.TOKEN:
            self.schedule_token(cycle)
        elif self.sched_mode == Sched.SJF:
            self.schedule_sjf(cycle)

    def preempt(self, cycle):
        if self.mecha_mode == Mecha.DYNAMIC:
            res = self.preempt_dynamic(cycle)
        elif self.mecha_mode == Mecha.STATIC:
            res = self.preempt_static(cycle)

        if res:
            self.preeempted += 1
            self.preeempted_list.append(cycle)
            self.drained_list.append(-1)
        else:
            self.drained += 1
            self.drained_list.append(cycle)
            self.preempted_list.append(-1)

        self.scheduled_reason_list.append(self.reason)
        if self.current != None:
            self.nnid_list.append(self.current.nnid)
        else:
            self.nnid_list.append(None)
        self.reason = None

    def schedule_prema(self, cycle):
        valid = 0
        single_task = None
        for i in self.queue:
            if i.dispatched and not i.done:
                valid += 1
                single_task = i
        if valid == 1:
            self.candidate = single_task
            #print(f"+++++ SCHEDULE [case 1] {self.candidate.nnid} at {cycle} +++++")
            return single_task
        elif valid == 0:
            #print(f"+++++ SCHEDULE NOTHING at {cycle} +++++")
            return None
        for i in self.queue:
            slowdown = i.waited / i.estimated
            i.token += i.priority * slowdown

        res = None
        for i in self.queue:
            if not i.dispatched or i.done:
                continue
            if i.token <= self.threshold:
                continue
            if res == None:
                res = i
                continue
            if i.estimated < res.estimated:
                res = i
        if res == None and len(self.queue) != 0:
            for i in self.queue:
                if i.dispatched and not i.done:
                    res = i
                    break
        self.candidate = res
        #print(f"+++++ SCHEDULE [case 2] {self.candidate.nnid} at {cycle} +++++")
        return res

    def schedule_fcfs(self, cycle):
        res = None
        for i in self.queue:
            if i.dispatched and not i.done:
                if res == None:
                    res = i
                elif i.dispatch_first_time < res.dispatch_first_time:
                    res = i
        self.candidate = res
        return res

    def schedule_rrb(self, cycle):
        current_nnid = -1
        res = None

        if self.current == None:
            for i in self.queue:
                if i.dispatched and not i.done:
                    res = i
                    break
            self.candidate = res
            return res

        current_nnid = self.current.nnid
        for i in range(current_nnid, len(self.queue)):
            if self.queue[i].dispatched and not self.queue[i].done:
                res = self.queue[i]
                break
        if res != None:
            self.candidate = res
            return res
        else:
            for i in range(0, current_nnid):
                if self.queue[i].dispatched and not self.queue[i].done:
                    res = self.queue[i]
                    break
        self.candidate = res
        return res

    def schedule_hpf(self, cycle):
        highest = -1
        res = None

        for i in self.queue:
            if not i.dispatched or i.done:
                continue
            if i.priority > highest:
                highest = i.priority
                res = i

        self.candidate = res
        return res

    def schedule_token(self, cycle):
        valid = 0
        single_task = None
        for i in self.queue:
            if i.dispatched and not i.done:
                valid += 1
                single_task = i
        if valid == 1:
            self.candidate = single_task
            return single_task
        elif valid == 0:
            return None
        for i in self.queue:
            slowdown = i.waited / i.estimated
            i.token += i.priority * slowdown

        res = None
        for i in self.queue:
            if not i.dispatched or i.done:
                continue
            if i.token <= self.threshold:
                continue
            if res == None:
                res = i
            elif i.dispatch_first_time < res.dispatch_first_time:
                res = i

        if res == None and len(self.queue) != 0:
            for i in self.queue:
                if i.dispatched and not i.done:
                    res = i
                    break

        self.candidate = res
        return res

    def schedule_sjf(self, cycle):
        res = None
        for i in self.queue:
            if i.dispatched and not i.done:
                if res == None:
                    res = i
                elif i.estimated < res.estimated:
                    res = i
        self.candidate = res
        return res

    def preempt_dynamic(self, cycle):
        if self.current == None:
            #print(f"***** PREEMPT [current_none] at {cycle} *****")
            self.current = self.candidate
            if self.current != None:
                self.current.switched += 1
            if self.current != None:
                self.current.running = True
            return True
        
        degradation_current = self.current.remaining / self.current.estimated
        degradation_candidate = self.candidate.remaining / self.candidate.estimated
        
        if degradation_current > degradation_candidate:
            #print(f"***** PREEMPT [DRAIN] at {cycle} *****")
            return False
        else:
            #print(f"***** PREEMPT [CHECKPOINT] at {cycle} *****")
            if self.current != None:
                self.current.running = False
                if self.current.nnid != self.candidate.nnid:
                    self.candidate.switched += 1
            else:
                self.candidate.switched =+ 1
            self.current = self.candidate
            if self.current != None:
                self.current.running = True
            return True

    def preempt_static(self, cycle):
        if self.current != None:
            self.current.running = False
            if self.current.nnid != self.candidate.nnid:
                self.candidate.switched += 1
        else:
            self.candidate.switched =+ 1
        self.current = self.candidate
        if self.current != None:
            self.current.running = True
        return True

    def dispatch(self):
        for i in self.queue:
            pre_state = i.dispatched
            i.dispatch_nn()
            if not pre_state and i.dispatched:
                #print(f"NEW DISPATCH: NNID = {i.nnid}")
                self.new_dispatch = True

            if i.running:
                i.runned += 1
            elif i.dispatched and not i.done:
                i.waited += 1

        self.elapsed += 1

    def check_done(self):
        return all(self.queue)

    def sched_check(self, cycle):
        res = False
        if self.current == None:
            #print(f"===== SCHED_CHECK [current_none] at {cycle} =====")
            self.reason = 'DONE'
            return True
        if self.new_dispatch:
            self.new_dispatch = False
            #print(f"===== SCHED_CHECH [new_dispatch] at {cycle} =====")
            self.reason = 'NEW'
            res = True
        if self.current.done:
            #print(f"===== SCHED_CHECH [current_done] at {cycle} =====")
            self.current = None
            self.reason = 'DONE'
            res = True
        if self.elapsed == self.slice:
            #print(f"===== SCHED_CHECH [time_slice] at {cycle} =====")
            self.elapsed = 0
            self.reason = 'SLICE'
            res = True

        return res        

    def antt(self):
        res = 0
        for i in self.queue:
            i.elapsed = i.runned + i.waited
            res += (i.elapsed / i.isolated)
        res /= len(self.queue)

        return res

    def stp(self):
        res = 0
        for i in self.queue:
            i.elapsed = i.runned + i.waited
            res += (i.isolated / i.elapsed)

        return res

    def fariness(self):
        pp = []
        p_sum = 0
        
        for i in self.queue:
            p_sum += i.token

        for i in self.queue:
            c_ratio = i.isolated / i.elapsed
            p_ratio = i.token / p_sum
            pp.append(c_ratio/p_ratio)

        max_pp = max(pp)
        min_pp = min(pp)

        return min_pp / max_pp    

    def cycle_info(self):
        res = []
        for i in self.queue:
            res.append(i.pc)
        return res

    def scheduler_info(self, filename='default'):
        print("\n========== Scheduler Summary ==========\n")
        print(f"  Preempted: {self.preeempted}")
        print(f"  Drained:  {self.drained}\n")
        
        filename = "result/scheduler_" + filename + ".csv"
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.preeempted_list)
            writer.writerow(self.drained_list)
            writer.writerow(self.scheduled_reason_list)
            writer.writerow(self.nnid_list)
            writer.writerow(["Preempted", self.preeempted, "Drained", self.drained])

            algo_str = None
            mecha_str = None

            if self.sched_mode == Sched.FCFS:
                algo_str = "FCFS"
            elif self.sched_mode == Sched.HPF:
                algo_str = "HPF"
            elif self.sched_mode == Sched.PREMA:
                algo_str = "PREMA"
            elif self.sched_mode == Sched.RRB:
                algo_str = "RRB"
            elif self.sched_mode == Sched.SJF:
                algo_str = "SJF"
            elif self.sched_mode == Sched.TOKEN:
                algo_str = "TOKEN"

            if self.mecha_mode == Mecha.DYNAMIC:
                mecha_str = "DYNAMIC"
            elif self.mecha_mode == Mecha.STATIC:
                mecha_str = "STATIC"

            writer.writerow(["algo_str, mecha_str"])

        print("  Scheduler CSV file generated!\n")

    def str_pre(self, cont=False):
        algo_str = None
        mecha_str = None

        if self.sched_mode == Sched.FCFS:
            algo_str = "FCFS"
        elif self.sched_mode == Sched.HPF:
            algo_str = "HPF"
        elif self.sched_mode == Sched.PREMA:
            algo_str = "PREMA"
        elif self.sched_mode == Sched.RRB:
            algo_str = "RRB"
        elif self.sched_mode == Sched.SJF:
            algo_str = "SJF"
        elif self.sched_mode == Sched.TOKEN:
            algo_str = "TOKEN"

        if self.mecha_mode == Mecha.DYNAMIC:
            mecha_str = "DYNAMIC"
        elif self.mecha_mode == Mecha.STATIC:
            mecha_str = "STATIC"

        res = "  Scheduler Algorithm: "
        res += algo_str
        res += "\n"

        res += "  Scheduler Mechanism: "
        res += mecha_str
        res += "\n\n"
        
        for i in self.queue:
            res += i.str_pre(cont)

        return res

    def str_current(self):
        res = ""
        for i in self.queue:
            res += i.str_current()

        return res

    def instance_info(self, N, filename='default'):
        nnid = ["NNID"]
        name = ["NET NAME"]
        isolated = ["ISOLATED"]
        priority = ["PRIORITY"]
        dispatched = ["DISPATCHED"]
        estimated = ["ESTIMATED"]
        runned = ["RUNNED"]
        waited = ["WAITED"]
        elapsed = ["ELAPSED"]
        switched = ["SWITCHED"]
        sla = ["SLA"]

        result = [nnid, name, isolated, priority, dispatched, estimated, runned, waited, elapsed, switched, sla]
        l = len(result)

        print("\n========== Instance Summary ==========\n")
        for i in self.queue:
            arr = i.summary(N)
            for j in range(l):
                result[j].append(arr[j])

        a = self.antt()
        s = self.stp()
        f = self.fariness()

        print("  ANTT:", a)
        print("  STP:", s)
        print("  FARINESS:", f)
        print("\n")

        filename = "result/instance_" + filename + ".csv"
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            for i in result:
                writer.writerow(i)

            algo_str = None
            mecha_str = None

            if self.sched_mode == Sched.FCFS:
                algo_str = "FCFS"
            elif self.sched_mode == Sched.HPF:
                algo_str = "HPF"
            elif self.sched_mode == Sched.PREMA:
                algo_str = "PREMA"
            elif self.sched_mode == Sched.RRB:
                algo_str = "RRB"
            elif self.sched_mode == Sched.SJF:
                algo_str = "SJF"
            elif self.sched_mode == Sched.TOKEN:
                algo_str = "TOKEN"

            if self.mecha_mode == Mecha.DYNAMIC:
                mecha_str = "DYNAMIC"
            elif self.mecha_mode == Mecha.STATIC:
                mecha_str = "STATIC"

            writer.writerow([algo_str, mecha_str])

            

            writer.writerow(["ANTT", a, "STP", s, "FAIRNESS", f])

        print("  Instance CSV file generated!\n")
        