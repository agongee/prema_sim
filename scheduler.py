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
            self.schedule_toekn(cycle)
        elif self.sched_mode == Sched.SJF:
            self.schedule_sjf(cycle)

    def preempt(self, cycle):
        if self.mecha_mode == Mecha.DYNAMIC:
            return self.preempt_dynamic(cycle)
        elif self.mecha_mode == Mecha.STATIC:
            return self.preempt_static(cycle)

    def schedule_prema(self, cycle):
        valid = 0
        single_task = None
        for i in self.queue:
            if i.dispatched and not i.done:
                valid += 1
                single_task = i
        if valid == 1:
            self.candidate = single_task
            print(f"+++++ SCHEDULE [case 1] {self.candidate.nnid} at {cycle} +++++")
            return single_task
        elif valid == 0:
            print(f"+++++ SCHEDULE NOTHING at {cycle} +++++")
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
        print(f"+++++ SCHEDULE [case 2] {self.candidate.nnid} at {cycle} +++++")
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
        if not self.current.done and self.elapsed != 1:
            self.candidate = self.current
            return self.candidate

        current_nnid = -1
        res = None

        if self.current == None:
            for i in self.queue:
                if i.dispatched and not i.done:
                    res = i
                    break
            self.candidate = res
            return res
        else:
            current_nnid = self.current.nnid
            for i in range(current_nnid+1, len(self.queue)):
                if self.queue[i].dispatched and not self.queue[i].done:
                    res = self.queue[i]
                    break
            if res != None:
                self.candidate = res
                return res
            else:
                for i in range(0, current_nnid+1):
                    if self.queue[i].dispatched and not self.queue[i].done:
                        res = self.queue[i]
                        break
                
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
            elif i.dispatch_time_first < res.dispatch_time_first:
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
            print(f"***** PREEMPT [current_none] at {cycle} *****")
            self.current = self.candidate
            return True
        
        degradation_current = self.current.remaining / self.current.estimated
        degradation_candidate = self.candidate.remaining / self.candidate.estimated

        self.current = self.candidate
        
        if degradation_current > degradation_candidate:
            print(f"***** PREEMPT [DRAIN] at {cycle} *****")
            return False
        else:
            print(f"***** PREEMPT [CHECKPOINT] at {cycle} *****")
            return True

    def preempt_static(self, cycle):
        return True

    def dispatch(self):
        for i in self.queue:
            pre_state = i.dispatched
            i.dispatch_nn()
            if not pre_state and i.dispatched:
                print(f"NEW DISPATCH: NNID = {i.nnid}")
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
            print(f"===== SCHED_CHECK [current_none] at {cycle} =====")
            return True
        if self.new_dispatch:
            self.new_dispatch = False
            print(f"===== SCHED_CHECH [new_dispatch] at {cycle} =====")
            res = True
        if self.current.done:
            print(f"===== SCHED_CHECH [current_done] at {cycle} =====")
            self.current = None
            res = True
        if self.elapsed == self.slice:
            print(f"===== SCHED_CHECH [time_slice] at {cycle} =====")
            self.elapsed = 0
            res = True

        return res        

    def cycle_info(self):
        res = []
        for i in self.queue:
            res.append(i.pc)
        return res

    def str_pre(self):
        res = ""
        for i in self.queue:
            res += i.str_pre()

        return res

    def str_current(self):
        res = ""
        for i in self.queue:
            res += i.str_current()

        return res