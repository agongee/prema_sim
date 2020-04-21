from layer_compiler.compiler import NN

class Scheduler:
    def __init__(self, slice=175000):
        self.queue = []
        self.threshold = 3
        self.current = None
        self.candidate = None
        self.new_dispatch = False
        self.slice = slice
        self.elapsed = 0
        self.mode = 
    
    def push_task(self, task: NN):
        if isinstance(task, list):
            self.queue.extend(task)
        else:
            self.queue.append(task)

    def schedule(self, cycle):
        print("+++++SCHEDULE SCHEDULE SCHEDULE+++++")
        valid = 0
        single_task = None
        for i in self.queue:
            if i.dispatched and not i.done:
                valid += 1
                single_task = i
        if valid == 1:
            self.candidate = single_task
            print(f"+++++ SCHEDULE case 1 {self.candidate.nnid} at {cycle} +++++")
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
        print(f"+++++ SCHEDULE case 2 {self.candidate.nnid} at {cycle} +++++")
        return res

    def preempt(self, cycle):
        if self.current == None:
            print(f"***** PREEMPT [current_none] at {cycle} *****")
            self.current = self.candidate
            return True
        
        degradation_current = self.current.remaining / self.current.estimated
        degradation_candidate = self.candidate.remaining / self.candidate.estimated

        self.current = self.candidate
        
        if degradation_current > degradation_candidate:
            return False
        else:
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
            elif i.dispatched:
                i.waited += 1

    def check_done(self):
        return all(self.queue)

    def sched_check(self, cycle):
        res = False
        if self.current == None:
            print(f"===== SCHED_CHECK [current_none] at {cycle} =====")
            return True
        if self.new_dispatch:
            self.new_dispatch = False
            print(f"===== SCHE_CHECH [new_dispatch] at {cycle} =====")
            res = True
        if self.current.done:
            print(f"===== SCHE_CHECH [current_done] at {cycle} =====")
            self.current = None
            res = True
        if self.elapsed == self.slice:
            print(f"===== SCHE_CHECH [time_slice] at {cycle} =====")
            self.elapsed = 0
            res = True

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