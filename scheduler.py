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
    
    def push_task(self, task: NN):
        if isinstance(task, list):
            self.queue.extend(task)
        else:
            self.queue.append(task)

    def schedule(self):
        valid = 0
        single_task = None
        for i in self.queue:
            if i.dispatched:
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
            if i.token <= self.threshold:
                continue
            if res == None:
                res = i
                continue
            if i.estimated < res.estimated:
                res = i

        if res == None and len(self.queue) != 0:
            res = self.queue[0]

        self.candidate = res
        return res

    def preempt(self):
        if self.current == None:
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
            i.dispatch()
            if i.running:
                i.runned += 1
            elif i.dispatched:
                i.waited += 1

    def check_done(self):
        return all(self.queue)

    def sched_check(self):
        res = False
        if self.current == None:
            return True
        if self.new_dispatch:
            self.new_ = False
            res = True
        if self.current.done:
            res = True
        if self.elapsed == self.slice:
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