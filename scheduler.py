from layer_compiler.compiler import NN

class Scheduler:
    def __init__(self, slice):
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
            return True
        
        degradation_current = self.current.remaining / self.current.estimated
        degradation_candidate = self.candidate.remaining / self.candidate.estimated
        
        if degradation_current > degradation_candidate:
            return False
        else:
            return True

    def dispatch(self):
        for i in self.queue:
            i.dispatch()

    def check_done(self):
        return all(self.queue)

    def sched_check(self):
        res = False
        if self.current == None:
            res = True
        if self.new_dispatch:
            self.new_dispatch = False
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