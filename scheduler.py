from layer_compiler.compiler import NN

class SchedAlgorithm:
    def __init__(self):
        self.queue = []
    
    def push_task(self, task: NN):
        if isinstance(task, list):
            self.queue.extend(task)
        else:
            self.queue.append(task)

    def schedule(self):
        



class SchedMechanism: