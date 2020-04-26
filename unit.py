class Mmunit:
    def __init__(self, height, width, depth):
        self.height = height
        self.width = width
        self.depth = depth
        self.processing = 0
    
    # matrix mxk, kxn
    def compute(self, m, k, n):
        self.processing = m + k + 2 * n

    def process(self, m=None, k=None, n=None):
        if self.processing == 0:
            if all((m, k, n)):
                self.compute(m, k, n)
            else:
                pass
        elif self.processing > 0:
            self.processing -= 1
        else:
            self.processing = 0
        
class Vecunit:
    def __init__(self, size):
        self.size = size
        self.processing = 0

    def compute(self, size):
        self.processing = int(size/self.size) + 1

    def process(self, size=None):
        if self.processing == 0:
            if size != None:
                self.compute(size)
        elif self.processing > 0:
            self.processing -= 1
        else:
            self.processing = 0