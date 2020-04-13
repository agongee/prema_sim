class Buffer:
    def __init__(self, size, bandwidth, latency, frequency):
        self.size = size
        self.bandwidth = bandwidth
        self.latency = latency
        self.frequency = frequency
        self.alloc_record = {}
        self.processing = 0

    def check_alloc(self, base, size=0):
        start = base
        end = base + size

        for i in self.alloc_record:
            start_alloc = i
            end_alloc = i + self.alloc_record[i][0]
            if start < start_alloc and start_alloc < end:
                return True
            elif start < end_alloc and end_alloc < end:
                return True

        return False

    def alloc(self, base, size, nnid):
        if self.check_alloc(base, size):
            self.alloc_record[base] = (size, nnid)
            return True
        else:
            return False

    def auto_alloc(self, size, nnid):
        for i in range(self.size - size):
            if self.alloc(i, size, nnid):
                return i
            
        return -1

    def load(self, size, nnid):
        
        

    def store(self, size, nnid):
        
        

    def process(self, op, size, nnid):
                