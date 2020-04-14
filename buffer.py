class Buffer:
    def __init__(self, size, bandwidth, latency, frequency):
        self.size = size
        self.bandwidth = bandwidth
        self.latency = latency
        self.frequency = frequency
        self.alloc_record = {}
        self.processing = 0
        self.to_load = None
        self.to_store = None

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
        # there is enough empty space
        for i in range(self.size - size):
            if self.alloc(i, size, nnid):
                return i

        # there isn't enough empty space
        success = -1
        while success > -1:
            if len(self.alloc_record) == 0:
                break;
            self.alloc_record.pop(0)
            for i in range(self.size - size):
                if self.alloc(i, size, nnid):
                    success = i
                    break;

        return success

    def load(self, size, nnid):
        self.to_load = self.auto_alloc(size, nnid)
        self.processing = int(size/self.bandwidth) + self.latency

    def store(self, addr, size, nnid, done=True):
        if self.alloc_record[addr][0] == size and self.alloc_record[addr][1] == nnid:
            self.to_store = (addr, done)
            self.processing = int(size/self.bandwidth) + self.latency
        else:
            print("Wrong Store Request")

    def process(self, op=None, size=None, nnid=None, done=True):
        if self.processing == 0:
            if all((op, size, nnid)):
                if op == 'LOAD':
                    self.load(size, nnid)
                elif op == 'STORE':
                    self.size(size, nnid, done)
        elif self.processing == 1:
            if to_load != None:
                return self.to_load
            elif to_store != None:
                if self.to_store[1]:
                    del self.alloc_record[self.to_store[1]]         
            self.to_load = None
            self.to_store = None
            self.processing = 0
        elif self.processing > 0:
            self.processing -= 1
        else:
            self.processing = 0
                