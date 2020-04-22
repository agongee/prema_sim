class Context:
    def __init__(self, nnid):
        self.nnid = nnid
        self.context = []

    def push_context(self, addr, size):
        self.context.append((addr, size, True))

    def del_context(self, addr):
        index = -1
        for i in range(len(self.context)):
            if self.context[i][0] == addr:
                index = i
                break
        if index != -1:
            self.context.pop(index)
            return True
        else:
            return False

    def flush(self):
        for i in self.context:
            i[2] = False

    def context_size(self):
        size = 0
        for i in self.context:
            size += i[1]
        return size

    def context_out(self):
        self.context = []

    def recover(self):
        for i in self.context:
            i[2] = True

    def __iter__(self):
        return self.context
        

class Buffer:
    def __init__(self, size, bandwidth, latency, name):
        self.size = size
        self.bandwidth = bandwidth
        self.latency = latency
        self.alloc_record = {}
        self.processing = 0
        self.to_load = None
        #self.to_store = None
        self.to_store = []
        self.context_list = {}
        self.name = name

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

    def alloc(self, base, size, nnid, flush=False):
        if self.check_alloc(base, size):
            self.alloc_record[base] = (size, nnid)
            return True
        else:
            return False

    def auto_alloc(self, size, nnid, flush=False):
        # there is enough empty space
        for i in range(self.size - size):
            if self.alloc(i, size, nnid, flush):
                return i

        # there isn't enough empty space
        success = -1
        while success > -1:
            if len(self.alloc_record) == 0:
                break
            self.alloc_record.pop(0)
            for i in range(self.size - size):
                if self.alloc(i, size, nnid, flush):
                    success = i
                    break

        return success

    def force_alloc(self, addr, size, nnid, flush=False):
        start = addr
        end = addr + size

        for i in self.alloc_record:
            start_alloc = i
            end_alloc = i + self.alloc_record[i][0]
            if start < start_alloc and start_alloc < end:
                del self.alloc_record[i]
            elif start < end_alloc and end_alloc < end:
                del self.alloc_record[i]

        self.alloc(addr, size, nnid, flush)
            

    def load(self, size, nnid):
        self.to_load = self.auto_alloc(size, nnid)
        self.processing = int(size/self.bandwidth) + self.latency
        if nnid not in self.context_list:
            print(f"{self.name}: Context of {nnid} created at load")
            self.context_list[nnid] = Context(nnid)

    '''
    def store(self, addr, size, nnid, done=True):
        if self.alloc_record[addr][0] == size and self.alloc_record[addr][1] == nnid:
            self.to_store = (addr, done)
            self.processing = int(size/self.bandwidth) + self.latency
        else:
            print("Wrong Store Request")
    '''
    def store(self, nnid):
        size = self.context_list[nnid].context_size()
        self.processing = int(size/self.bandwidth) + self.latency
        for i in self.context_list[nnid].context:
            self.to_store.append(i[0])

    def store_fake(self, nnid):
        self.context_list[nnid].context_out()

    def save(self, size, nnid):
        addr = self.auto_alloc(size, nnid, True)
        if nnid not in self.context_list:
            print(f"{self.name}: Context of {nnid} created at save")
            self.context_list[nnid] = Context(nnid)
        self.context_list[nnid].push_context(addr, size)

    def process(self, op=None, size=None, nnid=None, done=True):
        if self.processing == 0:
            if all((op, size, nnid)):
                if op == 'LOAD':
                    self.load(size, nnid)
                elif op == 'STORE':
                    #self.store(size, nnid, done)
                    self.store(nnid)
        elif self.processing == 1:
            if self.to_load != None:
                return self.to_load
            
            elif len(self.to_store) != 0:
                self.context_list[nnid].context_out()
                while len(self.to_store) != 0:
                    addr = self.to_store.pop()
                    del self.alloc_record[addr]
            '''
            elif self.to_store != None:
                self.context_list[nnid].del_context(self.to_store[0])     
                if self.to_store[1]:
                    del self.alloc_record[self.to_store[0]]
            '''
            self.to_load = None
            self.to_store = None
            self.processing = 0
        elif self.processing > 0:
            self.processing -= 1
        else:
            self.processing = 0

    def checkout(self, nnid):
        delay = 0
        if nnid not in self.context_list:
            return delay
        for i in self.context_list[nnid].context:
            delay += int(i[1]/self.bandwidth) + self.latency
            del self.alloc_record[i[0]]

        self.context_list[nnid].flush()
        return delay

    def recover(self, nnid):
        delay = 0
        for i in self.context_list[nnid]:
            delay += int(i[1]/self.bandwidth) + self.latency
            self.force_alloc(i[0], i[1], nnid, True)

        self.context_list[nnid].recover()

        return delay
        
                