from layer_compiler.enum_def import Buf, Op

# CONTEXT that ONLY CONSIDER SIZE (NOT ADDRESS)
class SimpleContext:
    def __init__(self, nnid, buf):
        self.nnid = nnid
        self.buf = buf
        self.size = 0
        self.to_recover = 0

    def push_context(self, size):
        self.size += size

    def flush(self, size=-1):
        if size < 0:
            self.size = 0
        else:
            self.size = max(self.size-size, 0)
    
    def checkout_context(self, size=-1):
        if size < 0:
            self.to_recover = self.size
            self.size = 0
        else:
            self.to_recover = min(self.size, size)
            self.size = max(self.size-size, 0)

        return self.to_recover

    def recover_context(self):
        temp = self.to_recover
        self.size += self.to_recover
        self.to_recover = 0

        return temp
        
        

# BUFFER that ASSUMES ENOUGH SPACE
class SimpleBuffer:
    def  __init__(self, size, bandwidth, latency, name):
        self.size = int(size)
        self.bandwidth = int(bandwidth)
        self.latency = int(latency)
        self.processing = 0
        self.name = name
        self.context_list = {}

        print(f"  Buffer {self.name} Initialization")
        print(f"- Size:      {self.size}")
        print(f"- Bandwidth: {self.bandwidth}")
        print(f"- Latency:   {self.latency}")

    # LOAD_TILE
    def load(self, nnid, size):
        if nnid not in self.context_list:
            print(f"{self.name}: Context of {nnid} created at load")
            self.context_list[nnid] = SimpleContext(nnid, self.name)
        self.processing = int(size/self.bandwidth) + self.latency

    # STORE_TILE
    def store(self, nnid, size=-1):
        size = self.context_list[nnid].size
        self.processing = int(size/self.bandwidth) + self.latency
        self.context_list[nnid].flush()

    # STORE_FAKE
    def save(self, size, nnid):
        if nnid not in self.context_list:
            print(f"{self.name}: Context of {nnid} created at load")
            self.context_list[nnid] = SimpleContext(nnid, self.name)
        self.context_list[nnid].push_context(size)

    def checkout(self, nnid):
        size = self.context_list[nnid].checkout_context()
        return int(size/self.bandwidth) + self.latency
            
    def recover(self, nnid):
        if nnid not in self.context_list:
            return 0
        size = self.context_list[nnid].recover_context()
        return int(size/self.bandwidth) + self.latency

    def store_fake(self, nnid):
        self.context_list[nnid].flush()

    def process(self, op=None, size=None, nnid=None, done=True):
        if self.processing == 0:
            if all((op, nnid)):
                if op == Op.LOAD_TILE and size != None:
                    self.load(nnid, size)
                elif op == Op.STORE_TILE:
                    self.store(nnid)
        elif self.processing > 0:
            self.processing -= 1
        else:
            self.processing = 0

    def context_status(self, nnid):
        print(f" Context Status of NNID [{nnid}] at {self.name}: {self.context_list[nnid].size}")