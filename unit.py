class Mmunit:
    def __init__(self, height, width, depth, freq):
        self.height = height
        self.width = width
        self.depth = depth
        self.freq = freq
        self.processing = 0
    
    # matrix mxk, kxn
    def compute(self, m, k, n):
        fit_m = int(m/self.width)
        fit_k = int(k/self.height)
        fit_n = int(n/self.depth)

        inner = (self.height + self.width * 2 + self.depth) * fit_m * fit_k * fit_n
        if n - fit_n * self.depth == 0:
            outer = 0
        else:
            outer = (self.height + self.width * 2 + (n - fit_n * self.depth)) * (fit_m * fit_k)

        self.processing = inner + outer

    def process(self, m=None, k=None, n=None):
        if self.processing == 0:
            if all((m, k, n)):
                self.compute(m, k, n)
        elif self.processing > 0:
            self.processing -= 0
        
class Vecunit:
    def __init__
