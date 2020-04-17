class Test:
    def __init__(self, depend=None):
        self.done = False
        self.depend = []
        if depend != None:
            self.depend.extend(depend)

    def finish(self):
        t = True
        for i in self.depend:
            if not i.done:
                t = False

        if t:
            self.done = True

    def __str__(self):
        return str(self.done)

if __name__ == '__main__':
    a = Test()
    b = Test()
    c = Test(depend=[a, b])
    d = Test(depend=[c])

    ll = []
    ll.append(a)
    ll.append(b)
    ll.append(c)
    ll.append(d)

    ll[0].finish()
    ll[1].finish()
    ll[2].finish()
    ll[3].finish()

    for i in ll:
        print(i)