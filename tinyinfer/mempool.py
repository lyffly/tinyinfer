class MemBlock:
    def __init__(self, name: str, start:int, size:int):
        self.start = start
        self.size = size
        self.is_free = False
        self.name = name

    def __repr__(self):
        return str(self)

    def __str__(self):
        p_str = "*{0},s={1},e={2}".format(self.name, self.start, self.start+self.size)
        if self.is_free:
            p_str = "*FREE*,s={0},e={1}".format(self.start, self.start+self.size)
        return p_str

class MemPool:
    def __init__(self):
        self.size = 0
        self.block_list = []
        self.names = []
        self.edge_uses = {}
        self.edge_mem_start = {}

    def __repr__(self):
        return "[pool] blocks = " + str(self.block_list)

    def __str__(self):
        return "[pool] blocks = " + str(self.block_list)

    def align_size(self, size : int):
        ALIGN = 1024
        if size%ALIGN == 0:
            return size
        return (size//ALIGN + 1) * ALIGN

    def find_free_buffer(self, name: str, size: int):
        index = -1
        for i, block in enumerate(self.block_list):
            if block.is_free and block.size >= size:
                index = i
                #print("find index free block : ", index)
                break
        if index == -1:
            #print("find *NO* free block ")
            return index
        # [ Free ] -> [ Used ] 
        # [ Free      ] -> [ Used ] + [ Free ]
        self.block_list[index].name = name
        self.names.append(name)
        extra_size = self.block_list[index].size - size
        self.block_list[index].size = size
        self.block_list[index].is_free = False
        if not (name in self.edge_mem_start.keys()):
            self.edge_mem_start[name] = self.block_list[index].start
        if extra_size > 0:
            extra_block = MemBlock("", self.block_list[index].size + self.block_list[index].start, extra_size)
            extra_block.is_free = True
            self.block_list.insert(index+1, extra_block)
        return index

    def malloc(self, name, size):
        for ename in self.names:
            if self.edge_uses[ename] <= 0:
                for block in self.block_list:
                    if block.name == ename:
                        block.is_free = True
        self.check_free()
        aligned_size = self.align_size(size)
        found_free = self.find_free_buffer(name, aligned_size)
        if found_free == -1 :
            block = MemBlock(name, self.size, aligned_size)
            self.block_list.append(block)
            self.names.append(name)
            if not (name in self.edge_mem_start.keys()):
                self.edge_mem_start[name] = self.size
            self.size += aligned_size

    def check_free(self):
        for name in self.names:
            if (name in self.edge_uses.keys()) and (self.edge_uses[name] <= 0):
                for bi, block in enumerate(self.block_list):
                    if block.name == name:
                        block.is_free = True
        for bi, block in enumerate(self.block_list):
            if bi+1 < len(self.block_list) and self.block_list[bi + 1].is_free and self.block_list[bi].is_free :
                self.block_list[bi].size += self.block_list[bi+1].size
                self.block_list[bi].name = ""
                del self.block_list[bi+1]
            if bi-1 >= 0 and self.block_list[bi - 1].is_free and self.block_list[bi].is_free :
                self.block_list[bi].size += self.block_list[bi-1].size
                self.block_list[bi].start = self.block_list[bi-1].start
                self.block_list[bi].name = ""
                del self.block_list[bi-1]


if __name__ == "__main__":
    nodes = {"conv1": {
                "in_names":["Eg1", "work1"],
                "out_names":["Eg2"],
            },
        "relu1": {
                "in_names":["Eg2"],
                "out_names":["Eg3"],
            },
        "pool1": {
            "in_names":["Eg3"],
            "out_names":["Eg4"],
        },
        "conv2": {
            "in_names":["Eg4", "work2"],
            "out_names":["Eg5"],
        },
        "relu2": {
            "in_names":["Eg5"],
            "out_names":["Eg6"],
        },
        "conv3": {
            "in_names":["Eg6", "work3"],
            "out_names":["Eg7"],
        },
        "add3": {
            "in_names":["Eg4", "Eg7"],
            "out_names":["Eg8"],
        },
        "relu3": {
            "in_names":["Eg8"],
            "out_names":["Eg9"],
        },}

    edges_sizes = {"Eg1": 1*3*224*224,
                "Eg2": 1*64*112*112,
                "Eg3": 1*64*112*112,
                "Eg4": 1*64*56*56,
                "Eg5": 1*64*56*56,
                "Eg6": 1*64*56*56,
                "Eg7": 1*64*56*56,
                "Eg8": 1*64*56*56,
                "Eg9": 1*64*56*56,
                "work1":1200,
                "work2":200,
                "work3":4000}


    print(nodes)
    print(edges_sizes)
    pool = MemPool()

    # use count
    pool.edge_uses = {}
    for node_name, value in nodes.items():
        for name in value["in_names"]:
            if name not in pool.edge_uses.keys():
                pool.edge_uses[name] = 0
            pool.edge_uses[name] += 1

    for node_name, value in nodes.items():
        print("\nnode name : ", node_name)
        sum = 0
        print("before malloc : ", str(pool))

        for name in value["in_names"]:
            if name in pool.names:
                continue
            sum += edges_sizes[name]
            pool.malloc(name, edges_sizes[name])
        for name in value["out_names"]:
            if name in pool.names:
                continue
            sum += edges_sizes[name]
            pool.malloc(name, edges_sizes[name])
        print("after malloc : ", pool)
        
        for name in value["in_names"]:
            pool.edge_uses[name] -= 1
        pool.check_free()
        print("after set block free :", pool)

    print("\nall edge memory start : ", pool.edge_mem_start)
    print("memory pool size : ", pool.size)