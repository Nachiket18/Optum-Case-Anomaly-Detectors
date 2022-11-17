import queue

def flatten(li: list):
    return [x for flat_li in li for x in flat_li]

##
##  sum = 0
##  for all nodes at the level
##      sum += sum(node.support_count)


### [[0, 69], [4, 10], [4, 23], [4, 34], [6, 10], [6, 21], [6, 34]]
### [{[0, 69]:4}, [4, 10,3], [4, 23,2], [4, 34], [6, 10], [6, 21], [6, 34]]

def getPairCombos(lst):
    result = []
    for i in lst[0].data:
        result.append(lst[1]+[i])
    return result

class LexicoNode:
    def __init__(self, data: list, children: dict, support_count: list):
        self.data = data
        self.children = children
        self.support_count = support_count
        
    def has_children(self):
        return len(self.children) > 0

    def add_node(self, edge, node):
        self.children[edge] = node

    def get_edges(self):
        return list(self.children.keys())

    def get_children(self):
        return list(self.children.values())

    def print_out(self, depth=0):
        def _indent(x):
            return "\t" * x

        print(_indent(depth), self.data, "SC: {}".format(self.support_count))

        if self.has_children():
            for i in self.children:
                print("\n", _indent(depth + 1), "---", i)
                self.children[i].print_out(depth + 1)

    def _trav_nodes(self, targ_col: int, level: int = 0):
        if level >= targ_col:
            return self.data
        elif level < targ_col:
            items = [x._trav_nodes(targ_col, level + 1) for x in self.children.values()]
            return flatten(items) #flattens list

    def nodes_at_level(self, col: int): #H_m - rule consequent
        return self._trav_nodes(col - 1)


    def _trav_freq(self, k: int, leading: list):

        dat = []
        #sc_list = []
        #s_c = self.calc_support_count()
        
        if k <= 1:
            return [leading + [x] for x in self.data]
            #leading.append(self.data)
            #return leading

        for x in range(len(self.children)):
            if self.data[x] == self.get_edges()[x]:
                #leading.append(self.get_edges()[x])
                dat.append(self.get_children()[x]._trav_freq(k - 1, leading + [self.get_edges()[x]]))
                #sc_list.append(s_c)

        return flatten(dat)

    def suppTravFreq(self, k: int, leading: list):
        dat =  [] #self.data
        sups = [] #self.support_count
        temp = []
        carr = []
        
        #print(edges)
        q = queue.Queue()
        q.put([self,[]])
        s = len(self.data)

        for n in range(s):
            sups.append(self.support_count[n])
            dat.append([]+[self.data[n]])

        count = 0
        numChildren = 1
        levelCount = 0
        
        while q.empty() != True:
            params = q.get()
            curr = params[0]
            weight = params[1]
            edges = curr.get_edges()
            pairs = getPairCombos(params)

            numChildren -= 1
            sups += curr.support_count
            if (curr != self):
                dat += pairs
            temp.append(["support Count: " + str(curr.support_count),curr.data,"edge: " + str(weight),"Pairs: ",pairs])
            for i in curr.get_children():
                q.put([i,weight + [edges[count]]])
                count += 1
            count = 0
            if (numChildren == 0):
                levelCount +=1
                print("Current Level: ",levelCount)
                print("Nodes on level: ",temp) 
                numChildren = q.qsize()
                temp.clear()
            q.task_done()
        out_t = (dat,sups)
        print("Type",type(out_t))
        return out_t
        
    def frequent_itemsets(self, k: int): #F_k -
        return self._trav_freq(k, [])

    def supp_frequent_itemsets(self, k: int,target=[None]): #F_k -
        
        data = self.suppTravFreq(k, [])
        itms,sups = data
        
        New_itms = []
        New_sups = []
        New_Data = []
        total = len(itms)
        if target[0] != None:
            for n in target:
                for i in range(total):
                    if n in itms[i] or len(itms[i]) == 1:
                        New_itms.append(itms[i])
                        New_sups.append(sups[i])
            New_Data = (New_itms,New_sups)
        data = [data,New_Data]
        return data

    def calc_support_count(self):
        return sum ( [x.support_count for x in self.get_children()] )
        ##
        ##  sum = 0
        ##  for all nodes at the level
        ##      sum += sum(node.support_count)
        


        pass
    
    def fetch_support_count_frequent_itemsets(self,val:int,dat:tuple):
        support_count_dict = {}
        X,Y = dat
        for j in range(0,len(X)):
            if len(X[j]) == val:
                for i in range(0,val):
                    if (X[j])[i] in support_count_dict:
                        support_count_dict[(X[j])[i]] += Y[j]
                    else:
                        support_count_dict[(X[j])[i]] =  Y[j]
        print("Support_Count",support_count_dict)
        return support_count_dict






if __name__ == "__main__":
    root = LexicoNode([0, 4, 6], {0: LexicoNode([69], {}, [2])}, [5,9,3])



    root.add_node(4, LexicoNode([10, 23, 34], {10: LexicoNode([50], {}, [5]),23: LexicoNode([70], {}, [1])}, [2,3,4]))
    root.add_node(6, LexicoNode([10, 21, 34], {10: LexicoNode([50], {}, [6])}, [5,4,2]))
    print(root.__dict__)

    print(root.print_out())


    m = 2

    print("H", m, root.nodes_at_level(m))

    print("FREQ")

    print("K", m, root.frequent_itemsets(m))

    print("K", 1, root.supp_frequent_itemsets(1))

    out = root.supp_frequent_itemsets(1)
    print(out[0])
    root.fetch_support_count_frequent_itemsets(1,out)
    # lex = LexicographicTree()

    # lex.addVertex(10)

    # print(lex.__dict__)
