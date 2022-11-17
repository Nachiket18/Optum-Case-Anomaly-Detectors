'''
Created on 17-Feb-2022

@author: Nachiket Deo
'''
import copy
import queue
from nbformat import current_nbformat
from lexicographic_tree import LexicoNode, flatten
from dataclasses import dataclass


class horizontalDataset:
    t_id = int()
    item = []
    
    def __init__(self, t_id, item):
        self.t_id = t_id
        self.item = item
         

@dataclass
class TreeNode:
    
    def __init__(self,item_id, item_count,children,interval_start = None,interval_end = None,parent=None):
        self.item_id = item_id
        self.item_count = item_count
        self.interval_start = interval_start
        self.interval_end = interval_end
        self.children = children
        

    def add_child(self, data):
        #assert isinstance(node, TreeNode)
        new_node = TreeNode(item_id=data,item_count=1,children = {})
        self.children[data] = new_node
        
    def incrementCount(self):
        self.item_count += 1        
        
    def findChild(self,data):
        return self.children[data]
    
        
    ##
    ## To locate a node with specified item_id in the Transaction Tree
    ##

    ##
    ## To locate a node with specified item_id in the Transaction Tree
    ##

    def searchNodeBFS(self,searchItem,searchChildren):
        
        output_interval_range = []
        output_searchItemRange = []
        q = queue.Queue()
        visited = []
        out_tree_node = None
        ##print(source.getId())
        q.put(self)

        visited.append(self.item_id)

        while( q.empty() == False):
            v = q.get()

            if v.item_id == searchItem:
                
                sub_q = queue.Queue()
                sub_q.put(v)

                sub_visited = []
                sub_visited.append(v.item_id)
                output_searchItemRange.append(v.interval_start)
                output_searchItemRange.append(v.interval_end)
                while (sub_q.empty() == False):
                    
                    b = sub_q.get()
                    for key,chd in b.children.items():
                        #if key not in sub_visited:
                        sub_q.put(chd)
                        sub_visited.append(key)
                        if key in searchChildren:
                            output_interval_range.append([chd.item_id,chd.interval_start,chd.interval_end])

                out_tree_node = v

            else:

                for key,nbr in v.children.items():
                    if key not in visited:
                        q.put(nbr)
                        visited.append(key)

        return output_interval_range, output_searchItemRange, out_tree_node

    def searchNodeBFS_freq_1_itemsets(self,searchItem):

        output_searchItemRange = []
        q = queue.Queue()
        visited = []
        q.put(self)
        visited.append(self.item_id)
        #print("Search Item", searchItem)
        while( q.empty() == False):
            v = q.get()

            if v.item_id == searchItem:
                output_searchItemRange.append(v.interval_start)
                output_searchItemRange.append(v.interval_end)
                break
            else:
                for key,nbr in v.children.items():
                    if key not in visited:
                        q.put(nbr)
                        visited.append(key)

        return output_searchItemRange

class transactionMapping:
    
    def __init__(self):
        pass

    def buildSubTree(self,node:TreeNode,value,i:int):
        #print("Node:",node.item_id,node.item_count,node.children.keys())

        if i <= (len(value) - 1):
            
            if value[i] in node.children.keys():
               node.children[value[i]].incrementCount()
               
               child = node.findChild(value[i])
               self.buildSubTree(child,value,i+1)
                 
            else:
                node.add_child(value[i])
                #print("Child-Added",node.item_id,node.children[value[i]].item_id)
                child = node.findChild(value[i])
                self.buildSubTree(child,value,i+1)
        else:
            return
        return

    def constructIntervalLists(self,node:TreeNode):

        queue = []  # Create a queue
        queue.append(node)

        while(len(queue) != 0):

            node_t = queue[0]
            queue.pop(0)
            i = 0
            
            
            for key, value in node_t.children.items():
                i +=1
                #print(value.item_id,value.item_count,node_t.item_count,node_t.item_id,node_t.interval_start)
                queue.append(value)
                if i == 1:
                    s_1 = node_t.interval_start        
                    e_1 = (s_1 + value.item_count) - 1
                    value.interval_start = s_1
                    value.interval_end = e_1
                    e_i_prime = e_1
                else:
                    s_i = e_i_prime + 1
                    e_i = (s_i + value.item_count) - 1
                    value.interval_start = s_i
                    value.interval_end = e_i
                    e_i_prime = e_i
                    
    
    def printSubTree(self,node:TreeNode):
        
        q = []  # Create a queue
        q.append(node)

        #print("In")
        while(len(q) != 0):
            
            n = len(q)
  
            # If this node has children
            while (n > 0):
         
                # Dequeue an item from queue and print it
                p = q[0]
                q.pop(0)
                print(p.item_id, p.item_count,p.interval_start,p.interval_end)
   
                # Enqueue all children of the dequeued item
                for key,value in p.children.items():
                    q.append(p.children[key])
                n -= 1
   
            #print()



    def createTransactionTree(self, dataset:list, length:int):



        freqent_itemset = {}
        ordered_frequent_itemset = {}

        
        
        ##
        ## Scan 1 
        ## Collecting the set of 1-frequent items F and their supports.
        ##
    
        for i in range(0,length):
            data = dataset[i].item
            for j in range(0,len(data)):
                if data[j] in freqent_itemset:
                    freqent_itemset[data[j]] +=1
                else:
                    freqent_itemset[data[j]] = 1
        # print(freqent_itemset)
        ##
        ## Sorting based on the values and delete the ones with frequency as  1
        ##

        ##print("fq",freqent_itemset)

        freqent_itemset_sorted = sorted(freqent_itemset.items(), key = lambda kv:(kv[1]), reverse = True)
        
        ##print(freqent_itemset_sorted)
        freqent_itemset_keys = {}
        i = 0
        for keys in freqent_itemset_sorted:
            if keys[1] == 1:
                freqent_itemset_sorted.pop(i)
            else:
                freqent_itemset_keys[keys[0]] = keys[1]
            i += 1        


        # freqent_itemset_keys = sorted(freqent_itemset_keys.items(), key = lambda kv:(kv[0]), reverse = False)
        # #print(freqent_itemset_keys)
        ##
        ## Generation of ordered_frequent_itemset for each transaction
        ##
        
        for i in range(0,length):
            data = dataset[i].item
            ordered_frequent_itemset[dataset[i].t_id] = []
            for key_frequent in freqent_itemset_keys:
                for keys in data:
                    if key_frequent == keys:
                        ordered_frequent_itemset[dataset[i].t_id].append(keys)

        #print(ordered_frequent_itemset)


        root = TreeNode(None,None,{},interval_start = 1)
        
        for key,value in ordered_frequent_itemset.items():
            #print(key,value)
            self.buildSubTree(root,value,0)
                
        ##
        ## Construction on Interval Lists
        ##

        self.constructIntervalLists(root)
        #self.printSubTree(root)
        return root,freqent_itemset_keys
    
    def findIntersection(self,intervals_1,interval_2):
        return max(0, min(intervals_1[1], interval_2[1]) - max(intervals_1[0], interval_2[0]) + 1)

    
    def depth_first_search_lexicographic_tree_build(self, lex_node: TreeNode, root: TreeNode):
        
        searchChildren = []
        lst = list(lex_node.data)
        #print("List of the Lexicographic Node",lst)

        for j in range(0,len(lst)-1):
        
            for i in range(j+1,len(lst)):
                searchChildren.append(lst[i])   

            out_range,output_searchItemRange,tree_node_current = root.searchNodeBFS(lst[j],searchChildren=searchChildren)
            #print(out_range,output_searchItemRange)

            if len(out_range) != 0:

                new_node_data = []

                total_support_count = []
                for node_data in out_range:   
                    support_count = self.findIntersection( [ node_data[1],node_data[2] ], output_searchItemRange )
                    #print(support_count)
                
                    if support_count >= 2:
                        new_node_data.append(node_data[0])
                        total_support_count.append(support_count)

                if new_node_data:
                    child = LexicoNode(new_node_data, {},total_support_count)     
                    lex_node.add_node(lst[j],child)
                    self.depth_first_search_lexicographic_tree_build(child,tree_node_current)

        return;       

    def constructLexicographicTree(self, root: TreeNode, freqent_itemset_keys: dict):

        
        total_support_count = []
        node_data = []

        for dt,support_c in freqent_itemset_keys.items():
            #print(dt,support_c)
            if support_c >= 2:
                total_support_count.append(support_c) 
                node_data.append(dt)
       
        #print("Root:",node_data,total_support_count)
        lex = LexicoNode(node_data, {}, total_support_count)
        self.depth_first_search_lexicographic_tree_build(lex, root)

        return lex

    def candidate_generation_pruning(self,m_val:int,dat:tuple):
        X = dat[0]
        Y = dat[1] 
        #X.append([2,1,4])
        #Y.append(2)
        #print(X)
        candidates_pruned = []

        if m_val == 1:
            for i in range(0,len(X)):
                if len(X[i]) == (m_val+1):    
                    candidates_pruned.append(X[i])
            return candidates_pruned

        for i in range(0,len(X)):
            if len(X[i]) == m_val:
                num_elements_threshold = 0
                for j in range(i,len(X)):
                    if len(X[j]) == (m_val + 1):
                        is_match = 0
                        for k in range(0,m_val):
                            if (X[i])[k] != (X[j])[k]:
                                is_match = 1
                                break
                        if is_match == 0:
                            num_elements_threshold += 1
                if num_elements_threshold > 1:
                    candidates_pruned.append(X[i])

            elif len(X[i]) == (m_val + 1):
                break
        
        #print("Candidates",candidates_pruned)
        return candidates_pruned

    def five_point_two (self, n:int, data, root_lexico:LexicoNode):
        output_rules = []
        for i in range(2, n):
            rules = self.ruleGeneration(i, 1, data, root_lexico)
            output_rules.append(rules)
        
        print(f"\n\nRULES: \n{output_rules}")

    def ruleGeneration(self, k_val:int, m_val:int, dat:tuple, lexico:LexicoNode):
        min_conf = 1.5
        f_k = set(flatten([f for f in dat[0] if len(f) == k_val]))
        f_k = list(f_k)
        ##print(dat[0])
        #print("f_k: ", f_k)
        #print(f"data: {dat}")
        rules = []

        if k_val > m_val + 1:

            candidates = self.candidate_generation_pruning(m_val, dat)
            sigma_f_k_all = lexico.fetch_support_count_frequent_itemsets(k_val, dat)
            sigma_f_k = sum(sigma_f_k_all.values())
            print("SIGMA_FK", sigma_f_k)
            for cnd in candidates:
                sigma_tmp = sum([v for k, v in sigma_f_k_all.items() if k not in cnd])


                print()
                print("CND,SIG-TMP", cnd, sigma_tmp)
                if sigma_tmp > 0:
                    conf = sigma_f_k / sigma_tmp
                    print("conf",conf)
                    if conf > min_conf:
                        rules.append(([i for i in f_k if i not in cnd],cnd))
                    else:
                        candidates.remove(cnd)

            rules.append(self.ruleGeneration(k_val, m_val+1, dat, lexico))
        print("\nRules",rules)
        return rules

def make_tree_from_data(data: list):
    dataset = [horizontalDataset(i, data[i]) for i in range(len(data))]
    print(len(dataset))
    tm = transactionMapping()
    root, freqent_itemset_keys = tm.createTransactionTree(dataset = dataset, length = len(data))
    # print(root)
    # print(freqent_itemset_keys)
    #print(len(freqent_itemset_keys))
    #breakpoint()
    root_lexico = tm.constructLexicographicTree(root, freqent_itemset_keys)

    return root_lexico

##
## This code generates 
##



## rules = [[(X_0,Y_0)],[(X,Y)],[(X_1,Y_1)]]

@dataclass
class Rule:
    antecedent: str
    precendent: str
    lift: float


def generate_top_10_rules(Q_itemsets:tuple,itemsets_normal:tuple, target, k:int):

    
    rules = []

    itemsets = Q_itemsets[0]
    itemsets_n = itemsets_normal[0]
    
    print("Target",target)
    

    for i in range(len(itemsets)):

        support_count__target = 0
        support_count_x = 0
        
        if type(itemsets[i]) == list and itemsets[i] != target:
            temp = [x for x in itemsets[i] if x not in target]
            trg = [x for x in itemsets[i] if x in target]

            #print("temp", temp, target, temp == target)
            
            for j in range(len(itemsets)):
                if itemsets[j] == trg:
                    support_count__target = Q_itemsets[1][j]

            for j in range(len(itemsets_n)):
                if itemsets_n[j] == temp:
                    support_count_x = itemsets_normal[1][j]
            
            if support_count_x == 0 or support_count__target == 0:
                continue

            #print("Sup-Data", support_count_x, support_count__target)

            lift = (Q_itemsets[1][i]) / ((support_count_x) * (support_count__target))        
            rule = Rule(trg, temp, lift)
            rules.append(rule)
    
    rules.sort(key=lambda x: x.lift, reverse=True)
    with open(r'C:\Users\Nachiket Deo\optum-competition-anomaly-detection\output\rules_3.txt', 'w') as fp:
        for item in rules:
        # write each item on a new line
            fp.write("%s\n" % item)
    print('Done')
   

    #([[2, 1], [2, 4], [2, 3], [2, 1, 3]], [2, 2, 2, 2])    


def main():
    t_1 = horizontalDataset(1,[2,1,5,3,19,20])
    t_2 = horizontalDataset(2,[2,6,3])
    t_3 = horizontalDataset(3,[1,7,8])
    t_4 = horizontalDataset(4,[3,1,9,10])
    t_5 = horizontalDataset(5,[2,1,11,3,17,18])
    t_6 = horizontalDataset(6,[2,4,12])
    t_7 = horizontalDataset(7,[1,13,14])
    t_8 = horizontalDataset(8,[2,15,4,16])

    dataset = []
    dataset.append(t_1)
    dataset.append(t_2)
    dataset.append(t_3)
    dataset.append(t_4)
    dataset.append(t_5)
    dataset.append(t_6)
    dataset.append(t_7)
    dataset.append(t_8)
    tm = transactionMapping()
    root, freqent_itemset_keys = tm.createTransactionTree(dataset = dataset, length = 8)

    
    ##print(freqent_itemset_keys)


    root_lexico = tm.constructLexicographicTree(root, freqent_itemset_keys)
    
    data = root_lexico.supp_frequent_itemsets(k=1)
    print(data)
    #tm.ruleGeneration(3,1,data,root_lexico)    
    #print("Full Function")
    tm.five_point_two(4, data, root_lexico)

if __name__ == '__main__': 
    main()