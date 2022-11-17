from lexicographic_tree import LexicoNode
import table_data as t_b
import anomaly_detection as a_d




if __name__ == "__main__":
    def _depth(node: LexicoNode):
        if not(node.has_children()):
            
            return 1
        
        else:
            return max([_depth(child) + 1 for child in node.get_children()])



    disc_data = t_b.csv_to_data(r"./dataset/adult_census.csv", [(2, 2), (4, 4), (6, 10), (14, 15)])

    undisc_data = t_b.csv_to_data(r"./dataset/adult_census.csv", [(1, 1), (11, 13)])

    disc_table = t_b.json_to_tables("./dataset/census_desc.json")

    newdisc_data = []
    for i in undisc_data:
        newdisc_data.append(t_b.discretize_data(i, disc_table))

    lexico = a_d.make_tree_from_data(newdisc_data)

    #lexico.print_out()
    #[print(f"{i}: {x.data}\n") for (i, x) in lexico.children.items()]
    
    #print(_depth(lexico))
    
    #breakpoint()
    #data_list = lexico.supp_frequent_itemsets(k=1,target=["CL_NoLoss"])
    #print("Data",data_list[0])
    #a_d.transactionMapping().five_point_two(4, data, lexico)
    #a_d.generate_top_10_rules(data_list[1], data_list[0], ["CL_NoLoss"], 5)