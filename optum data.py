# %%
import pandas as pd
from anomaly_detection import make_tree_from_data,generate_top_10_rules
from lexicographic_tree import LexicoNode
import re
from sklearn.preprocessing import KBinsDiscretizer


df = pd.read_csv(r'C:\Users\Nachiket Deo\Optum-Case-Anomaly-Detectors\dataset\chr_analytic_data2022.csv',header=[0,1])
f = open(r'C:\Users\Nachiket Deo\Optum-Case-Anomaly-Detectors\dataset\chr_analytic_data2022.csv')
lines = f.readlines()


df.columns = df.columns.map('_'.join)

df.head()

drop_df = df
count  = 0
features = []
for n in df:
    if (df[n].isnull().sum()) >= (len(df)*(.5)):
        features.append(n)
        print(n,df[n].isnull().sum())
        count += 1
drop_df = df.drop(features,axis=1)


from sklearn.impute import KNNImputer
cat_features = []
num_features = []
for v in drop_df:
    if drop_df[v].dtypes == object:
        cat_features.append(v)
drop_df_two = drop_df
for i in cat_features:
    drop_df_two = drop_df_two.loc[:, drop_df_two.columns != i]
num_features = drop_df_two.columns
imputer = KNNImputer(n_neighbors=2)
drop_df_two = imputer.fit_transform(drop_df_two)
print(num_features)


drop_df_prep = pd.DataFrame(drop_df_two, columns = num_features)
drop_full = drop_df
for f in drop_df_prep:
    if f in drop_full.columns:
        print(f)
        drop_full[f] = drop_df_prep[f]


count = 0
for m in drop_full:
    if drop_full[m].isnull().sum() != 0:
        count += 1


def check_prev(col_n):
    arr = col_n.split()
    ident = arr[-1].split('_')
    return (ident[0],ident[1])


prev = ""
count = 0
col_name = ""

z_score_df = drop_full
features = []
for j in drop_full:
    if prev == "":
        prev = j
    else:
        prev_data = check_prev(prev)
        curr_data = check_prev(j)
        if prev_data[0] == 'numerator' and curr_data[0] == 'denominator':
            for f in j.split()[:-1]:
                col_name += f + "_"
            if prev_data[1] == curr_data[1]:
                features.append(prev)
                features.append(j)
                index = z_score_df.columns.get_loc(j)
                temp = drop_full[prev]/drop_full[j]
                col_name += prev_data[1] + "_z_score"
                #z_score_df.insert(index+1, col_name, temp, True)
                count += 1
        if curr_data[0] == 'denominator' or curr_data[0] == 'numerator':
            if j not in features:
                features.append(j)
        col_name = ""
        prev = j



z_score_df = z_score_df.drop(features,axis=1)
for i in z_score_df.columns:
    print(i)

z_score_df



def make_bins(col_name,bin_target,num_bins):
    binned_data = pd.cut(bin_target, num_bins)
    return binned_data

def in_bin(val,lower,upper):
    if val >= lower and val <= upper:
        return True
    else:
        return False

variable = "Premature death raw value_v001_rawvalue"
x = z_score_df[variable].values
y = z_score_df['Preventable hospital stays raw value_v005_rawvalue']
make_bins(variable,x,4)


drp_feats = ['State FIPS Code_statecode','County FIPS Code_countycode','5-digit FIPS Code_fipscode','Name_county','Release Year_year','County Ranked (Yes=1/No=0)_county_ranked']
fnl_drop_df = z_score_df.drop(drp_feats,axis=1)


count = 0
start = 32
end = 41
temp_df = pd.DataFrame()
for n in fnl_drop_df:
  if count == end:
    temp_df['Preventable hospital stays raw value_v005_rawvalue'] = fnl_drop_df['Preventable hospital stays raw value_v005_rawvalue']
    break
  if n == 'Preventable hospital stays raw value_v005_rawvalue':
    continue
  if count >= start and count <= end-1:
    temp_df[n] = fnl_drop_df[n]
  count += 1

fnl_drop_df = temp_df
fnl_drop_df


target = 'Preventable hospital stays raw value_v005_rawvalue'
cats_frame = pd.DataFrame()
fnl_cats_frame = pd.DataFrame()
all_bins = []
count = 0
for n in fnl_drop_df:
    if fnl_drop_df[n].dtypes != object:
        arr = make_bins(n,fnl_drop_df[n].values,8)
        cats_frame[n] = arr
    else:
        cats_frame[n] = fnl_drop_df[n]
for l in cats_frame:
    if l != "State Abbreviation_state":
        new_arr = []
        for x in cats_frame[l]:
            new_arr.append(str(count) + "_" + str(x))
        fnl_cats_frame[l] = new_arr
    else:
        fnl_cats_frame[l] = cats_frame[l]
    count += 1
                    



fnl_cats_frame


pre_algo = fnl_cats_frame.to_numpy()


len(pre_algo)


rules = []
pre_algo_list = pre_algo.tolist()
for j in fnl_cats_frame:
    if j == "Preventable hospital stays raw value_v005_rawvalue":
        rules = fnl_cats_frame[j].value_counts().index.tolist()
print(rules)

print("J",j)


lexico_tree = make_tree_from_data(pre_algo_list)
data_list = lexico_tree.supp_frequent_itemsets(k=1,target=rules)

generate_top_10_rules(data_list[1], data_list[0],rules,5)


