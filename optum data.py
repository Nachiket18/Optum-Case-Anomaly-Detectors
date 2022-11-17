# %%
import pandas as pd
from anomaly_detection import make_tree_from_data,generate_top_10_rules
from lexicographic_tree import LexicoNode

# %%
df = pd.read_csv(r'C:\Users\Nachiket Deo\optum-competition-anomaly-detection\dataset\chr_analytic_data2022.csv',header=[0,1])
f = open(r'C:\Users\Nachiket Deo\optum-competition-anomaly-detection\dataset\chr_analytic_data2022.csv')
lines = f.readlines()

# %%
# for n in range(20):
#     arr = lines[n].split(',')
#     print(arr)
#     print(len(arr))

# %%
df.columns = df.columns.map('_'.join)


# %%
df.head()


# %%
df['% rural CI low_v058_cilow'].isnull().count()


# %%
df['% rural CI high_v058_cihigh'].isnull().count()


# %%
df['% female CI low_v057_cilow'].isnull().count()

# %%
df['% female CI high_v057_cihigh'].isnull().count()

# %%
# count = 0
# for n in df:
#     if (df[n].isnull()) == 3194:
#         count += 1
# print(count)
drop_df = df
count  = 0
features = []
for n in df:
    if (df[n].isnull().sum()) >= (len(df)*(.5)):
        features.append(n)
        print(n,df[n].isnull().sum())
        count += 1
drop_df = df.drop(features,axis=1)
#print(features)
#print(count)

# %%


# %%
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

# %%

drop_df_prep = pd.DataFrame(drop_df_two, columns = num_features)
drop_full = drop_df
for f in drop_df_prep:
    if f in drop_full.columns:
        print(f)
        drop_full[f] = drop_df_prep[f]

# %%
count = 0
for m in drop_full:
    if drop_full[m].isnull().sum() != 0:
        count += 1


# %%
def check_prev(col_n):
    arr = col_n.split()
    ident = arr[-1].split('_')
    return (ident[0],ident[1])

# %%

prev = ""
count = 0
col_name = ""

z_score_df = drop_full
features = []
for j in drop_full:
#     print(j)
    if prev == "":
        prev = j
    else:
        prev_data = check_prev(prev)
        curr_data = check_prev(j)
#         print(prev_data)
#         print(curr_data)
        if prev_data[0] == 'numerator' and curr_data[0] == 'denominator':
            for f in j.split()[:-1]:
                col_name += f + "_"
            if prev_data[1] == curr_data[1]:
                features.append(prev)
                features.append(j)
                index = z_score_df.columns.get_loc(j)
                temp = drop_full[prev]/drop_full[j]
                col_name += prev_data[1] + "_z_score"
                z_score_df.insert(index+1, col_name, temp, True)
                count += 1
        if curr_data[0] == 'denominator' or curr_data[0] == 'numerator':
            if j not in features:
                features.append(j)
        col_name = ""
        prev = j


# %%
z_score_df = z_score_df.drop(features,axis=1)
for i in z_score_df.columns:
    print(i)

# %%
z_score_df

# %%
# variable = "Premature death raw value_v001_rawvalue"
# x = z_score_df[variable].values
# y = z_score_df['Preventable hospital stays raw value_v005_rawvalue']

# %%
# !pip install optbinning

# %%
# optb.fit(x,y)

# %%
# binning_table = optb.binning_table.build()
# binning_table

# %%
import re
from sklearn.preprocessing import KBinsDiscretizer

def make_bins(col_name,bin_target,num_bins):
    binned_data = pd.cut(bin_target, num_bins)
    return binned_data

# %%
def in_bin(val,lower,upper):
    if val >= lower and val <= upper:
        return True
    else:
        return False

# %%
variable = "Premature death raw value_v001_rawvalue"
x = z_score_df[variable].values
y = z_score_df['Preventable hospital stays raw value_v005_rawvalue']
make_bins(variable,x,4)

# %%
drp_feats = ['State FIPS Code_statecode','County FIPS Code_countycode','5-digit FIPS Code_fipscode','Name_county','Release Year_year','County Ranked (Yes=1/No=0)_county_ranked']
fnl_drop_df = z_score_df.drop(drp_feats,axis=1)

# %%
fnl_drop_df

# %%
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
                    


# %%
fnl_cats_frame

# %%
pre_algo = fnl_cats_frame.to_numpy()

# %%
len(pre_algo)

# %%
pre_algo_list = pre_algo.tolist()
pre_algo_list

# %%
lexico_tree = make_tree_from_data(pre_algo_list)
lexico_tree.print_out()
 
data_list = lexico_tree.supp_frequent_itemsets(k=1,target=["72_(3839.5, 5569.75]",      
"72_(2109.25, 3839.5]",     
"72_(5569.75, 7300.0]",      
"72_(365.158, 2109.25]",     
"72_(7300.0, 9030.25]",       
"72_(9030.25, 10760.5]",     
"72_(10760.5, 12490.75]",      
"72_(12490.75, 14221.0]"])

generate_top_10_rules(data_list[1], data_list[0],["72_(3839.5, 5569.75]",      
"72_(2109.25, 3839.5]",     
"72_(5569.75, 7300.0]",      
"72_(365.158, 2109.25]",     
"72_(7300.0, 9030.25]",       
"72_(9030.25, 10760.5]",     
"72_(10760.5, 12490.75]",      
"72_(12490.75, 14221.0]"], 5)

# %%
count = 0
for i in fnl_cats_frame:
    if count in [369,327,380,381]:
        print(i)
    count += 1

# %%



