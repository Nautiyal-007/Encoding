import pandas as pd

data = pd.read_csv(r'C:\Users\Prateek Nautiyal\Desktop\Prateek Nautiyal\AIMS\melb_data.csv')

print("The initial columns of the data were:")
print(data.columns)

cols_with_missing = [col for col in data.columns if data[col].isnull().any()] 
data.drop(cols_with_missing, axis=1, inplace=True)

low_cardinality_cols = [cname for cname in data.columns if data[cname].nunique() < 10 and data[cname].dtype == "object"]
numerical_cols = [cname for cname in data.columns if data[cname].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numerical_cols 
data = data[my_cols]
print("\nThe new columns of the data are:")
print(data.columns)

s = (data.dtypes == 'object')
object_cols = list(s[s].index)
print("\nCategorical variables:")
print(object_cols)

df = pd.DataFrame(data)

def ordinal_encode(df, object_cols):
    for column in object_cols:
        order = df[column].unique().tolist()  
        ordinal_dict = {category: idx for idx, category in enumerate(order)}
        df[column + '_ordinal'] = df[column].map(ordinal_dict)  
        df.drop(column, axis=1, inplace=True)  
    df.to_csv('ordinal_encode.csv', index=False)
    return df

def one_hot_encode(df, object_cols):
    for column in object_cols:
        one_hot = pd.get_dummies(df[column], prefix=column)  
        df = pd.concat([df, one_hot], axis=1)  
        df.drop(column, axis=1, inplace=True)  
    df.to_csv('one_hot_encode.csv', index=False)
    return df

val = int(input("\nEnter 1 for ordinal encoding and 2 for one-hot encoding: "))

if val == 1:
    df = ordinal_encode(df, object_cols)
elif val == 2:
    df = one_hot_encode(df, object_cols)
else:
    print("\nInvalid input")

print("\n")
print(df)
