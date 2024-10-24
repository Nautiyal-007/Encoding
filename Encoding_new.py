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
print("\n")
print("The new columns of the data are:")
print(data.columns)

s = (data.dtypes == 'object')
object_cols = list(s[s].index)
print("\n")
print("Categorical variables:")
print(object_cols)

df = pd.DataFrame(data)

def ordinal_encode(df, object_cols):
    """Encodes categorical data into ordinal values based on a provided order."""
    for column in object_cols:
        order = df[column].unique().tolist()  # Generate the unique values for each column
        ordinal_dict = {category: idx for idx, category in enumerate(order)}  # Create mapping
        df[column + '_ordinal'] = df[column].map(ordinal_dict)  # Map values
        df.drop(column, axis=1, inplace=True)  # Drop original column
    df.to_csv('ordinal_encode.csv', index=False)
    return df

def one_hot_encode(df, object_cols):
    """Encodes categorical data into one-hot encoded format."""
    for column in object_cols:
        one_hot = pd.get_dummies(df[column], prefix=column)  # Generate one-hot columns
        df = pd.concat([df, one_hot], axis=1)  # Concatenate with original DataFrame
        df.drop(column, axis=1, inplace=True)  # Drop original column
    df.to_csv('one_hot_encode.csv', index=False)
    return df

print("\n")
val = int(input("Enter 1 for ordinal encoding and 2 for one-hot encoding: "))

if val == 1:
    df = ordinal_encode(df, object_cols)
elif val == 2:
    df = one_hot_encode(df, object_cols)
else:
    print("\n")
    print("Invalid input")

print("\n")
print(df)
