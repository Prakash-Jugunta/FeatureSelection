import pandas as pd
data=pd.read_excel('dataset02.xlsx')
print(data.columns)
data.to_csv('data.csv')