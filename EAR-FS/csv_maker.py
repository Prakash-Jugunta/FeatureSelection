import pandas as pd
data=pd.read_excel('data.xlsx')
print(data.columns)
data.to_csv('data03.csv')