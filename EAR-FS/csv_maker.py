import pandas as pd
data=pd.read_excel('hqa_data.xlsx')
print(data.columns)
data.to_csv('hqa_data.csv')