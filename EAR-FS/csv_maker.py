import pandas as pd
data=pd.read_excel('DCT_orginal_Telugu_test.xlsx')
print(data.columns)
data.to_csv('DCT_orginal_Telugu_test.csv')