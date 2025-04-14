import pickle
l=[1,2,3,4]
with open("savedata.pkl",'wb') as f:
    pickle.dump(l,f)
    