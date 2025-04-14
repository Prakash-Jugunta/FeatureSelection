import pickle
with open("savedata.pkl","rb") as f:
    print(pickle.load(f))