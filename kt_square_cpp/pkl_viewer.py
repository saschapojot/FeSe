import pickle
import numpy

inFileName="./dataAll/N6/row0/T1/U_s_dataFiles/s/s_init.pkl"

with open(inFileName,"rb") as fptr:
    data=pickle.load(fptr)

print(len(data))