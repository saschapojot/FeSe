import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats
from scipy.special import ellipk
#This script loads avg U data, with confidence interval
# and plots U for all T

if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()

N=int(sys.argv[1])
row=sys.argv[2]
csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut/"
inCsvFile=csvDataFolderRoot+"/U_plot.csv"

df=pd.read_csv(inCsvFile)

TVec=np.array(df["T"])
UValsAll=np.array(df["U"])
interval_lowerValsAll=np.array(df["lower"])

interval_upperValsAll=np.array(df["upper"])

U_err_bar=UValsAll-interval_lowerValsAll
mask = (TVec > 0.01) & (TVec <18)

TInds = np.where(mask)[0]

print(f"TInds={TInds}")
TToPlt=TVec[TInds]
print(TToPlt)

#plt U
fig,ax=plt.subplots()

ax.errorbar(TToPlt,UValsAll[TInds],
            yerr=U_err_bar[TInds],fmt='o',color="black",
            ecolor='r', capsize=0.1,label='mc',
            markersize=3)

print(f"UValsAll[TInds]={UValsAll[TInds]}")
ax.set_xlabel('$T$')
ax.set_ylabel("U")
ax.set_title("U per unit cell, unit cell number="+str(N**2))
plt.legend(loc="best")
plt.savefig(csvDataFolderRoot+"/UPerUnitCell.png")
plt.close()
