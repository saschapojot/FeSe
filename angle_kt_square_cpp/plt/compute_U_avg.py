import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats

#This script loads csv data of U, with confidence interval,
# and computes mean value for each T
if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()

N=int(sys.argv[1])
row=sys.argv[2]
csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut/"
TVals=[]
TFileNames=[]

for TFile in glob.glob(csvDataFolderRoot+"/T*"):

    matchT=re.search(r"T([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)",TFile)
    # if float(matchT.group(1))<1:
    #     continue

    if matchT:
        TFileNames.append(TFile)
        TVals.append(float(matchT.group(1)))


sortedInds=np.argsort(TVals)
sortedTVals=[TVals[ind] for ind in sortedInds]
sortedTFiles=[TFileNames[ind] for ind in sortedInds]

def U_estimator(UVec):
    """

    :param UVec:
    :return:
    """

    EU=np.mean(UVec)
    return EU/N**2


def U_jackknife(UVec):
    n=len(UVec)
    jackknife_samples = np.zeros(n)
    for i in range(0,n):
        sampleU=np.delete(UVec,i)
        jackknife_samples[i]=U_estimator(sampleU)

    # Jackknife estimate of the statistic
    jackknife_estimate = np.mean(jackknife_samples)
    variance_estimate = (n - 1) / n * np.sum((jackknife_samples - jackknife_estimate) ** 2)
    return jackknife_estimate, variance_estimate


def U_confidence_interval(UVec,confidence_level=0.95):
    """

    :param UVec:
    :param confidence_level:
    :return:
    """
    jackknife_estimate, jackknife_variance=U_jackknife(UVec)
    n=len(UVec)
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha / 2, df=n-1)
    # Calculate the standard error
    standard_error = np.sqrt(jackknife_variance)
    # Calculate the confidence interval
    ci_lower = jackknife_estimate - t_critical * standard_error
    ci_upper = jackknife_estimate + t_critical * standard_error
    return jackknife_estimate,ci_lower, ci_upper


def generate_one_U_point(oneTFile):
    """

    :param oneTFile: corresponds to one temperature
    :return:
    """
    matchT=re.search(r'T([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)',oneTFile)
    TVal=float(matchT.group(1))
    U_distPath=oneTFile+"/U.csv"
    df=pd.read_csv(U_distPath,header=None)
    UVec=np.array(df.iloc[:,0])
    print("T="+str(TVal)+", data num="+str(len(UVec)))
    jackknife_estimate,ci_lower, ci_upper=U_confidence_interval(UVec)
    return [jackknife_estimate,ci_lower, ci_upper]

UValsAll=[]
interval_lowerValsAll=[]
interval_upperValsAll=[]
tStart=datetime.now()
for k in range(0,len(sortedTFiles)):
    oneTFile=sortedTFiles[k]
    jackknife_estimate,ci_lower, ci_upper=generate_one_U_point(oneTFile)
    UValsAll.append(jackknife_estimate)
    interval_lowerValsAll.append(ci_lower)
    interval_upperValsAll.append(ci_upper)

csv_file_name=csvDataFolderRoot+"U_plot.csv"
df=pd.DataFrame({
    "T":sortedTVals,
    "U":np.array(UValsAll),
    "lower":interval_lowerValsAll,
    "upper":interval_upperValsAll
}
)
df.to_csv(csv_file_name,index=False)
tEnd=datetime.now()

print(f"time: {tEnd-tStart}")