import sys
import glob
import re
import json
from decimal import Decimal, getcontext
import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
import pickle
import os
import random
#this script loads previous data
# np.random.seed(29)
numArgErr=4
valErr=5

if (len(sys.argv)!=2):
    print("wrong number of arguments.")
    exit(numArgErr)


jsonDataFromConf =json.loads(sys.argv[1])

confFileName=jsonDataFromConf["confFileName"]
TDirRoot=os.path.dirname(confFileName)
TDirRoot=TDirRoot+"/"

# print(confFileName)

#create directory for raw data of U and s
U_s_dataDir=TDirRoot+"/U_s_dataFiles/"
NStr=jsonDataFromConf["N"]

N=int(NStr)
if N<=0:
    print("invalid N="+str(N))
    exit(valErr)

#search flushEnd
pklFileList=[]
flushEndAll=[]

#read U files
for file in glob.glob(U_s_dataDir+"/U/flushEnd*.pkl"):
    pklFileList.append(file)
    matchEnd=re.search(r"flushEnd(\d+)",file)
    if matchEnd:
        flushEndAll.append(int(matchEnd.group(1)))

flushLastFile=-1
def format_using_decimal(value, precision=4):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)


def create_init_s(U_s_dataDir):
    #s
    outPath_s=U_s_dataDir+"/s_angles/"
    Path(outPath_s).mkdir(exist_ok=True,parents=True)
    outFileName_s=outPath_s+"/s_angle_init.pkl"
    s_angle_init_mat=[] #theta, phi, theta, phi,...
    for n0 in range(0,N):
        for n1 in range(0,N):
            theta=random.uniform(0,2*np.pi)
            phi=random.uniform(0,np.pi)

            s_angle_init_mat.append(theta)
            s_angle_init_mat.append(phi)
    s_angle_init_mat=np.array(s_angle_init_mat)
    with open(outFileName_s,"wb") as fptr:
        pickle.dump(s_angle_init_mat,fptr)


def create_loadedJsonData(flushLastFileVal):

    initDataDict={

        "flushLastFile":str(flushLastFileVal)
    }
    # print(initDataDict)
    return json.dumps(initDataDict)


#if no data found, return flush=-1
if len(pklFileList)==0:
    create_init_s(U_s_dataDir)
    out_U_path=U_s_dataDir+"/U/"
    Path(out_U_path).mkdir(exist_ok=True,parents=True)
    loadedJsonDataStr=create_loadedJsonData(flushLastFile)
    loadedJsonData_stdout="loadedJsonData="+loadedJsonDataStr
    print(loadedJsonData_stdout)
    exit(0)

#if found pkl data with flushEndxxxx
sortedEndInds=np.argsort(flushEndAll)
sortedflushEnd=[flushEndAll[ind] for ind in sortedEndInds]
loadedJsonDataStr=create_loadedJsonData(sortedflushEnd[-1])
loadedJsonData_stdout="loadedJsonData="+loadedJsonDataStr
print(loadedJsonData_stdout)
exit(0)