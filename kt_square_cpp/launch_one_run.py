import re
import subprocess
import sys
import os
import json
argErrCode=2

if (len(sys.argv)!=2):
    print("wrong number of arguments")
    print("example: python launch_one_run.py /path/to/mc.conf")
    exit(argErrCode)

confFileName=str(sys.argv[1])
invalidValueErrCode=1
summaryErrCode=5
loadErrCode=3
confErrCode=4

#################################################
#parse conf, get jsonDataFromConf
confResult=subprocess.run(["python3", "./init_run_scripts/parseConf.py", confFileName], capture_output=True, text=True)
confJsonStr2stdout=confResult.stdout
# print(confJsonStr2stdout)

if confResult.returncode !=0:
    print("Error running parseConf.py with code "+str(confResult.returncode))
    # print(confResult.stderr)
    exit(confErrCode)
match_confJson=re.match(r"jsonDataFromConf=(.+)$",confJsonStr2stdout)
if match_confJson:
    jsonDataFromConf=json.loads(match_confJson.group(1))
else:
    print("jsonDataFromConf missing.")
#     exit(confErrCode)
# # print(jsonDataFromConf)

##################################################

###############################################
#load previous data, to get paths
#get loadedJsonData

loadResult=subprocess.run(["python3","./init_run_scripts/load_previous_data.py", json.dumps(jsonDataFromConf)],capture_output=True, text=True)
# print(loadResult.stdout)
if loadResult.returncode!=0:
    print("Error in loading with code "+str(loadResult.returncode))
    exit(loadErrCode)

match_loadJson=re.match(r"loadedJsonData=(.+)$",loadResult.stdout)
if match_loadJson:
    loadedJsonData=json.loads(match_loadJson.group(1))
else:
    print("loadedJsonData missing.")
    exit(loadErrCode)

# print(f"loadedJsonData={loadedJsonData}")
###############################################

###############################################
#construct parameters that are passed to mc
TStr=jsonDataFromConf["T"]
J11Str=jsonDataFromConf["J11"]
J12Str=jsonDataFromConf["J12"]
J21Str=jsonDataFromConf["J21"]
J22Str=jsonDataFromConf["J22"]
KStr=jsonDataFromConf["K"]


NStr=jsonDataFromConf["N"]

sweep_to_write=jsonDataFromConf["sweep_to_write"]
flushLastFile=loadedJsonData["flushLastFile"]
newFlushNum=jsonDataFromConf["default_flush_num"]

confFileName=jsonDataFromConf["confFileName"]
TDirRoot=os.path.dirname(confFileName)
TDirRoot=TDirRoot+"/"

#directory for raw data of U and s
U_s_dataDir=TDirRoot+"/U_s_dataFiles/"
sweep_multipleStr=jsonDataFromConf["sweep_multiple"]

num_parallel=jsonDataFromConf["num_parallel"]

params2cppInFile=[
    TStr+"\n",
    J11Str+"\n",
    J12Str+"\n",
    J21Str+"\n",
    J22Str+"\n",
    KStr+"\n",
    NStr+"\n",
    sweep_to_write+"\n",
    newFlushNum+"\n",
    flushLastFile+"\n",
    TDirRoot+"\n",
    U_s_dataDir+"\n",
    sweep_multipleStr+"\n",
    num_parallel+"\n"
    ]


cppInParamsFileName=TDirRoot+"/cppIn.txt"
with open(cppInParamsFileName,"w+") as fptr:
    fptr.writelines(params2cppInFile)