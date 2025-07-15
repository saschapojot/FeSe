from pathlib import Path
from decimal import Decimal, getcontext

import numpy as np
import pandas as pd


def format_using_decimal(value, precision=4):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)



N=24#unit cell number
N0=N
N1=N
which_row=0#use which_row in param.csv

TVals=[0.1,0.5,1]
default_flush_num=10

num_parallel=24
print(f"num_parallel={num_parallel}")
print(f"default_flush_num={default_flush_num}")
dataRoot="./dataAll/"
# effective_data_num_required=1000
sweep_to_write=500
sweep_multiple=6
in_param_file="./params.csv"
param_arr=pd.read_csv(in_param_file)
J11=param_arr.iloc[which_row,0]
J12=param_arr.iloc[which_row,1]
J21=param_arr.iloc[which_row,2]
J22=param_arr.iloc[which_row,3]
K=param_arr.iloc[which_row,4]
print(f"N={N}")
print(f"TVals={TVals}")
print(f"J11={J11}, J12={J12}, J21={J21}, K={K}")
TDirsAll=[]
TStrAll=[]
NStr=format_using_decimal(N)

for k in range(0,len(TVals)):
    T=TVals[k]
    TStr=format_using_decimal(T)
    TStrAll.append(TStr)

def contents_to_conf(k):
    contents=[
        "#This is the configuration file for 2d Square Kitaev mc computations\n",
        "\n" ,
        "#parameters\n",
        "\n",
        f"N={NStr}\n",
        "#Temperature\n",
        "T="+TStrAll[k]+"\n",
        "\n",
        f"J11={J11}\n",
        "\n",
        f"J12={J12}\n",
        "\n",
        f"J21={J21}\n",
        "\n",
        f"J22={J22}\n",
        "\n",
        f"K={K}\n",
        "\n",
        f"row={which_row}\n"
        "\n",
        "#this is the data number in each pkl file, i.e., in each flush\n"
        f"sweep_to_write={sweep_to_write}\n",
        "#within each flush,  sweep_to_write*sweep_multiple mc computations are executed\n",
        "\n",
        f"default_flush_num={default_flush_num}\n",
        "\n",
        "#the configurations of the system are saved to file if the sweep number is a multiple of sweep_multiple\n",
        "\n",
        f"sweep_multiple={sweep_multiple}\n",
        "\n",
        f"num_parallel={num_parallel}\n"
    ]
    outDir=dataRoot+f"/N{NStr}/row{which_row}/T{TStrAll[k]}/"
    Path(outDir).mkdir(exist_ok=True,parents=True)
    outConfName=outDir+f"/run_T{TStrAll[k]}.mc.conf"
    with open(outConfName,"w+") as fptr:
        fptr.writelines(contents)


for k in range(0,len(TVals)):
        contents_to_conf(k)