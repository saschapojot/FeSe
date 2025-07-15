import re
import sys
import json
import os

# this script parse conf file and return the parameters as json data

fmtErrStr = "format error in parseConf.py: "
fmtCode = 1
valueMissingCode = 2
paramErrCode = 3
fileNotExistErrCode = 4

if len(sys.argv) != 2:
    print("wrong number of arguments.")
    exit(paramErrCode)
inConfFile = sys.argv[1]

def removeCommentsAndEmptyLines(file):
    """
    :param file: conf file
    :return: contents in file, with empty lines and comments removed
    """
    with open(file, "r") as fptr:
        lines = fptr.readlines()

    linesToReturn = []
    for oneLine in lines:
        oneLine = re.sub(r'#.*$', '', oneLine).strip()
        if oneLine:
            linesToReturn.append(oneLine)
    return linesToReturn

def parseConfContents(file):
    """
    :param file: conf file
    :return:
    """
    if not os.path.exists(file):
        print(file + " does not exist.")
        exit(fileNotExistErrCode)

    linesWithCommentsRemoved = removeCommentsAndEmptyLines(file)

    # Initialize all variables
    params = {
        "T": "",
        "J11": "",
        "J12": "",
        "J21": "",
        "J22": "",
        "K": "",
        "N": "",
        "row": "",
        "sweep_to_write": "",
        "default_flush_num": "",
        "sweep_multiple": "1",  # default value
        "num_parallel": "1"     # default value
    }

    # Define which parameters expect floats vs integers
    float_params = {"T", "J11", "J12", "J21", "J22", "K", "N", "row"}
    int_params = {"sweep_to_write", "default_flush_num", "sweep_multiple", "num_parallel"}

    # Regex patterns
    float_pattern = r"^[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?$"
    int_pattern = r"^\d+$"

    for oneLine in linesWithCommentsRemoved:
        matchLine = re.match(r'(\w+)\s*=\s*(.+)', oneLine)
        if matchLine:
            key = matchLine.group(1).strip()
            value = matchLine.group(2).strip()

            if key in float_params:
                if re.match(float_pattern, value):
                    params[key] = value
                else:
                    print(fmtErrStr + oneLine)
                    exit(fmtCode)
            elif key in int_params:
                if re.match(int_pattern, value):
                    params[key] = value
                else:
                    print(fmtErrStr + oneLine)
                    exit(fmtCode)
            else:
                print("line: " + oneLine + " is discarded (unknown parameter).")
        else:
            print("line: " + oneLine + " is discarded (invalid format).")

    # Check for required parameters
    required_params = ["T", "J11", "J12", "J21", "J22", "K", "N", "row",
                       "sweep_to_write", "default_flush_num"]

    for param in required_params:
        if params[param] == "":
            print(f"{param} not found in {file}")
            exit(valueMissingCode)

    # Add filename to the dictionary
    params["confFileName"] = file

    return params

jsonDataFromConf = parseConfContents(inConfFile)
confJsonStr2stdout = "jsonDataFromConf=" + json.dumps(jsonDataFromConf)
print(confJsonStr2stdout)