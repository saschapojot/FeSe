

python mk_dir.py, to set coefficients, T, and directories
generate bash files for supercomputer:
python gen_exec_sh_gansu.py
##########################################
To manually perform each step of computations for s
1. python launch_one_run.py ./path/to/mc.conf (runs ./init_run_scripts/parseConf.py and load_previous_data.py, the data are angles)
2. cmake .
3. make run_mc
4. ./run_mc ./path/to/cppIn.txt
5. after mc,  generate U by:
   python data2csv/pkl_U_data2csv.py N T row startingfileInd sweep_to_write lag sweep_multiple
    on supercomputer, use gen_pkl_U_data2csv_gansu.py to generate bash files, modify parameters in this file
###########################
##############################
plot U 
in plt/
the plots iterate different T for the same N
1. cd plt/
2. convert csv file of U to average value, for all T
   python compute_U_avg.py N row
3. plot U for all T
   python load_csv_plt_U.py  N  row
