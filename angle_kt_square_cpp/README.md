

python mk_dir.py, to set coefficients, T, and directories
generate bash files for supercomputer:
python gen_exec_sh_gansu.py
##########################################
To manually perform each step of computations for s
1. python launch_one_run.py ./path/to/mc.conf (runs ./init_run_scripts/parseConf.py and load_previous_data.py, the data are angles)
2. cmake .
3. make run_mc
4. ./run_mc ./path/to/cppIn.txt