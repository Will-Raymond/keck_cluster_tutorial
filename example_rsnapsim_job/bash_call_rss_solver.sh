# First we touch the log file so it isnt buggy when we write to it
# then we call our LOCAL enviroment here, cc_env should be replaced here with wahatever you named your enviroment. Basically which python do you want to use
touch Logs/Log_bactin_kis_$save_name.txt
~/.conda/envs/cc_env/bin/python run_rss_solver.py $genefile $time $ki $n_traj $save_dir $save_name > Logs/Log_bactin_kis_$save_name.txt
