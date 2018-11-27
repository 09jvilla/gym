import subprocess

memsizes = [5000,10000,50000,100000]
epochs = [1, 10, 50, 100]

#memsizes = [3000]
#epochs = [1]

for m in memsizes:
    for e in epochs:
        logname = "logs_sweep/log_e" + str(e) + "_mem" + str(m) + ".txt"
        strcmd = "python -u qlearning_deep.py --num_train_trials=10000 --num_test_trials=0 --num_epochs=" + str(e) + " --memsize=" + str(m) + " > " + logname
        print("Starting: " + strcmd)
        subprocess.run(strcmd, shell=True)
