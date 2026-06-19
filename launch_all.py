import subprocess
import time

datasets = ['wisdm2','harth','pamap2', 'ecg2','skoda','sleepm']
baselines = ['Rand_Init','Shift_CL','SimCLR','TS2Vec','InfoTS']
seeds = [1,2,3,4,5]

# Loop through baselines and datasets
for baseline in baselines:
    for dataset in datasets:
        # Launch up to 5 seeds (on GPUs 0, 1, 2) at once
        processes = []
        for i, seed in enumerate(seeds):
            gpu = i % 3  # GPU 0, 1, 2
            config_file = f"configs/{dataset}config.yml"
            cmd = [
                "python",
                "pretrain_hyper.py",
                baseline,
                dataset,
                "-p", config_file,
                "-s", str(seed),
                #"--evaluate", "supervised",
                "-g", str(gpu)
            ]
            print(f"Launching: {' '.join(cmd)} on GPU {gpu}")
            p = subprocess.Popen(cmd)
            processes.append(p)

            # If 3 processes are running, wait before starting next
            if len(processes) == 3:
                for p in processes:
                    p.wait()
                processes = []

        # Wait for any remaining processes in the last batch
        for p in processes:
            p.wait()

print("✅ All runs finished.")
