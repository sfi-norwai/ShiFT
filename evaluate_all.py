import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import tasks
from src.src_utils.load_data import preload_data
from src.src_utils.load_model import initialize_model
import src
from special_tasks.train_model import train_supervised_pretrain, train_supervised_pretrain_pool
from special_tasks.evaluate_model import eval_supervised, eval_supervised2
from src.loader.dataloader import stratified_fixed_count, stratified_percentage, stratified_percentage_ecg2, stratified_fixed_count_ecg2
from sklearn.preprocessing import label_binarize
import random

RESULTS_DIR = "./results"
SUB_TASK = 'linear_probing'
#SUB_TASK = 'finetuning'
eval_dir = f'evaluate_csv/{SUB_TASK}'
os.makedirs(eval_dir, exist_ok=True)

gpu_num = 2
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")

# ------------------------------
# MAIN LOOP
# ------------------------------
for dataset in tqdm(os.listdir(RESULTS_DIR), desc="Datasets"):
    dataset_path = os.path.join(RESULTS_DIR, dataset)
    if not os.path.isdir(dataset_path):
        continue

    print(f"\n🔹 Processing dataset: {dataset}")
    config = src.config.Config(f"configs/{dataset}config.yml")

    # Get algorithm and dataset arguments
    for args in src.utils.grid_search(config.ALGORITHM_ARGS):
        args = args
    for ds_args in src.utils.grid_search(config.DATASET_ARGS):
        ds_args = ds_args

    results = []  # store results for this dataset only

    for seed_folder in os.listdir(dataset_path):
        seed_path = os.path.join(dataset_path, seed_folder)
        if not os.path.isdir(seed_path):
            continue

        seed = int(seed_folder.replace("seed_", ""))

        # Set all seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Multi-GPU
        
        train_ds, valid_ds = preload_data(dataset)

        # Load dataset
        # Subset train dataset to fixed samples per class
        print("Original size:", len(train_ds))

       # if dataset=='ecg2':
       #     train_ds = stratified_percentage_ecg2(train_ds, percentage=0.01)
            #train_ds = stratified_fixed_count_ecg2(train_ds, n_per_class=50)
       # else:
       #     train_ds = stratified_percentage(train_ds, percentage=0.01)
            #train_ds = stratified_fixed_count(train_ds, n_per_class=50)

        print("Subset size:", len(train_ds))


        for method_folder in os.listdir(seed_path):
            method_path = os.path.join(seed_path, method_folder)
            if not os.path.isdir(method_path):
                continue

            # Extract method name (everything before timestamp)
            method = "_".join(method_folder.rsplit("_", 2)[:-2])
            model_path = os.path.join(method_path, f"model.pt")
            time_path = os.path.join(method_path, "time.txt")

            if not os.path.exists(model_path) or not os.path.exists(time_path):
                continue

            # Load training time
            with open(time_path, "r") as f:
                train_time = f.read().strip()

            # Load model
            print(f"\nEvaluating method: {method}, seed: {seed}")
            model = initialize_model(method, args, config, ds_args, device)
            model.load(model_path)

            # print(f"\nEvaluating method: {method}, seed: {seed}")

            # --------------------
            # EVALUATE MODEL
            # --------------------
            if SUB_TASK == 'linear_probing':
                if method in ['SimMTM','SimCLR','ShiFT','Rand_Init']:
                    eval_res = tasks.supervised_evaluation_ponly_block(
                        model, train_ds, valid_ds,
                        args['out_features'], args['linear_epochs'], args['batch_size'], config
                    )
                elif method == 'SupervisedE2E':
                    eval_res = eval_supervised(model,
                        valid_ds, args['out_features'], args['linear_epochs'], args['batch_size'], config)

                else:
                    eval_res = tasks.supervised_evaluation_ponly(
                        model, train_ds, valid_ds,
                        args['out_features'], args['linear_epochs'], args['batch_size'], config
                    )

            elif SUB_TASK == 'finetuning':
                if method in ['SimMTM','SimCLR','Shift_CL','Rand_Init']:
                    trained_model = train_supervised_pretrain_pool(
                        model, train_ds, ds_args['num_labels'], args, config, device=device
                    )
                    eval_res = eval_supervised2(
                        trained_model, valid_ds,
                        args['out_features'], args['linear_epochs'], args['batch_size'], config
                    )
                else:
                    trained_model = train_supervised_pretrain(
                        model, train_ds, ds_args['num_labels'], args, config, device=device
                    )
                    eval_res = eval_supervised2(
                        trained_model, valid_ds,
                        args['out_features'], args['linear_epochs'], args['batch_size'], config
                    )

            print("Eval result:", eval_res)

            # --------------------
            # STORE METRICS
            # --------------------
            metric_names = ['accuracy', 'f1', 'precision', 'recall']
            avg_metrics = {name: eval_res[i] for i, name in enumerate(metric_names)}
            avg_metrics_std = {name + "_std": 0.0 for name in metric_names}  # single run, std = 0

            results.append({
                "dataset": dataset,
                "method": method,
                "seed": seed,
                **avg_metrics,
                **avg_metrics_std,
                "train_time": train_time
            })

    # ------------------------------
    # AGGREGATE RESULTS FOR THIS DATASET
    # ------------------------------
    if results:
        df = pd.DataFrame(results)

        # Count number of runs per (dataset, method)
        df["n"] = df.groupby(["dataset", "method"])["accuracy"].transform("count")

        # --- Aggregate ---
        agg = df.groupby(["dataset", "method"]).agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            n=("n", "mean"),
            train_time=("train_time", lambda x: np.sum([float(t.split()[0]) if t else 0 for t in x])),
        ).reset_index()

        # --- Compute 95% Confidence Intervals ---
        for metric in ["accuracy", "f1", "precision", "recall"]:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"

            agg[f"{metric}_ci_lower"] = agg[mean_col] - 1.96 * (agg[std_col] / np.sqrt(agg["n"]))
            agg[f"{metric}_ci_upper"] = agg[mean_col] + 1.96 * (agg[std_col] / np.sqrt(agg["n"]))

        # Save results
        output_csv = f"{eval_dir}/{dataset}_results.csv"
        agg.to_csv(output_csv, index=False)

        print(agg)
        print(f"✅ Saved {output_csv}")

    else:
        print(f"⚠️ No valid results found for {dataset}")

