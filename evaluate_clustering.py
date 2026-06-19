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
from src.loader.dataloader import stratified_fixed_count
from sklearn.preprocessing import label_binarize

RESULTS_DIR = "./results"
SUB_TASK = 'clustering'
eval_dir = f'evaluate_csv/{SUB_TASK}'
os.makedirs(eval_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Load dataset
    train_ds, valid_ds = preload_data(dataset)

    # Subset train dataset to fixed samples per class
    # print("Original size:", len(train_ds))
    # train_ds = stratified_fixed_count(train_ds, n_per_class=1)
    # print("Subset size:", len(train_ds))

    #print("Original size:", len(valid_ds))
    #valid_ds = stratified_fixed_count(valid_ds, n_per_class=100)
    #print("Subset size:", len(valid_ds))

    results = []  # store results for this dataset only

    for seed_folder in os.listdir(dataset_path):
        seed_path = os.path.join(dataset_path, seed_folder)
        if not os.path.isdir(seed_path):
            continue

        seed = int(seed_folder.replace("seed_", ""))

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
            if SUB_TASK == 'clustering':
                if method in ['SimMTM','ShiFT','SimCLR','Rand_Init']:
                    eval_res = tasks.clustering_evaluation_ponly_block(
                        model, valid_ds, seed
                    )
                    
                else:
                    eval_res = tasks.clustering_evaluation_ponly(
                        model, valid_ds, seed
                    )

            print("Eval result:", eval_res)

            # --------------------
            # STORE METRICS
            # --------------------
            metric_names = ['nmi', 'ari']
            avg_metrics = {name: eval_res[i] for i, name in enumerate(metric_names)}
            avg_metrics_std = {name + "_std": 0.0 for name in metric_names}  # single run, std = 0

            results.append({
                "dataset": dataset,
                "method": method,
                "seed": seed,
                **avg_metrics,
                **avg_metrics_std,
            })

    # ------------------------------
    # AGGREGATE RESULTS FOR THIS DATASET
    # ------------------------------
    if results:
        df = pd.DataFrame(results)
        agg = df.groupby(["dataset", "method"]).agg({
            "nmi": ["mean", "std"],
            "ari": ["mean", "std"]
        }).reset_index()

        # Flatten multi-level columns
        agg.columns = ["_".join(col).strip("_") for col in agg.columns.values]
        agg.rename(columns={"dataset_": "dataset", "method_": "method"}, inplace=True)

        output_csv = f"{eval_dir}/{dataset}_results.csv"
        agg.to_csv(output_csv, index=False)
        print(f"✅ Saved {output_csv}")
    else:
        print(f"⚠️ No valid results found for {dataset}")
