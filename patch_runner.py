import json
from pathlib import Path
from uuid import uuid4
import time

from natsort import natsorted
from tqdm import tqdm
import multiprocessing

from swebench.harness.run_evaluation import main as run_evaluation

def main(predictions_dir, run_id: str, workers=1) -> None:
    prediction_files = natsorted(Path(predictions_dir).glob("all_preds.jsonl"))
    for idx, prediction_file in tqdm(
        enumerate(prediction_files),
        desc="Evaluating prediction files",
        total=len(prediction_files),
    ):
        run_swe_bench_eval_official(run_id, str(prediction_file.resolve()), workers=workers)

def _task_function(run_id, p_idx, pred):
    swe_temp_input_file = f"/tmp/temp_{uuid4()}.json"
    instance_id = pred["instance_id"]
    indiv_run_id = f"{run_id}--{instance_id}--{p_idx}"
    with open(swe_temp_input_file, "w") as f:
        json.dump([pred], f)
    time.sleep(0.5)
    print(f"Starting {indiv_run_id}")

    run_evaluation(
        "/home/sungmin/ACR-setup/Agentless/local_swebench_verified.json",
        "test",
        [instance_id],
        swe_temp_input_file,
        1,
        force_rebuild=False,
        cache_level="env",
        clean=False,
        open_file_limit=4096,
        run_id=indiv_run_id,
        timeout=600,
    )

def run_swe_bench_eval_official(run_id: str, swe_input_file: str, workers=1):
    try:
        predictions = []
        with open(swe_input_file) as f:
            for line in f:
                predictions.append(json.loads(line))
    except Exception:
        print("Failed to read json")
        return None, None

    tasks = [
        (run_id, p_idx, pred)
        for p_idx, pred in enumerate(predictions)
    ]

    with multiprocessing.Pool(processes=workers) as pool:
        results = list(tqdm(
            pool.starmap(_task_function, tasks),
            total=len(tasks),
            desc="Evaluating prediction files",
        ))

    


if __name__ == "__main__":
    # Will run evaluations and results will be in logs/

    import argparse

    parser = argparse.ArgumentParser(description="Evaluate all predictions.")
    parser.add_argument(
        "--predictions-dir",
        "-d",
        type=str,
        help="Directory containing prediction files.",
        required=True,
    )
    parser.add_argument(
        "--run_id", type=str, help="Run ID for the evaluation.", required=True
    )
    parser.add_argument(
        "--workers", type=int, help="Run ID for the evaluation.", required=True
    )
    args = parser.parse_args()
    main(args.predictions_dir, args.run_id, workers=args.workers)
