import json
from pathlib import Path

from natsort import natsorted
from tqdm import tqdm


def main(predictions_dir, run_id: str) -> None:
    prediction_files = natsorted(Path(predictions_dir).glob("all_preds.jsonl"))
    for idx, prediction_file in tqdm(
        enumerate(prediction_files),
        desc="Evaluating prediction files",
        total=len(prediction_files),
    ):
        run_swe_bench_eval_official(f"{run_id}-{idx}", str(prediction_file.resolve()))


def run_swe_bench_eval_official(run_id: str, swe_input_file: str):
    from swebench.harness.run_evaluation import main as run_evaluation

    try:
        predictions = []
        with open(swe_input_file) as f:
            for line in f:
                predictions.append(json.loads(line))
    except Exception:
        print("Failed to read json")
        return None, None

    swe_temp_input_file = "temp.json"
    for p_idx, pred in enumerate(predictions):
        with open(swe_temp_input_file, "w") as f:
            json.dump([pred], f)

        print(pred["instance_id"])
        run_evaluation(
            "princeton-nlp/SWE-bench",
            "test",
            [pred["instance_id"]],
            swe_temp_input_file,
            1,
            force_rebuild=False,
            cache_level="env",
            clean=False,
            open_file_limit=4096,
            run_id=f"{run_id}-{p_idx}",
            timeout=1800,
        )


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
    args = parser.parse_args()
    main(args.predictions_dir, args.run_id)
