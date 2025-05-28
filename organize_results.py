import argparse
import os, json
import re
from glob import glob

def find_processed_files(expr_dir):
    return glob(os.path.join(".", expr_dir, "**", "output_*_processed.jsonl"), recursive=True)

def get_patch_from_file(fpath):
    with open(fpath) as f:
        patch_info = json.load(f)
    for key in ["raw_model_patch", "original_file_content", "edited_files", "new_file_content"]:
        del patch_info[key]
    return patch_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr_dir")
    args = parser.parse_args()

    patch_files = find_processed_files(args.expr_dir)
    all_info = []
    with open(os.path.join(args.expr_dir, "all_preds.jsonl"), "w") as f:
        for fpath in patch_files:
            patch_info = get_patch_from_file(fpath)
            print(json.dumps(patch_info), file=f)
        