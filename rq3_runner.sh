EXPL_PATH=$1
BUG_LIST_FILE=$2
EXPR_NAME=$3
DATETIME=$(date '+%Y%m%d_%H%M%S')
SAVEDIR="results/RQ3_${DATETIME}_${EXPR_NAME}"
MODEL="gpt-4o-2024-11-20"

for bugname in $(cat $BUG_LIST_FILE); do
    bug_savedir="$SAVEDIR/$bugname"
    PYTHONPATH=. python agentless/fl/localize.py \
        --dataset princeton-nlp/SWE-bench_Verified \
        --file_level \
        --output_folder ${bug_savedir}/file_fl \
        --num_threads 1 \
        --target_id $bugname \
        --model $MODEL \
        --expl_file $EXPL_PATH

    PYTHONPATH=. python agentless/fl/localize.py \
        --dataset princeton-nlp/SWE-bench_Verified \
        --related_level \
        --output_folder ${bug_savedir}/related_elements \
        --top_n 3 \
        --start_file ${bug_savedir}/file_fl/loc_outputs.jsonl \
        --num_threads 1 \
        --target_id $bugname \
        --model $MODEL \
        --expl_file $EXPL_PATH

    PYTHONPATH=. python agentless/fl/localize.py \
        --dataset princeton-nlp/SWE-bench_Verified \
        --fine_grain_line_level \
        --output_folder ${bug_savedir}/statement_fl \
        --top_n 3 \
        --start_file ${bug_savedir}/related_elements/loc_outputs.jsonl \
        --num_threads 1 \
        --target_id $bugname \
        --model $MODEL \
        --expl_file $EXPL_PATH

    PYTHONPATH=. python agentless/repair/repair.py \
        --dataset princeton-nlp/SWE-bench_Verified \
        --loc_file ${bug_savedir}/statement_fl/loc_outputs.jsonl \
        --output_folder ${bug_savedir}/repair_sample_1 \
        --loc_interval \
        --top_n 3 \
        --context_window 10 \
        --max_samples 10 \
        --cot \
        --diff_format \
        --gen_and_process \
        --num_threads 2 \
        --model $MODEL \
        --expl_file $EXPL_PATH
    
    total_cost=0
    for jsonl_fname in $(find ${bug_savedir} -iname "*loc_outputs.jsonl"); do
        indiv_cost=$(PYTHONPATH=. python dev/util/cost.py --output_file ${jsonl_fname})
        total_cost=$(python -c "print($total_cost + $indiv_cost)")
    done
    for jsonl_fname in $(find ${bug_savedir} -iname "*output.jsonl"); do
        indiv_cost=$(PYTHONPATH=. python dev/util/cost.py --output_file ${jsonl_fname})
        total_cost=$(python -c "print($total_cost + $indiv_cost)")
    done
    echo $total_cost > ${bug_savedir}/cost.txt
    echo $EXPL_PATH > ${bug_savedir}/used_expl_filename.txt
    
    rm -rf playground/*
done

python organize_results.py --expr_dir $SAVEDIR