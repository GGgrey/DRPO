#!/usr/bin/env bash
set -e

export VLLM_WORKER_MULTIPROC_METHOD=spawn

date=`date '+%Y-%m-%d-%H-%M-%S'`

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

NUM_GPU_PER_NODE=8
NUM_NODES=1
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

echo "NUM_GPUS: ${NUM_GPUS}"
echo "NUM_NODES: ${NUM_NODES}"
echo "NUM_GPU_PER_NODE: ${NUM_GPU_PER_NODE}"

start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Start time: $start_time"

MODEL=/workspace/verl_exp/ckpts/Qwen2.5-1.5B-CDPO-Math-1Epoch

MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=5120,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:4096,temperature:1.0,top_p:0.95,seed:1234}"

OUTPUT_DIR=./eval
CUSTOM_TASKS_PATH=./eval/custom_tasks/eval_all.py

tasks=(
    math500_avgn_custom
    gsm8k_avgn_custom
    aime25_avgn_custom
    aime24_avgn_custom
    gpqa_diamond_avgn_custom
    minerva_avgn_custom
    amc23_avgn_custom
    olympiadbench_avgn_custom
    gsm_plus_avgn_custom
    mmlu_pro_avgn_custom
)

for task in "${tasks[@]}"; do
    echo "===== Running task: ${task} ====="
    lighteval vllm \
        $MODEL_ARGS \
        "${task}" \
        --custom-tasks "$CUSTOM_TASKS_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --save-details

    if [[ "${task}" != "${tasks[-1]}" ]]; then
        sleep 30
    fi
done

end_time=$(date +"%Y-%m-%d %H:%M:%S")   
echo "End time: $end_time"
