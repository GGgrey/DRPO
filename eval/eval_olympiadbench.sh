#!/usr/bin/env bash
set -e

export VLLM_WORKER_MULTIPROC_METHOD=spawn

date=`date '+%Y-%m-%d-%H-%M-%S'`

export CUDA_VISIBLE_DEVICES="0,1,2,3"

NUM_GPU_PER_NODE=4
NUM_NODES=1
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

echo "NUM_GPUS: ${NUM_GPUS}"
echo "NUM_NODES: ${NUM_NODES}"
echo "NUM_GPU_PER_NODE: ${NUM_GPU_PER_NODE}"

total_start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Total start time: $total_start_time"

MODEL=/root/siton-data-0072803f053947c8bb3fe64d115b30e3/models/Qwen/Qwen2.5-1.5B
OUTPUT_DIR=./eval
CUSTOM_TASKS_PATH=./eval/custom_tasks/eval_all.py

START_SEED=1234
NUM_RUNS=16

MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=5120,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:4096,temperature:0.0,seed:1234}"

 # Greedy pass@1 evaluation
lighteval vllm \
    $MODEL_ARGS \
    "olympiadbench_custom" \
    --custom-tasks "$CUSTOM_TASKS_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --save-details

sleep 30

# Avg@n evaluation
for (( i=0; i<NUM_RUNS; i++ ))
do
    SEED=$((START_SEED + i))
    echo "----------------------------------------------------------------"
    echo "Running with SEED: ${SEED} ($((i+1))/${NUM_RUNS})"
    echo "----------------------------------------------------------------"

    MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=5120,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:4096,temperature:1.0,top_p:0.95,seed:${SEED}}"

    run_start_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "Run start time: $run_start_time"

    lighteval vllm \
        $MODEL_ARGS \
        "olympiadbench_custom" \
        --custom-tasks "$CUSTOM_TASKS_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --save-details

    sleep 30  # Wait for 30 seconds between runs
    
    echo "Finished run for seed ${SEED}"
done

total_end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "================================================================"
echo "All runs completed"
echo "Total start time: $total_start_time"
echo "Total End time:   $total_end_time"
echo "================================================================"