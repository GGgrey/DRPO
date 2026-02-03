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

start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Start time: $start_time"

MODEL=/root/siton-data-0072803f053947c8bb3fe64d115b30e3/verl_exp/ckpts/Qwen2.5-1.5B-DRPO-Math-1Epoch-1

MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=5120,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:4096,temperature:1.0,top_p:0.95,seed:1234}"
# MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=5120,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:4096,temperature:0.0,seed:1234}"

OUTPUT_DIR=./eval
CUSTOM_TASKS_PATH=./eval/custom_tasks/eval_all.py

# lighteval vllm \
#     $MODEL_ARGS \
#     "math500_custom" \
#     --custom-tasks "$CUSTOM_TASKS_PATH" \
#     --output-dir "$OUTPUT_DIR" \
#     --save-details

lighteval vllm \
    $MODEL_ARGS \
    "math500_avgn_custom" \
    --custom-tasks "$CUSTOM_TASKS_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --save-details

# lighteval vllm \
#     $MODEL_ARGS \
#     "gsm8k_custom" \
#     --custom-tasks "$CUSTOM_TASKS_PATH" \
#     --output-dir "$OUTPUT_DIR" \
#     --save-details

lighteval vllm \
    $MODEL_ARGS \
    "gsm8k_avgn_custom" \
    --custom-tasks "$CUSTOM_TASKS_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --save-details

# lighteval vllm \
#     $MODEL_ARGS \
#     "aime25_custom" \
#     --custom-tasks "$CUSTOM_TASKS_PATH" \
#     --output-dir "$OUTPUT_DIR" \
#     --save-details

lighteval vllm \
    $MODEL_ARGS \
    "aime25_avgn_custom" \
    --custom-tasks "$CUSTOM_TASKS_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --save-details

# lighteval vllm \
#     $MODEL_ARGS \
#     "aime24_custom" \
#     --custom-tasks "$CUSTOM_TASKS_PATH" \
#     --output-dir "$OUTPUT_DIR" \
#     --save-details

lighteval vllm \
    $MODEL_ARGS \
    "aime24_avgn_custom" \
    --custom-tasks "$CUSTOM_TASKS_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --save-details

# lighteval vllm \
#     $MODEL_ARGS \
#     "gpqa_diamond_custom" \
#     --custom-tasks "$CUSTOM_TASKS_PATH" \
#     --output-dir "$OUTPUT_DIR" \
#     --save-details

lighteval vllm \
    $MODEL_ARGS \
    "gpqa_diamond_avgn_custom" \
    --custom-tasks "$CUSTOM_TASKS_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --save-details

# lighteval vllm \
#     $MODEL_ARGS \
#     "minerva_custom" \
#     --custom-tasks "$CUSTOM_TASKS_PATH" \
#     --output-dir "$OUTPUT_DIR" \
#     --save-details

lighteval vllm \
    $MODEL_ARGS \
    "minerva_avgn_custom" \
    --custom-tasks "$CUSTOM_TASKS_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --save-details

# lighteval vllm \
#     $MODEL_ARGS \
#     "amc23_custom" \
#     --custom-tasks "$CUSTOM_TASKS_PATH" \
#     --output-dir "$OUTPUT_DIR" \
#     --save-details

lighteval vllm \
    $MODEL_ARGS \
    "amc23_avgn_custom" \
    --custom-tasks "$CUSTOM_TASKS_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --save-details

# lighteval vllm \
#     $MODEL_ARGS \
#     "olympiadbench_custom" \
#     --custom-tasks "$CUSTOM_TASKS_PATH" \
#     --output-dir "$OUTPUT_DIR" \
#     --save-details

lighteval vllm \
    $MODEL_ARGS \
    "olympiadbench_avgn_custom" \
    --custom-tasks "$CUSTOM_TASKS_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --save-details

# lighteval vllm \
#     $MODEL_ARGS \
#     "gsm_plus_custom" \
#     --custom-tasks "$CUSTOM_TASKS_PATH" \
#     --output-dir "$OUTPUT_DIR" \
#     --save-details

lighteval vllm \
    $MODEL_ARGS \
    "gsm_plus_avgn_custom" \
    --custom-tasks "$CUSTOM_TASKS_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --save-details

# lighteval vllm \
#     $MODEL_ARGS \
#     "mmlu_pro_custom" \
#     --custom-tasks "$CUSTOM_TASKS_PATH" \
#     --output-dir "$OUTPUT_DIR" \
#     --save-details

lighteval vllm \
    $MODEL_ARGS \
    "mmlu_pro_avgn_custom" \
    --custom-tasks "$CUSTOM_TASKS_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --save-details

end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "End time: $end_time"
