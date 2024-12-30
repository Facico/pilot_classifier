CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-0.5B \
    --enable-lora \
    --lora-modules lora1=/home/fch/pilot_classifier/output \
    --dtype bfloat16
