FROM vllm/vllm-openai:latest

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    DRIVELM_VLLM_HOST=0.0.0.0 \
    DRIVELM_VLLM_PORT=8001 \
    DRIVELM_LORA_PATH=/app/models/qwen-lora

COPY src/ ./src/

EXPOSE 8001

ENTRYPOINT []
CMD ["python3", "-m", "src.vllm_launcher"]
