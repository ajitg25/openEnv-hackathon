FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir transformers>=4.45.0 accelerate && \
    pip install --no-cache-dir nest_asyncio peft trl \
    pydantic fastapi uvicorn websockets httpx matplotlib

RUN pip install --no-cache-dir "openenv-core[core]>=0.2.2"

RUN useradd -m -u 1000 appuser
WORKDIR /app
COPY . .
RUN mkdir -p /app/plots && chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH="/app/envs:/app:$PYTHONPATH"
ENV AMBULANCE_DIFFICULTY="easy"

EXPOSE 7860

CMD ["sh", "-c", "python3 /app/train.py; echo 'Training done (exit $?). Serving plots...'; mkdir -p /app/plots && cd /app/plots && python3 -m http.server 7860"]
