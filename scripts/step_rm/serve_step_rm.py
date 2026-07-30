# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Serve a trained step RM behind HTTP, for step-wise PPO / branch selection.

    python scripts/step_rm/serve_step_rm.py --model-path $CKPT_DIR/step_rm --port 8200

API (consumed by seed.step_rm.StepRewardModelClient):
    POST /score  {"texts": ["...", ...]}  ->  {"scores": [float, ...]}
    GET  /health ->  {"status": "ok"}
"""

import argparse

import torch
from fastapi import FastAPI
from pydantic import BaseModel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32, help="internal forward batch size")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


class ScoreRequest(BaseModel):
    texts: list


def build_app(model_path: str, max_length: int, batch_size: int, device: str) -> FastAPI:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok", "model_path": model_path}

    @app.post("/score")
    def score(request: ScoreRequest):
        texts = [str(t) for t in request.texts]
        scores = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                encoded = tokenizer(
                    chunk,
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                logits = model(**encoded).logits.squeeze(-1)
                scores.extend(logits.float().cpu().tolist())
        return {"scores": scores}

    return app


def main():
    import uvicorn

    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    app = build_app(args.model_path, args.max_length, args.batch_size, device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
