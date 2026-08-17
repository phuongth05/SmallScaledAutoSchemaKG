# AutoSchemaKG Colab v1

This workflow runs a genuinely local LLM and does not require a commercial
LLM API. It is intentionally small enough for a personal Colab GPU runtime.

## Model choice

There is no official Qwen3.5 3B checkpoint. The default is the closest smaller
official checkpoint, `Qwen/Qwen3.5-2B`. The model is served in text-only mode
through vLLM and called through its local OpenAI-compatible endpoint.

`Qwen/Qwen3.5-4B` can be selected by changing `MODEL_ID` in the notebook, but
the 2B model is the safer default for free notebook GPUs.

## Run on Colab

1. Open `colab/AutoSchemaKG_v1_Qwen35_2B.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Run the cells from top to bottom.
4. Download `autoschemakg_colab_v1.zip` at the end.

The notebook performs these stages:

1. clones this repository;
2. installs AutoSchemaKG and the current vLLM runtime;
3. starts `Qwen/Qwen3.5-2B` locally;
4. extracts entity/entity, entity/event, and event/event relations;
5. induces concepts;
6. exports CSV and GraphML;
7. writes a machine-readable run summary.

The install cell removes Colab's preinstalled `torchaudio` and `torchvision`
wheels after installing vLLM. They are not used by this text-only workflow and
can otherwise remain compiled for a different CUDA version than the PyTorch
version selected by vLLM.

### Recover an already-running Colab session

If vLLM reports that PyTorch and TorchAudio were compiled with different CUDA
versions, run this once and then rerun the model-server cell:

```python
!pip uninstall -y torchaudio torchvision
```

## Run from a Linux GPU machine

Start the model server:

```bash
vllm serve Qwen/Qwen3.5-2B \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --language-model-only
```

Then run the pipeline:

```bash
python scripts/run_colab_v1.py --overwrite
```

Useful options:

```bash
# Faster extraction-only check
python scripts/run_colab_v1.py --phase extract --overwrite

# Build CSV/GraphML from an existing extraction
python scripts/run_colab_v1.py --phase build

# Use a different local model exposed by the same endpoint
python scripts/run_colab_v1.py --model Qwen/Qwen3.5-4B --overwrite
```

Generated outputs are ignored by Git. Do not commit model weights, API tokens,
FAISS indices, or experimental outputs.
