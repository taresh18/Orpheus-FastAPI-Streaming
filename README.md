### download models from huggingface
- `git lfs install`
- `git clone https://huggingface.co/lex-au/Orpheus-3b-FT-Q4_K_M.gguf` # or (https://huggingface.co/lex-au/Orpheus-3b-FT-Q8_0.gguf)

### how to run
- `git clone https://github.com/taresh18/Orpheus-FastAPI-Streaming.git`
- `cd Orpheus-FastAPI-Streaming`
- `conda env create -f environment.yml`
- `conda activate orpheus-fs`
- `cp .env.example .env`  # specify the model name / path here
- `python app.py` # it takes ~5min to get it running 

### benchmark
`python benchmark.py`

RTX 4090

| model | ttfb (ms) |
| --- | --- |
| canopylabs/orpheus-3b-0.1-ft | 360-370 |
| lex-au/Orpheus-3b-FT-Q8_0.gguf | ~300 |
| lex-au/Orpheus-3b-FT-Q4_K_M.gguf | ~250 |
