### download models from huggingface
1. git lfs install
2. git clone https://huggingface.co/lex-au/Orpheus-3b-FT-Q4_K_M.gguf or (https://huggingface.co/lex-au/Orpheus-3b-FT-Q8_0.gguf)

### how to run
1. git clone https://github.com/taresh18/Orpheus-FastAPI-Streaming.git
2. cd Orpheus-FastAPI-Streaming
3. conda env create -f environment.yml
3. conda activate orpheus-fs
4. cp .env.example .env  # specify the model name / path here
56. python app.py