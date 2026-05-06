pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install flash-attn==2.8.1 --no-build-isolation

# FP8
# git clone https://github.com/thu-ml/SageAttention.git
# cd SageAttention 
# export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 # Optional
# python setup.py install