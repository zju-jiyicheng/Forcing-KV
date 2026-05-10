# Chunk Discontinuity Metric
This metric evaluateds the chunk discontinuity of autoregressive video diffusion models, using the source code of paper:

[RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)<br/>



## Requirements
```Shell
conda create --name raft
conda activate raft
conda install pytorch torchvision
pip install -r requirements.text
```

## Model
Pretrained models can be downloaded by running
```Shell
./download_models.sh
```
or downloaded from [google drive](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing)



## Evaluating Chunk Discontinuity
You can evaluate single video or videos in a folder at `optical_metrics_xs.sh` by setting the number of chunks.