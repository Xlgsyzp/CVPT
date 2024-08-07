## Data Preparation
We provide two ways for preparing VTAB-1k:
- Download the source datasets, please refer to [NOAH](https://github.com/ZhangYuanhan-AI/NOAH/#data-preparation).
- We provide the prepared datasets, which can be download from  [google drive](https://drive.google.com/file/d/1yZKwiKdsBzTfBgnStRveYMokc7GMMd5p/view?usp=share_link).
Download FGVC, Please refer to [VPT](https://github.com/KMnP/vpt)

After that, the file structure should look like:
```
$ROOT/data
|-- cifar
|-- caltech101
......
|-- FGVC/
    |-- CUB_200_2011
    ......
|-- diabetic_retinopathy
```

- Download the [pretrained ViT-B/16](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) to `./ViT-B_16.npz`

## Environment settings

conda create -n cvpt python=3.7

conda activate cvpt

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch

pip install timm==0.5.4

pip install avalanche-lib==0.2.1


## Testing
bash scripts/test_all.sh
