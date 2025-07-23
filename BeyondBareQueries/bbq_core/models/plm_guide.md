# Perception LM installation

**![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUdXkJxzs3iifZidRznm3IjU2YOtZ0aqL0o_WrOYSe900hMlJycO_Q6iYqQ7gOSlAo8vKPISpVSesfXwieNjdFsqWgVWB4DFXqCY1BFU7JvWViodBRx4mY70fgF-npekRCnlOjpzjw=s2048?key=7BOUoDaKP7QdhnI58dK_8AnP)**

## 1st step
Inside our bbq docker container run follow command to prepare venv:
```
git clone https://github.com/facebookresearch/perception_models.git
cd perception_models

conda create --name perception_models python=3.12
conda activate perception_models

# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers --index-url https://download.pytorch.org/whl/cu118

# We use torchcodec for decoding videos into PyTorch tensors
conda install ffmpeg
pip install torchcodec==0.1 --index-url=https://download.pytorch.org/whl/cu118

pip install -e .

#additional dependencies
pip install uvicorn
pip install fastapi
```

## 2nd step
Inside `perception_models` env run:
```
cd BeyondBareQueries/bbq/models
CUDA_VISIBLE_DEVICES=1 uvicorn plm_server:app --host 0.0.0.0 --port 31623
```

After that you can execute main code in `bbq_env`:
```
cd BeyondBareQueries
CUDA_VISIBLE_DEVICES=1 python server.py
```
