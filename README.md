# TODO : REWRITE THIS


## Installing requirements (Ubuntu 20.04)
```
sudo apt install python3.8-venv
sudo apt install libpq-dev libhdf5-dev cython3 python-dev libfreetype6-dev
```

For automatically synching with google doc (but need to configure ~/.config/rclone/rclone.conf):
```
sudo apt install rclone
```

## Downloading the data
... assuming it is downloaded on `../data`

## Setting up for running
```
ln -s ../data .
python3 -m venv venv
ln -snf venv/bin/activate activate_venv
source activate_venv
```

### Torch 1.5 (older GPUs)
```
pip install -r requirements.txt
```

### Torch 1.7 (newer GPUs requiring CUDA 11)
```
pip install -r requirements_torch1.7.txt -f https://download.pytorch.org/whl/torch_stable.html
```
