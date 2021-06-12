## CPM-Triplet-loss
* **Introduction**: This project implements the CPM-Triplet loss for person re-identification. Our work is based on : Bag of Tricks ([paper](https://arxiv.org/abs/1903.07071) and [official code](https://github.com/michuanhaohao/reid-strong-baseline)) and light-reid ([official code](https://github.com/wangguanan/light-reid)).
[](2.1-bot)


## Dependencies
* [Anaconda (Python 3.7)](https://www.anaconda.com/download/)
* [PyTorch 1.1.0](http://pytorch.org/)
* PrettyTable (```pip install prettytable```)
* GPU Memory >= 10G
* Memory >= 10G

## Dataset Preparation
* Market-1501 ([Project](http://www.liangzheng.com.cn/Project/project_reid.html), [Google Drive](https://drive.google.com/open?id=1M8m1SYjx15Yi12-XJ-TV6nVJ_ID1dNN5))
* DukeMTMC-reID ([Project](https://github.com/sxzrt/DukeMTMC-reID_evaluation), [Google Drive](https://drive.google.com/open?id=11FxmKe6SZ55DSeKigEtkb-xQwuq6hOkE))
* MSMT17 ([Project](https://www.pkuvmc.com/dataset.html), [Paper](https://arxiv.org/pdf/1711.08565.pdf))

## Run
#### Train on Market-1501/DukeMTMC-reID/MTMC17
```
python3 main.py --mode train \
    --train_dataset market --test_dataset market \
    --market_path /path/to/market/dataset/ \
    --output_path ./results/market/ 
python3 main.py --mode train \
    --train_dataset duke --test_dataset duke \
    --duke_path /path/to/duke/dataset/ \
    --output_path ./results/duke/
python3 main.py --mode train \
    --train_dataset msmt --test_dataset msmt --steps 400 --pid_num 1041 \
    --duke_path /path/to/msmt/dataset/ \
    --output_path ./results/msmt/
```



#### Train with ResNet50-IBNa backbone
```
# download model to ./core/nets/models/ from https://drive.google.com/file/d/1_r4wp14hEMkABVow58Xr4mPg7gvgOMto/view
python3 main.py --mode train -cnnbackbone res50ibna \
    --train_dataset market --test_dataset market \
    --market_path /path/to/market/dataset/ \
    --output_path ./results/market/ 
```

#### Train with OSNet-AIN backbone
```
# download model to ./core/nets/models/ from https://mega.nz/#!YTZFnSJY!wlbo_5oa2TpDAGyWCTKTX1hh4d6DvJhh_RUA2z6i_so
python3 main.py --mode train -cnnbackbone osnetain \
    --train_dataset market --test_dataset market \
    --market_path /path/to/market/dataset/ \
    --output_path ./results/market/ 
```

#### Test on Market-1501/DukeMTMC-reID/MTMC-17
```
python3 main.py --mode test \
    --train_dataset market --test_dataset market \
    --market_path /path/to/market/dataset/ \
    --resume_test_model /path/to/trained/model.pkl \ 
    --output_path ./results/test-on-market/
python3 main.py --mode test \
    --train_dataset duke --test_dataset duke \
    --market_path /path/to/duke/dataset/ \
    --resume_test_model /path/to/trained/model.pkl \ 
    --output_path ./results/test-on-duke/
python3 main.py --mode test \
    --train_dataset msmt --test_dataset msmt --pid_num 1041 \
    --market_path /path/to/msmt/dataset/ \
    --resume_test_model /path/to/trained/model.pkl \ 
    --output_path ./results/test-on-msmt/
```

#### Visualize Market-1501/DukeMTMC-reID
```
python3 main.py --mode visualize --visualize_mode inter-camera \
    --train_dataset market --visualize_dataset market \
    --market_path /path/to/market/dataset/ \
    --resume_visualize_model /path/to/trained/model.pkl \ 
    --visualize_output_path ./results/vis-on-market/ 
python3 main.py --mode visualize --visualize_mode inter-camera \
    --train_dataset duke --visualize_dataset duke \
    --market_path /path/to/duke/dataset/ \
    --resume_visualize_model /path/to/trained/model.pkl \ 
    --visualize_output_path ./results/vis-on-duke/ 
```

#### Visualize Customed Dataset with Trained Model

```
# customed dataset structure
|____ data_path/
     |____ person_id_1/
          |____ pid_1_imgid_1.jpg
          |____ pid_1_imgid_2.jpg
          |____ ......
     |____ person_id_2/
     |____ person_id_3/
     |____ ......
```
```
python3 demo.py \
    --resume_visualize_model /path/to/pretrained/model.pkl \
    --query_path /path/to/query/dataset/ --gallery_path /path/to/gallery/dataset/ \
    --visualize_output_path ./results/vis-on-cus/
```

## Experiments
Please select CPM-Triplet loss and Triplet loss in ```core/base.py```.




## Contacts
If you have any question about the project, please contact with me.

E-mail: weiyu_zeng@foxmail.com
