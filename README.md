# Learning without Forgetting for 3D Point Cloud Objects

## Requirements
Install necessary packages from `requirements.txt` [file](./requirements.txt).

## Data
Semantic embedding for the dataset will be found [here](https://drive.google.com/drive/folders/1rqRyRI_i3MVRd_cLEBlmNNvggGBO-lEl?usp=sharing).

## Model
Pretrained model of old task: [here](https://drive.google.com/drive/folders/1rfJQXXCOts024vnBisD5ITCaGRSVeKSt?usp=sharing)

Pretrained model of new task: [hare](https://drive.google.com/drive/folders/1mYmwUTHscR3pdP8MJGHGM7bwO1VWlZqy?usp=sharing)

## Training and Evaluation
For each dataset, there is a corresponding configuration files located in `config` folder. Below is the description of configuration file.
```
seen_class : number of classes for old task
unseen_class : number of classes for new task
total_class : number of total classes
dataset_path : path of the dataset i.e. "content/ModelnetNew"
saved_model : folder to save model for new task
batch_size : batch size
lr : learning rate
wd : weighting decay
T: temperature for KD loss
pointnet_old_model_path_none: model path for old task with no semantic information using pointnet
pointnet_old_model_path_w2v: model path for old task with w2v using pointnet
pointnet_old_model_path_glove: model path for old task with glove using pointnet
pointconv_old_model_path_none: model path for old task with no semantic information using pointconv
pointconv_old_model_path_w2v: model path for old task with w2v using pointconv
pointconv_old_model_path_glove: model path for old task with glove using pointconv
dgcnn_old_model_path_none: model path for old task with no semantic information using dgcnn
dgcnn_old_model_path_w2v:  model path for old task with w2v using dgcnn
dgcnn_old_model_path_glove: model path for old task with glove using dgcnn
```

For training, arguments for each python script are:
```
--dataset: ModelNet, ScanObjectNN
--epoch: number of epochs 
--sem: using semantic representation i.e. w2v, glove, none
```

## Acknowledgements
This implementation has been based on these repositories: [PointNet](https://github.com/yanx27/Pointnet_Pointnet2_pytorch), [PointConv](https://github.com/DylanWusee/pointconv_pytorch) and [DGCNN](https://github.com/WangYueFt/dgcnn/tree/master/pytorch).

## Citation
```
@inproceedings{lwf3D2021,
  title={Learning without Forgetting for 3D Point Cloud Objects},
  author={Townim Chowdhury, Mahira Jalisha, Ali Cheraghian, and Shafin Rahman},
  booktitle = {International Work-Conference on Artificial Neural Networks (IWANN)},
  year={2021}
}
```
