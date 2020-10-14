# [Fish identification based on lateral skin](https://github.com/charlesLucky/lateral)


****

## Contents
* [Installation](#Installation)
* [Data Preparing](#Data-Preparing)
* [Training and Testing](#Training-and-Testing)
* [References](#References)

## Installation
Create a new python virtual environment by [Anaconda](https://www.anaconda.com/) or just use pip in your python environment and then clone this repository as following.

### Clone this repo
```bash
git clone https://github.com/charlesLucky/lateral
cd lateral
```

### Conda
```bash
conda env create -f environment.yml
conda activate lateral-tf2
```

### Pip
```bash
pip install -r requirements.txt
```

## Image pre-process and Data Preparing

The data directory structure:

``
(base) xingbo@xingbo-pc:/lateral/data/SESSION_TENT_NEW$ ls
SESSION1_LT  SESSION1_ST  SESSION2  SESSION3  SESSION4
``

To generate proper images which can be used directly in training, we employ the same pre-processing protocol as Petr[?] proposed. 

To segment the fish from the background, images are firstly transformed into HSV color space, green colour was detected and used to detect the background area. The fish was detected as the largest object inside the background. Finally the detected fish was rotated based on automatically estimated ellipse around the fish.

However, the segmentation of the whole fish is challenging due to semitransparent tail fin, crop a portion of the fish body will be a good option to generate stable and consistent images for model training. The ROI crop area is determined as the area between eye position (EP) and upper fin (UP) beginning in horizontal direction and between upper fin beginning + belly point (BP)/20 and BP/2 in vertical direction. 


Deep neural network can learn very complex model from given training dataset. However, the training will tend to be overfitting on small dataset.  Data augmentation is a very simple and effective method against overfitting and helps the model generalize better.

In order to make the most of our few training examples, we will augment captured fish images via a number of random transformations, including random cropping, horizontal random flipping,random saturation and brightness.

To prepare the training set and testing dataset, 



### Training Dataset

### Testing Dataset

## Training and Testing

### To train the model
You can modify your own dataset path or other settings of model in [./configs/*.yaml]() for training and testing, which like below.

```python
# general (shared both in training and testing)
batch_size: 128
input_size: 112
embd_shape: 512
sub_name: 'arc_res50'
backbone_type: 'ResNet50' # or 'MobileNetV2'
head_type: ArcHead # or 'NormHead': FC to targets.
is_ccrop: False # central-cropping or not

# train
train_dataset: './data/ms1m_bin.tfrecord' # or './data/ms1m.tfrecord'
binary_img: True # False if dataset is online decoding
num_classes: 85742
num_samples: 5822653
epochs: 5
base_lr: 0.01
w_decay: !!float 5e-4
save_steps: 1000

# test
test_dataset: '/your/path/to/test_dataset'
```

Note:
- The `sub_name` is the name of outputs directory used in checkpoints and logs folder. (make sure of setting it unique to other models)
- The `head_type` is used to choose [ArcFace](https://arxiv.org/abs/1801.07698) head or normal fully connected layer head for classification in training. (see more detail in [./modules/models.py](https://github.com/peteryuX/arcface-tf2/blob/master/modules/models.py#L90-L94))
- The `is_ccrop` means doing central-cropping on both trainging and testing data or not.
- The `binary_img` is used to choose the type of training data, which should be according to the data type you created in the [Data-Preparing](#Data-Preparing).

### To train deep IoM in the second stage

The model (checkpoints file) of the first stage shall be stored under [checkpoints/arc_res50/*]()

You can modify settings of model in [./config_*/*.yaml]() for training and testing, which like below.

```python
# general
samples_per_class: 4
classes_per_batch: 50
batch_size: 240
eval_batch_size: 200
input_size: 112
# embd_shape is for the Resnet, backbone
embd_shape: 512
```
```python
sub_name: 'cfg9_1layer_arc_all_T300_256x32_0'
backbone_type: 'ResNet50' # 'ResNet50', 'MobileNetV2'
head_type: IoMHead # 'ArcHead', 'NormHead'
bin_lut_loss: 'tanh'
hidden_layer_remark: '1'
#T: 300
code_balance_loss: True
quanti: True
# train /media/xingbo/Storage/facedata/vgg_mtcnnpy_160 ./data/split /media/xingbo/Storage/facedata/ms1m_align_112/imgs
train_dataset: '/home/datascience/xingbo/ms1m_align_112/imgs'
#train_dataset: '/media/xingbo/Storage/facedata/vgg_mtcnnpy_160'
img_ext: 'jpg'
dataset_ext: 'ms'
# for metric learning, we have 1. triplet_semi batch_hard_triplet 2. batch_all_triplet_loss 3. batch_all_arc_triplet_loss batch_hard_arc_triplet_loss
loss_fun: 'batch_hard_arc_triplet_loss'
triplet_margin: 0
binary_img: False
num_classes: 85742
# I generate 1 milion triplets
num_samples: 3000000
epochs: 50
base_lr: 0.001
w_decay: !!float 5e-4
save_steps: 10000
q: 32
# m, the projection length would be m x q
m: 256
# test
test_dataset: './data/test_dataset'
test_dataset_ytf: './data/test_dataset/aligned_images_DB_YTF/'
test_dataset_fs: './data/test_dataset/facescrub_images_112x112/'
```
Note:
- The `sub_name` is the name of outputs directory used in checkpoints and logs folder. (make sure of setting it unique to other models)
- The `head_type` is used to choose [ArcFace](https://arxiv.org/abs/1801.07698) head or normal fully connected layer head for classification in training. (see more detail in [./modules/models.py](https://github.com/peteryuX/arcface-tf2/blob/master/modules/models.py#L90-L94))
- The `bin_lut_loss` is the name of binarization look up table (LUT) loss used in training. (tanh,sigmoid, or none)
- The `hidden_layer_remark` means how many hidden layers used. (possible value: 1,2,3,T1,T2)
- The `code_balance_loss` means using code balance loss on trainging or not.
- The `quanti` means using quantization loss on trainging or not.
- The `loss_fun` is the name of training loss used in traning. (possible value:batch_hard_triplet,batch_all_triplet_loss,batch_all_arc_triplet_loss,batch_hard_arc_triplet_loss,semihard_triplet_loss )
- The `triplet_margin` is the name of outputs directory used in checkpoints and logs folder. (make sure of setting it unique to other models)
- The `q` q in IoM
- The `m` m in IoM

### Training

Stage 1: Here have two modes for training the arcface your model, which should be perform the same results at the end.
```bash
# traning with tf.GradientTape(), great for debugging.
python train.py --mode="eager_tf" --cfg_path="./configs/config_arc/arc_res50.yaml"

# training with model.fit().
python train.py --mode="fit" --cfg_path="./configs/config_arc/arc_res50.yaml"
```
Stage 2: Training the deep IoM:

```bash
# traning with tf.GradientTape(),For deep IoM, can only train by eager_tf
nohup python -u train_twostage_tripletloss_online.py --cfg_path ./configs/config_10/iom_res50_twostage_1layer_hard_arcloss_256x8_0.yaml >1layer_hard_arcloss_256x8_0.log & 


### Testing the performance of deep IoM

```bash
python  test_twostage_iom.py --cfg_path ./configs/config_10/iom_res50_twostage_1layer_hard_arcloss_256x8_0.yaml 
```
#IJBC-Evaluation
Please run `IJB_11.py` first, then run `IJB_1N.py `secondly.


# Using-InsightFace-pre_build-model
In this work, we also try to adopt the original pre-build model by InsightFace team. However, their original model is trained on Mxnet, which is not fit tensorflow directly. Hence we perform the model conversion firstly to generate a tensorflow model. 

We adopted their ResNet100 model, the original performance is:

<table><thead><tr><th>Method</th><th>LFW(%)</th><th>CFP-FP(%)</th><th>AgeDB-30(%)</th><th>MegaFace(%)</th></tr></thead><tbody><tr><td>Ours</td><td>99.77</td><td>98.27</td><td>98.28</td><td>98.47</td></tr></tbody></table>

While after the model conversion, the generated TF2 model performance is:

<table><thead><tr><th></th><th>LFW</th><th>AgeDB30</th><th>CFP - FP</th></tr></thead><tbody><tr><td>Accuracy</td><td>0.9960</td><td>0.9752</td><td>0.9643</td></tr><tr><td>EER</td><td>0.0040</td><td>0.0305</td><td>0.0387</td></tr><tr><td>AUC</td><td>0.9987</td><td>0.9900</td><td>0.9877</td></tr><tr><td>Threshold</td><td>0.7340</td><td>0.7710</td><td>0.8320</td></tr></tbody></table>

There is a slightly accuracy drop, but it is still better than our own trained model.

To use this pre-build model, just set the **backbone_type** in the config file as Insight_ResNet100:

```
batch_size: 16
eval_batch_size: 16
input_size: 112
embd_shape: 512
sub_name: 'arc_Insight_ResNet100'
backbone_type: 'Insight_ResNet100' # 'ResNet50', 'MobileNetV2'
head_type: ArcHead # 'ArcHead', 'NormHead'
is_ccrop: False # central-cropping or not
```

Please note that the weight file is required, it is stored in `pre_models/resnet100/resnet100.npy`

The weight file and other related files can be downloaded from [this link](https://drive.google.com/file/d/1aOy12NnkEBmzLa9atQQCAlKiRO8zck49/view?usp=sharing).

## References
