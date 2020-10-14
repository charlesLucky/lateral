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

```console
(base) xingbo@***:~/lateral/data/SESSION_TENT_NEW$ ls
SESSION1_LT  SESSION1_ST  SESSION2  SESSION3  SESSION4
```

To generate proper images which can be used directly in training, we employ the same pre-processing protocol as Petr[?] proposed. 

To segment the fish from the background, images are firstly transformed into HSV color space, green colour was detected and used to detect the background area. The fish was detected as the largest object inside the background. Finally the detected fish was rotated based on automatically estimated ellipse around the fish.

However, the segmentation of the whole fish is challenging due to semitransparent tail fin, crop a portion of the fish body will be a good option to generate stable and consistent images for model training. The ROI crop area is determined as the area between eye position (EP) and upper fin (UP) beginning in horizontal direction and between upper fin beginning + belly point (BP)/20 and BP/2 in vertical direction. 


Deep neural network can learn very complex model from given training dataset. However, the training will tend to be overfitting on small dataset.  Data augmentation is a very simple and effective method against overfitting and helps the model generalize better.

In order to make the most of our few training examples, we will augment captured fish images via a number of random transformations, including random cropping, horizontal random flipping,random saturation and brightness. See `modules/dataset.py`:

```python

def loadTrainDS(test_data_dir, BATCH_SIZE=64, cfg=None):
    def get_label(file_path):
        parts = tf.strings.split(file_path, '/')
        label = tf.strings.to_number(parts[-2], out_type=tf.dtypes.int64)
        return label

    def _transform_images(is_ccrop=False, cfg=None):
        def transform_images(x_train):
            x_train = tf.image.resize(x_train, (cfg['input_size_w'] + 20, cfg['input_size_h'] + 20))
            x_train = tf.image.random_crop(x_train, (cfg['input_size_w'], cfg['input_size_h'], 3))
            x_train = tf.image.random_flip_left_right(x_train)
            x_train = tf.image.random_saturation(x_train, 0.6, 0.8)
            x_train = tf.image.random_brightness(x_train, 0.4)
            x_train = tf.image.random_contrast(x_train, 0.2, 0.5)
            x_train =  x_train / 255 # new add
            return x_train

        return transform_images

    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        image_encoded = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(image_encoded, channels=3)
        img = _transform_images(cfg=cfg)(img)
        return img, label

    list_ds = tf.data.Dataset.list_files(str(test_data_dir + '*/*')).repeat()

    # print(f"we have total {self.image_count} images in this folder")
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = labeled_ds.batch(BATCH_SIZE)
    dataset = dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

```

To prepare the training set and testing dataset, 20 IDs out of 10 in the long-term session are taken out to join the short-term session fish, to form the training set:

```python
def generateDatasetBySplitingIdentity(directory_str, train_list, train_dir, test_dir, pre='sess1'):
    g = os.walk(directory_str)
    for path, dir_list, file_list in g:
        for id_dir in dir_list:
            if train_list.__contains__(
                    id_dir):  # in train set, maybe the filename will be same, hence we need to rename each session
                data_dir = pathlib.Path(os.path.join(path, id_dir))
                pic_list = list(data_dir.glob('*.png'))
                for images in pic_list:
                    # print(images)
                    dst_dir = os.path.join(train_dir, id_dir)
                    check_folder(dst_dir)
                    head, tail = os.path.split(images)
                    finalpath = os.path.join(dst_dir, pre + tail)
                    copy(images, finalpath)

            else:
                data_dir = pathlib.Path(os.path.join(path, id_dir))
                pic_list = list(data_dir.glob('*.png'))
                for images in pic_list:
                    dst_dir = os.path.join(test_dir, str.split(directory_str, '/')[-1], id_dir)
                    check_folder(dst_dir)
                    copy(images, dst_dir)

generateDataset('SESSION_TENT_NEW',byIDorByImages=True,
                        train_weight=0.67)  # half as train and half as test  0.67-> 20 as train 10 as test
```

Just run the `prepare_ds.py`, it will split the original dataset and generate a new directory contains training images and test images:

```python
(base) xingbo@xingbo-pc:/media/xingbo/Storage/lateral$ python prepare_ds.py  --stage 1
[*] remark: SESSION_TENT_NEW
['D50EBB', '9A7933', 'FD0901', '550253', 'D12372', '0F571E', '54EBF6', 'D03E9F', 'A2F79F', 'F987F9', '4D8B97', '9AFEF7', 'FCEAB5', 'DA0B56', '440270', '54B9F0', 'FCF1EF', 'FCDCB7', 'AC0F15', '55F1EB']
above IDs are used in training, remianing will be used for testing, in dir tmp_tent/test/*

```

It will generate below data dir:
```python
.
├── test
│ ├── SESSION1_LT
│ ├── SESSION2
│ ├── SESSION3
│ └── SESSION4
└── train
    ├── 001
    ├── 002
    ├── 003
  .....

```

The DS loader will load train dir, while the test dir will be used to test the performance.

In a nutshell, it will take 319 IDs for training, and remaining 10 IDs for testing.

## Training and Testing

### To train the model
You can modify your own dataset path or other settings of model in [./configs/*.yaml]() for training and testing, which like below.

```python
# general (shared both in training and testing)
batch_size: 20
input_size_w: 75
input_size_h: 320
embd_shape: 512
sub_name: 'VGG16'
backbone_type: 'VGG16' # 'ResNet50', 'MobileNetV2' VGG16
head_type: NormHead # 'NormHead'
is_ccrop: True # central-cropping or not
rnn: False
# train
num_classes: 319
#1848
num_samples: 10000
epochs: 100
base_lr: 0.01
w_decay: !!float 5e-4
save_steps: 10000

```

Note:
- The `sub_name` is the name of outputs directory used in checkpoints and logs folder. (make sure of setting it unique to other models)
- The `backbone_type` is the name of backbone wish to use, can be VGG16, VGG19, ResNet50.
- The `num_classes` is the number of class in the training set. 

### To train the model in the second stage

The model (checkpoints file) of the first stage shall be stored under [checkpoints/VGG16/*]()

You can modify settings of model in [./config_*/*_2ed_stage.yaml]() for training, which like below.

```python
batch_size: 20
input_size_w: 75
input_size_h: 320
embd_shape: 512
sub_name: 'VGG16_2ed'
1st_sub_name: 'VGG16'
backbone_type: 'VGG16' # 'ResNet50', 'MobileNetV2' VGG16
head_type: NormHead # 'NormHead'
is_ccrop: True # central-cropping or not
rnn: False
# train
num_classes: 319
#1848
num_samples: 10000
epochs: 100
base_lr: 0.01
w_decay: !!float 5e-4
save_steps: 10000

```

Note:
- The `sub_name` is the name of outputs directory used in checkpoints and logs folder. (make sure of setting it unique to other models)
- The `1st_sub_name` is model save dir used in the first stage. 


### Training

Stage 1: Training the model based on 319 IDs.
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
