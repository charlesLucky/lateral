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
# traning with tf.GradientTape()
(base) xingbo@* ~/lateral$ python Train_DS.py --cfg_path ./configs/VGG16_1st.yaml --rename 1

```
Stage 2: Testing with the remaining 10 IDs:

In reality, given a fish, multiple images of one fish can be captured and used for identification. To make full use of those available information, all images of this fish are used to predict the label, then the mode of those predicted labels will be used as the final predicted label. 

Execute below command:

```python
(base) xingbo@* ~/lateral$ python Test_DS_2ed.py --cfg_path ./configs/VGG16_2ed.yaml
```

The accuracy is shown as below:

| VGG16 | SL1 | SL2 | SL3  | SL4   |
|-------|-----|-----|------|-------|
| SL1   |   1 | 0.7 |  0.3 |   0.2 |
| SL2   |     |   1 | 0.65 | 0.275 |
| SL3   |     |     |    1 |  0.45 |
| SL4   |     |     |      |     1 |


## References
