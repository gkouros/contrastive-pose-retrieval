# contrastive-pose-retrieval
Code for paper "Category-Level Pose Retrieval with Contrastive Features Learnt with Occlusion Augmentation"

## Contents
- **scripts**: scripts for training, evaluation, and data visualization.
- **src**:
    - **losses**: Contains the custom Weighted Margin loss.
    - **samplers**: Custom samplers.
    - **dataloader**: Contains code for data loading, processing, and rendering.
    - **pipeline**: Utility functions for creating a training pipeline. Example usage in [train.py](scripts/train.py).


## Setup conda environment

- Install [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

- Create and activate a new conda environment:
```shell
conda create -n myenv
conda activate myenv
```

- Install pytorch related packages:
```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```

- Install dependencies:
```shell
pip3 install -r /path/to/requirements.txt
```

## Download dataset(s)
Download NeMo for downloading and preparing PASCAL3D+ and OccludedPASCAL3D+.
This process requires cuda and takes a while to finish. The datasets will be
in "./NeMo/data/", so move them to your dataset folder for convenience.
PASCAL3D+_release1.1 is the original PASCAL3D+ dataset, PASCAL3D_train_NeMo
is the training set, PASCAL3D_NeMo is the test set and PASCAL3D_OCC_NeMo is the occluded testset.
```shell
git clone "https://github.com/Angtian/NeMo.git"
cd NeMo
chmod +x PrepareData.sh
./PrepareData.sh
```

Download PASCAL VOC 2012 for synthetic-occlusion data augmentation
```shell
wget "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
tar -xf "VOCtrainval_11-May-2012.tar"
```

## Generate datasets of renderings
Generate rendering counterparts for train and test splits of PASCAL3D+.
The script will generate rendering, silhouette, depth, and normal images for the
train and test splits, located on PASCAL_(train_)NeMo/renderings/. The script
takes a few hours at most to generate around 11k rendering per split.
```shell
python3 scripts/generate_pascal3d_renderings.py \
    --split="all" \
    --from_scratch="False" \
    --object_category="car" \
    --positive_type='all' \
    --root_dir="/path/to/datasets" \
    --downsample_rate="2" \
```
You can use the following command to inspect the images:
```shell
python3 scripts/visualize_pascal3d.py \
    --root_dir="/path/to/datasets"
    --symmetric \
    --split=train \
    --positive=normals
```

Now generate the large database of 889k renderings for one rendering type
e.g. normals. This can take over 1-3 days depending on GPU. The images will be
saved in an hdf5 file to avoid saving almost a million images in a single folder
thus increasing file access speed.
```shell
python3 scripts/generate_pascal3d_renderings.py \
    --split="db" \
    --from_scratch="False" \
    --object_category="car" \
    --positive_type='normals' \
    --root_dir="/path/to/datasets" \
    --downsample_rate="2" \
    --nouse_fixed_cad_model \
    --db_name='db'
```
To visualize the images use same script as before but add the argument
"--positive_from_db".
