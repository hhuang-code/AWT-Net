### Dataset
* Create a `data` folder under the `AWT-Net` folder: `mkdir data`.
* __ModelNet40__:
	Download and unzip [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) to the `data` folder.
* __ScanObjectNN__:
	1. Create a `ScanObjectNN` folder under the `data` folder: `mkdir ScanObjectNN`. 
	2. Download and unzip [h5_file](https://hkust-vgd.github.io/scanobjectnn/) to the `ScanObjectNN` folder.
* __ShapeNet Part__:
	1. Download and unzip [ShapeNet Part](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) to the `data` folder.
	2. [Optional for visualization] Download [part_color_mapping.json](https://github.com/FENGGENYU/PartNet/blob/master/part_color_mapping.json) into the `data/shapenetcore_partanno_segmentation_benchmark_v0_normal` folder.

### Test the pretrained models
* __ModelNet40__: 
	Accuracy: `cd classification` and `bash scripts/eval.sh`.
* __ScanObjectNN__:
	1. OBJ_ONLY: `cd classification`; set `exp_name` to`pretrained/sonn_obj`; `bash scripts/sonn_eval.sh`.
	2. OBJ_BG: `cd classification`; set `exp_name` to`pretrained/sonn_bg`; append `--with_bg` to the python command; `bash scripts/sonn_eval.sh`.
* __ShapeNet Part__: 
	1. Instance mIoU: `cd segmentation`;  set `model_type` to `insiou` in `scripts/eval.sh`; `bash scripts/eval.sh`.
	1. Class mIou:  `cd segmentation`;  set `model_type` to `clsiou` in `scripts/eval.sh`; `bash scripts/eval.sh`.

### Train from scratch
* __ModelNet40__: 
	Run: `cd classification` and `bash scripts/main.sh`.
* __ScanObjectNN__:
	1. OBJ_ONLY: `cd classification` and `bash scripts/sonn_main.sh`.
	2. OBJ_BG: `cd classification`; append `--with_bg` to the python command; `bash scripts/sonn_main.sh`.
* __ShapeNet Part__:
	Run: `cd segmentation` and `bash scripts/main.sh`.
	
### Performance and reproducibility
* We train our model on __ModelNet40__ and  __ScanObjectNN__ using two V100 GPUs with 32GB memory per GPU, and train on __ShapeNet Part__ using four V100 GPUs.
* The accuracy of the pretrained models for __ModelNet40__ is 93.8%, for __ScanObjectNN__ are 91.1% (OBJ_ONLY) and 89.6% (OBJ_BG). The mIoU for __ShapeNet Part__ are 86.5% (instance) and 85.0% (class).
* Due to the random assignment procedure in our method (Line 1 in Algorithm 1), the complete reproducibility are not guaranteed. Using the same hardwares, -0.5%/~+0.5% deviation for the classification tasks and -0.2%/~+0.2% deviation for the part segmentation task are in expectation.
* The factors including but not limited to: the random assignment procedure in the method, the nature of datasets, the hardware platform, and the release version of PyTorch/Python, may cause the results not to be completely reproducible. However, the results are still expected to be within a reasonable small deviation.

### Acknowledgment
This code is partially borrowed from [GDA-Net](https://github.com/mutianxu/GDANet), [DGCNN](https://github.com/WangYueFt/dgcnn), [GCN](https://github.com/tkipf/gcn), [Transformer](https://d2l.ai/chapter_attention-mechanisms/transformer.html), [PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch). Detailed credits are listed on the top of each .py file (if applicable).
