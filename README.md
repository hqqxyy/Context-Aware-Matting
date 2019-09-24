# Inference codes of Context-aware Image Matting for Simultaneous Foreground and Alpha Estimation

This is the official inference codes of [paper](https://arxiv.org/abs/1909.09725). 

## Environments
System: Ubuntu

Tensorflow version: tf1.8 or tf1.13

GPU memory: >= 12G

System RAM: >= 64G

## Prepare

1, Clone Context-aware Matting repository
```shell
git clone https://github.com/hqqxyy/Context-Aware-Matting.git
``` 

2, Download our models at [here](http://web.cecs.pdx.edu/~qiqi2/files/papers/conmat/files/model.tgz). Unzip them and move it to 
root of this repository.

```shell
tar -xvf model.tgz
```

After moving, it should be like
```shell
.
├── conmat
│   ├── common.py
│   ├── core
│   ├── demo.py
│   ├── model.py
│   └── utils
├── examples
│   ├── img
│   └── trimap
├── model
│   ├── lap
│   ├── lap_fea_da
│   └── lap_fea_da_color
└── README.md
```


## Run

You can first set the image and trimap path by:
```bash
export IMAGEPATH=./examples/img/2848300_93d0d3a063_o.png
export TRIMAPPATH=./examples/trimap/2848300_93d0d3a063_o.png
```

For the model(3) ME+CE+lap in the paper,
```bash
python conmat/demo.py \
--checkpoint=./model/lap/model.ckpt \
--vis_logdir=./log/lap/ \
--fgpath=$IMAGEPATH \
--trimappath=$TRIMAPPATH \
--model_parallelism=True
```

You can find the result at `./log/`


For the model(5) ME+CE+lap+fea+DA in the paper. 
```bash
python conmat/demo.py \
--checkpoint=./model/lap_fea_da/model.ckpt \
--vis_logdir=./log/lap_fea_da/ \
--fgpath=$IMAGEPATH \
--trimappath=$TRIMAPPATH \
--model_parallelism=True
```
You can find the result at `./log/`

For the model(7) ME+CE+lap+fea+color+DA in the paper. 
```bash
python conmat/demo.py \
--checkpoint=./model/lap_fea_da_color/model.ckpt \
--vis_logdir=./log/lap_fea_da_color/ \
--fgpath=$IMAGEPATH \
--trimappath=$TRIMAPPATH \
--branch_vis=1 \
--branch_vis=1 \
--model_parallelism=True
```
You can find the result at `./log/`


###Note
Please note that since the input image is high resolution. You might need to use gpu whose memory 
is bigger or equal to 12G. You can set the `--model_parallelism=True` in order to further save the GPU memory. 

If you still meet problems, you can run the codes in CPU by 
```bash
export CUDA_VISIBLE_DEVICES=''
```
, and you may need to set  `--model_parallelism=False`. 

Or you can resize the image and trimap to a smaller size and change the `vis_comp_crop_size`  and `vis_patch_crop_size` accordingly. 



## Results
We also provide the our results of Compisition-1k dataset and the real-world image dataset. You can download them at
[here](http://web.cecs.pdx.edu/~qiqi2/files/papers/conmat/files/result.tgz).


If you find this code is helpful, please consider to cite our paper. It is very important to us.


If you find any bugs of the code, feel free to send me an email: qiqi2 AT pdx DOT edu. You can find more information in my 
[homepage](http://web.cecs.pdx.edu/~qiqi2/)

## Reference
Deeplab: https://github.com/tensorflow/models/tree/master/research/deeplab
