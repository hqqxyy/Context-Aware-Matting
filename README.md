# Inference codes of Context-aware Image Matting for Simultaneous Foreground and Alpha Estimation

This is the official inference codes of Context-aware Image Matting for Simultaneous Foreground and Alpha Estimation 
[arxiv](https://arxiv.org/abs/1909.09725) using Tensorflow. Given an image and its trimap, it will predict the alpha matte
and foreground color. 

<a href="https://arxiv.org/pdf/1909.09725" rel="Paper"><img src="http://web.cecs.pdx.edu/~qiqi2/files/papers/conmat/demo.jpg" alt="Paper" width="100%"></a>

## Setup

### Requirements
System: Ubuntu

Tensorflow version: tf1.8, tf1.12 and tf1.13 (It might also work for other versions.)

GPU memory: >= 12G

System RAM: >= 64G


### Download codes and models

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


### Note
Please note that since the input image is high resolution. You might need to use gpu whose memory 
is bigger or equal to 12G. You can set the `--model_parallelism=True` in order to further save the GPU memory. 

If you still meet problems, you can run the codes in CPU by disable GPU
```bash
export CUDA_VISIBLE_DEVICES=''
```
, and you need to set  `--model_parallelism=False`.  Otherwise, you can resize the image and trimap to a smaller size 
and then change the `vis_comp_crop_size`  and `vis_patch_crop_size` accordingly. 



## Results
We also provide the our results of Compisition-1k dataset and the real-world image dataset. You can download them at
[here](http://web.cecs.pdx.edu/~qiqi2/files/papers/conmat/files/result.tgz).


## License
The provided implementation is strictly for academic purposes only. 
Should you be interested in using our technology for any commercial use, please feel free to contact us.


 
If you find this code is helpful, please consider to cite our paper.
```
@article{hou2019context,
  title={Context-Aware Image Matting for Simultaneous Foreground and Alpha Estimation},
  author={Hou, Qiqi and Liu, Feng},
  journal={arXiv preprint arXiv:1909.09725},
  year={2019}
}
```

If you find any bugs of the code, feel free to send me an email: qiqi2 AT pdx DOT edu. You can find more information in my 
[homepage](http://web.cecs.pdx.edu/~qiqi2/).



## Acknowledgments
The source images in the demo figure are used under a Creative Commons license from Flickr users Robbie Sproule, 
MEGA PISTOLO and Jeff Latimer. The background images are from the MS-COCO dataset. The images in the examples are from 
 Composition-1k dataset and the real-world image. 


