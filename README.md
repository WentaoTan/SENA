# SENA
Code for Beyond Human Data: Aligning Multimodal Large Language Models by Iterative Self-Evolution (AAAI 2025)

## Install
Our code is built on [Seva](https://github.com/Kevinz-code/SeVa).
```
pip install torch==2.0.1 torchvision==0.15.2
pip install -e .
```

## Dataset
We expect the **image dataset** to have the following structure:
```
data/
|-- texvqa/
|---- train_images/
......
|-- ocrvqa/
|---- images/
......
|-- coco2014/
|---- val2014/
```
We use the 665K pre-training data of LLaVA-1.5 for training. Please download [COCO](http://images.cocodataset.org/zips/train2017.zip), [GQA](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip), [OCR-VQA](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), [TextVQA](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip), [VisualGenome_part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [VisualGenome_part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). 

We have included a **detailed DPO data construction pipeline** in `data/` folder, with *step1*, *step2* and *step3*. Refer to [README](data/README.md)
```
data/
|-- step1/
|-- step2/
|-- step3/
|-- README.md
```

## Training
You need to first download weights of [LLaVA-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b). 

For running with DPO data:
```
sh run/run.sh
```

## Acknowledgement
This repo is based on [Seva](https://github.com/Kevinz-code/SeVa) and [LLaVA](https://github.com/haotian-liu/LLaVA). We thank their efforts in building their codebase. 

## Citation
If you find our paper or codebase useful, please consider cite
```
@article{tan2024beyond,
  title={Beyond Human Data: Aligning Multimodal Large Language Models by Iterative Self-Evolution},
  author={Tan, Wentao and Cao, Qiong and Zhan, Yibing and Xue, Chao and Ding, Changxing},
  journal={arXiv preprint arXiv:2412.15650},
  year={2024}
}
```

### Contact
Email: ftwentaotan@mail.scut.edu.cn or 731584671@qq.com

如果可以当然还是希望用中文contact我啦！
