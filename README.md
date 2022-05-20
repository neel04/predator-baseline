# Fork from @YassineYousfi

Simple modifications to improve parameters efficience by adopting pre-processing methods as well as a Fully-Pyramidal Network (FPN) decoder over a HRNet (High resolution Net) encoder to achieve finer-grained segmentations.

Turns out, after trying all the complex modifications the simplest one was simply applying pre-processing like `SLIC` (Simple Linear Iterative Clustering) for creating Superpixels and good ol' `Canny`, concatting features with the HRNet+FPN arrangement leads to much higher parameter efficiency that plain old effnet.        

![canny_example](https://user-images.githubusercontent.com/11617870/169622947-ae79ec17-654b-4801-b888-01c1211718e5.jpg)

This may prove to be a compute hindrance however, especially on edge devices if one's computing complex pre-processing in parallel unless its optimized. Thus, I don't concat Supexpixel output which doesn't help performance too much via metrics, but does help to smooth mask boundaries on experiments in repo. 

It's still not entirely clear why using smaller `efficient_net`s doesn't help with concatted representations. A cursory guess would be that the network simply doesn't propogate features across _all_ the channels over multiple-layers effectively which is an advantage of `HrNet`. Squeeze-and-excite blocks can also be introduced in the encoder, but I didn't test that for preserving parameters. 


### This repo can be run exactly as instructions below in the OG repository❗

# Bonus 🎄

This is the `VQ-VAE-2` reconstructions (compressing images to 32x32 **discrete** latent vector). Original image (left), reconstruction (right)

![image](https://user-images.githubusercontent.com/11617870/169623583-6898794b-4bd2-4c1c-9c30-80c1a61a97ab.png)

Nothing much, just wanted to show off its accuracy in re-constructing complex details and weird lighting too 😜

---
---

# 🚗 comma10k-baseline 

A semantic segmentation baseline using [@comma.ai](https://github.com/commaai)'s [comma10k dataset](https://github.com/commaai/comma10k).

Using U-Net with efficientnet encoder, this baseline reaches 0.044 validation loss.

## Visualize
Here is an example (randomly from the validation set, no cherry picking)
#### Ground truth 
![Ground truth](example.png)
#### Predicted
![Prediction](example_pred.png)

## Info 

The comma10k dataset is currently being labeled, stay tuned for:
- A retrained model when the dataset is released
- More features to use the model


## How to use
This baseline uses two stages (i) 437x582 (ii) 874x1164 (full resolution)
```
python3 train_lit_model.py --backbone efficientnet-b4 --version first-stage --gpus 2 --batch-size 28 --epochs 100 --height 437 --width 582
python3 train_lit_model.py --backbone efficientnet-b4 --version second-stage --gpus 2 --batch-size 7 --learning-rate 5e-5 --epochs 30 --height 874 --width 1164 --augmentation-level hard --seed-from-checkpoint .../efficientnet-b4/first-stage/checkpoints/last.ckpt
```

## WIP and ideas of contributions! 
- Update to pytorch lightning 1.0
- Try more image augmentations
- Pretrain on a larger driving dataset (make sure license is permissive)
- Try over sampling images with small or far objects


## Dependecies
Python 3.5+, pytorch 1.6+ and dependencies listed in requirements.txt.
