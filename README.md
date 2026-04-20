# SA-Conv
Code for " Aligning Network Equivariance with Data Symmetry: A Theoretical Framework and Adaptive Approach for Image Restoration".

<!-- ![](https://github.com/tanfy929/SA-Conv/blob/main/Sample-Adaptive Equivariant Network.png) -->

## 🧠 Abstract
Image restoration is an inherently ill posed inverse problem. Equivariant networks that embed geometric symmetry priors can mitigate this ill posedness and improve performance.
In this work, we conduct an analysis from an optimization perspective and formalize the intrinsic relationship among data symmetry priors, model equivariance, and generalization capability. Guided by proposed theoretical framework, we propose a Sample-Adaptive Equivariant Network that utilizes a hypernetwork and transformation learnable equivariant convolutions to dynamically align with the inherent symmetry of each individual sample. Extensive experiments on super-resolution, denoising (across natural and remote sensing images), and deraining validate our theoretical findings and demonstrate the significant superiority of our method over both standard baselines and traditional equivariant models. 

<p align = "center">  
    <img src="https://github.com/tanfy929/ImageFolder/blob/main/Sample-Adaptive%20Equivariant%20Network.png?raw=true"/>
</p>

## 📦 Pretrained Models
All model checkpoints are available in the **[Releases](https://github.com/tanfy929/SA-Conv/releases)** of this repository. You can download the corresponding `.pt` files for super-resolution, denoising, and deraining to reproduce the results or run inference directly.

## 📂 SR
Codes for Image Super-Resolution. 

The main directory structure is organized as follows:
```
src/: Contains all the source code.

src/model/: Contains the model implementations for all the compared methods mentioned in our paper. Interested readers can easily train and evaluate these comparison models by simply modifying the --model argument in the training commands. It is important to note that the number of channels in a standard CNN should be set to the corresponding multiple of tranNum used in the equivariant methods.
```


🚀 Taking the proposed method (models with the '_SAconv' suffix) as an example, you can use the following commands to train and test the models.

**EDSR and its variants**
```
--train-- 
CUDA_VISIBLE_DEVICES=6 python main_train2.py --model edsr_SAconv --scale 2 --save edsr_SAconv --n_resblocks 16 --n_feats 32 --res_scale 0.1 --epoch 150 --decay 100 --patch_size 96 --kernel_size 5 --tranNum 8 --lr 3e-4 --device 0

--test--  
python main.py --model edsr_SAconv --scale 2 --pre_train ../experiment/edsr_SAconv/model/model_best.pt --save ../experiment/edsr_SAconv/ --n_resblocks 16 --n_feats 32 --res_scale 0.1 --kernel_size 5 --tranNum 8 --test_only True --device 6
```
**RDN and its variants**
```
--train-- 
CUDA_VISIBLE_DEVICES=7 python main_train2.py --model rdn_SAconv--scale 2 --save rdn_SAconv --res_scale 0.1 --batch_size 16 --patch_size 64 --G0 8 --kernel_size 5 --epochs 150 --tranNum 8 --decay 3-100-130 --lr 3.2e-4 --device 0

--test--  
python main.py --pre_train ../experiment/rdn_SAconv/model/model_best.pt --save ../experiment/rdn_SAconv/ --model rdn_SAconv --scale 2 --res_scale 0.1 --ini_scale  0.1 --G0 8 --kernel_size 5 --device 7 --tranNum 8 --test_only True
```

**RCAN and its variants**
```
--train-- 
CUDA_VISIBLE_DEVICES=6 python main_train2.py --model rcan_SAconv --scale 2 --save rcan_SAconv --n_resblocks 16  --n_feats 8 --res_scale 0.1 --epochs 150 --decay 3-100-130 --patch_size 96 --kernel_size 5 --tranNum 8 --lr 4.2e-4 --device 0

--test-- 
python main.py --model rcan_SAconv --n_resblocks 16 --ini_scale 0.1 --n_feats 8 --res_scale 0.1 --kernel_size 5 --tranNum 8 --test_only True --scale 2 --pre_train ../experiment/rcan_SAconv/model/model_best.pt --save ../experiment/rcan_SAconv/ --device 6
```

## 📂 DeNoise
Codes for Image Denoising.

The main directory structure is organized as follows:
```
src/: Contains all the source code.

src/model/: Contains the model implementations for all the compared methods mentioned in our paper. Interested readers can easily train and evaluate these comparison models by simply modifying the --model argument in the training commands.
```


🚀 Taking the proposed method (edsr_SAconv) as an example, you can use the following commands to train and test the models.

```
--train-- 
CUDA_VISIBLE_DEVICES=6 python main.py --model edsr_SAconv --scale 1 --save edsr_SAconv --n_resblocks 16 --n_feats 32 --res_scale 0.1 --epoch 150 --decay 100 --patch_size 48 --kernel_size 5 --tranNum 8  --lr 2e-4 --device 0

--test--  
python main_test.py --model edsr_SAconv --scale 1 --pre_train ../experiment/edsr_SAconv/model/model_best.pt --save ../experiment/edsr_SAconv/ --kernel_size 5 --n_resblocks 16 --n_feats 32 --res_scale 0.1 --device 6
```


## 📂 DeRain
Codes for Single Image Rain Removal.

The main directory structure is organized as follows:
```
for_syn/src/: Contains all the source code.

for_syn/src/model/: Contains the model implementations for RCDNet and proposed SA-RCDNet.

for_syn/data/: Provides the training and testing datasets.

test_matlab/: Test code based on MATLAB.
```


🚀 Taking the proposed method (SA-RCDNet) as an example, you can use the following commands to train and test the models.

```
--train-- 
CUDA_VISIBLE_DEVICES=1 python main.py  --save rcdnet_SAconv --model rcdnet_SAconv --scale 2 --epochs 100 --batch_size 16 --patch_size 64 --data_train RainHeavy --n_threads 0 --data_test RainHeavyTest --stage 17 --lr_decay 45 --gamma 0.2 --num_M 32 --num_Z 32 --data_range 1-200/1-100 --loss 1*MSE --save_models --lr 5e-4 --device 0

--test--  
To maintain consistency with the evaluation protocols of prior work, we compute PSNR and SSIM using the same MATLAB code as in previous studies. Save the network-generated images to the 'test_matlab' folder and then run statistic.m.
```

💡 We will continue to update and refine both the codes and corresponding usage instructions. If you have any questions or would like to discuss further, please feel free to contact us 💌!

