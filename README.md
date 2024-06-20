# Edge-preserving image deraining network using cumulative feature aggregation 
Abstract
---
This study proposes an edge-preserving image deraining network using a wavelet feature aggregation method. Wavelet subbands are correlated with each other, and the high-frequency subband in the horizontal direction is the least affected by rain streak contamination. On this basis, we introduce a single image deraining network that cumulatively aggregates wavelet subband features according to their importance. The network architecture primarily comprises a wavelet feature aggregation block and a residue channel guide block. The aggregation of features with the cumulative wavelet feature aggregation block moves downward and upward, and a long short-term memory-based multiscale attentive rain streak removal block is developed to serve as the backbone for rain streak removal. We use a residual channel map based on the low-frequency subband to construct guide features that assist in rain streak removal. A repetitive image restoration framework that incorporates two proposed blocks is used to iteratively improve rainy images. We test the proposed network on various image datasets and compare the deraining performance with those of existing methods. The experimental results demonstrate that the performance of the proposed scheme is superior to that of other tested deraining methods.

Requirements
---

    Python 3.6, torch 0.4.1, torchvision 0.2.0. Other details are in the requirements.txt.

Datasets
---
__Synthetic datasets__

+ Rain100L : 200 training pairs and 100 testing pairs
+ Rain100H : 1800 training pairs and 100 testing pairs
+ Rain200L : 1800 training pairs and 200 testing pairs
+ Rain200H : 1800 training pairs and 200 testing pairs
+ Rain800 : 700 training pairs and 100 testing pairs
+ Rain1200 : 12000 training pairs and 1200 testing pairs

__Real-world datasets__

+ GT-rain : 26124 training pairs and 2100 testing pairs
+ LHP-rain : 2100 training pairs and 300 testing pairs

Training
---
1. Set training and testing dataset's paths in the Rainheavy, Rainheavytest in the data folder, respectively.
2. Run python main.py.
---

    python main.py --save cwfanet(100H) --model model_cwfanet --patch_size 64 --epochs 300 --batch_size 16 --n_feats 32 --lr 5e-4 --data_train RainHeavy --data_test RainHeavyTest --data_range 1-1800/1-100 --loss 1*MSE+0.2*SSIM

Testing
---
1. Set testing dataset's path in the Rainheavytest in the data folder and set the path of the pretrained model.
2. Run python main.py.
---

    python main.py --save cwfanet_test(100H) --model model_cwfanet --data_test RainHeavyTest --data_range 1-1800/1-100 --pre_train '---pretrained model path---' --test_only


The results can be executed in a Matlab file to produce PSNR and SSIM results.
    
Citation
---

So Young Choi, Su Yeon Park, Il Kyu Eom,  
Edge-preserving image deraining network using cumulative feature aggregation,  
Applied Soft Computing,  
2024,  
https://doi.org/10.1016/j.asoc.2024.111887

Contact
---
If you have any questions, please contact sso8215@pusan.ac.kr.
