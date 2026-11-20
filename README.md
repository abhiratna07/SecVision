SecVision: Adversarial Robustness Benchmark & Defense in CIFAR-10

-AIM-: To create a framework that provides benchmark results for the accuracy of ResNet-18 and CLIP models by implementing Fast Gradient Sign Method(FGSM) and Project Gradient Descent(PGD) attacks on the CIFAR-10 dataset, followed by adversarialy training the ResNet-18 model

-OVERVIEW-: Modern vision architecture models such as ResNet-18 and CLIP(Contrastive Language-Image Pre-Training) are trained on the images from the CIFAR-10 dataset consisting of 50000 images. However the introduction of white box attacks like FGSM and PGD decreases the accuracy of these models by a drastic amount, thus in order to improve the accuracy against perturbed inputs the models are adversarialy trained against these perturbed images only.

----------------BENCHAMARK RESULTS--------------------------

Epsilon(Perturbatuion) | Attack | Baseline Resnet-18 | CLIP  | PGD Trained Resnet-18 | Gain in accuracy(compared to Baseline)
0/255                  |NONE    |93.71%              |87.82% |81.29%                 |-12.42%
2/255                  |PGD     |32.36%              |27.13% |75.27%                 |+42.91%
2/255                  |FGSM    |7.01%               |3.48%  |75.06%                 |+68.05%
4/255                  |PGD     |18.44%              |23.88% |68.33%                 |+49.89%
4/255                  |FGSM    |0.03%               |0.25%  |67.09%                 |+67.07%
8/255                  |PGD     |11.70%              |22.85% |55.55%                 |+43.85%
8/255                  |FGSM    |0.00%               |0.01%  |48.96%                 |+48.96% \*\*\*
16/255                 |PGD     |8.90%               |25.18% |34.74%                 |+25.79%
16/255                 |FGSM    |0.00%               |0.00%  |18.19%                 |+18.19%

-DISCUSSION
1-Strong Foundational Accuracy does not guarantee robustness against White Box Attacks- Resnet-18 which demostrated 93.71% accuracy and CLIP which demonstrated 87.82% accuracy both collapsed to 0.00% and 0.01% accuracy when exposed to PGD-20 attacks
2-Adversarial Training acts as a robust method to provide accuracy gain by adversarialy training the Resnet-18 model, lifting its accuracy at epsilon=8/255 from 11.70% to 55.55% under FGSM attack  
3-Loss of robustness occurs due to adversarial training-The accuracy decreases from 93.71% in Baseline Resnet to 81.29% in PGD trained Resnet-18 indicating a decrease of 12.42% in accuracy which occurs due to the suppression by the perturbed images during training
