# Vector Quantized VAE: Image Reconstruction <br>(Implementation in PyTorch)
![image](https://github.com/user-attachments/assets/c1edc415-37a4-4754-8d35-b621ed789870)
This repository contains my PyTorch implementation of the Vector Quantized Variational Auto Encoder as described by van den Oord et al in *Neural Discrete Representation Learning* applied to images (source of the architecture figure). The implementation was built according to the description in the paper. 

The implementation is demonstrated by reconstructing on 3 different datasets. It is also shown how embeddings can be extracted, and how these can be used for training a prediction model.

![1](https://github.com/user-attachments/assets/17848f6f-7924-42a2-926e-cc9e13c6a03e)

*Animation of the progression of image reconstruction during training: shows a reconstructed validation sample for each epoch while training*

## Roadmap
I plan to work on the following features in the coming time, listed by order:
- Implement hierarchical latent representations to model image information at several scales simultaneously.
- Replace convolution based encoder/decoders by vision transformers
- allow model to directly reconstruct 3D data.

  
## Introduction
The Vector Quantized VAE is a variation of the Variational Auto Encoder, where the latent space consists of a limited amount of discrete embeddings, which are "quantized" to, using a closest neighbour search. These discrete embeddings are learned during training, and result in easier training, less likelyhood of posterior-collapse and often sharper looking reconstructions.


A downside of having these discrete quantized vectors as the latent space is that there is no normally distributed latent space where we can sample new valid samples from. See "improvement" section how that could be mitigated.

My actual motivation for this implementation is that i want to train such a VQ-VAE on large amount of medical images, and then use the feature extraction capabilities of the encoder to generate meaningfull encodings to improve model performance for prediction problems with litle available labelled data.

The training loop can be found in `train.py`, and the implementation of the VQ-VAE can be found in `model/vq_vae.py`

## Universal embeddings: extraction and prediction demos:
A demonstration of how a trained VQ VAE model can be used to first extract usefull features, and to then use these extracted features in a prediction model, can be found in `extract_features.py` and `predict_on_embeddings.py`.


## Demonstration: Reconstruction on 3 different datasets
The effectivity of the implementation was shown on three different datasets: MNIST, SLT10 and a Pneumonia Chest X-Ray dataset. The datasets were chosen to cover a variaty of resolutions, levels of image diversity, and presence of color.

### Pneumonia Chest X-Ray dataset
The following results were obtained on the dataset as publised in (*Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*
Kermany, Daniel S. et al.) The dataset contains a total of 5,863 images, with a resolution of 256x256, where each image is labelled with the presence or abscence of Pneumonia. see `train_x_ray.py` for the used script and hyperparameters.
#### X-Ray: Validation Reconstruction

<img src="figures/xray.png" width="700">

![alt text](figures/xray_loss.png)

### SLT10
The STL-10 dataset is an image recognition dataset for developing unsupervised feature learning, were a very large set of unlabeled examples is provided with a resolution of 96x96. see `train_SLT10.py` for the used script and hyperparameters.

#### SLT10: Validation Reconstruction
<img src="figures/slt10.png" width="700">

![alt text](figures/slt10_loss.png)

### MNIST
MNIST is handwritten digit recognition dataset containing 70,000 grayscale images of handwritten digits, sized 28x28 pixels. see `train_MNIST.py` for the used script and hyperparameters.

<img src="figures/mnist.png" width="700">

![alt text](figures/mnist_loss.png)

## Future steps/ improvements

### Deep Feature Encoding for Medical Data
  My actual goal in creating this implementation, is to use the embeddings as created by this model for feature extraction of medical images, improving model performance in cases were few labelled, but plenty of unlabelled images are available, which is often the case. This will be the next step. I want to show the validity of this approach by implementation.

### Hierarchical VQ-VAE
Hierarchical VQ-VAE is an extension of the VQ-VAE architecture, where different latent spaces are learned at different scales of the image, effectively retaining information at all scales. In practice this VASTLY improves the amount of detail in the reconstruction.

### PixelCNN for sampling new data
Since the latent space in VQ VAE consists of discrete codes, there is no normally distributed latent space where we can sample new valid samples from. However PixelCNN, a separate autoregressive model can be trained to effectvily generate new meaningfull samples.

