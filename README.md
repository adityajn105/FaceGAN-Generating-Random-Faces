# FaceGAN-Generating-Random-Faces
Generate Random Faces which do not exists. I trained a Deep Convolution GAN (DCGAN) on 100k celebrities photos.

[Fork the notebook in Colab](https://colab.research.google.com/drive/1uj5U_2Fgr5579oT_2wZCrmsfNGAk8yd_)

Training Progress           | Web App view            
:-------------------------:|:-------------------------:
![](screenshots/progress_animation.gif)  |  ![](screenshots/app_view3.png) 


## Getting Started
Here I will explain how to train the FaceGAN, also how to setup the app running in your own environment. Also I will try to explain basics of Deep Convolutional Generative Adversarial Network.

### Prerequisites
You will need Python 3.X.X with some packages which you can install direclty using requirements.txt.
> pip install -r requirements.txt

### Get the Dataset
I have used 100k celebrities dataset. You can download that dataset from this location [Link](https://www.kaggle.com/greg115/celebrities-100k).

### Train the model (Get new weights for generator or train further)
Run the train_gan.py file. Using the following flags:
* -w, --weights - saved weights directory
* -p, --photo_dir - directory containing face images
* -b, --batch_size - batch size
* -e, --epochs - number of epocs to run
* -i, --inp_dim - dimension of noise for generator
> python3 train_gan.py --weights ./saved_weights -p ./photos -b 32 -e 10000 -i 100 

### Running the app
Use the following command in the app directory to run the app.
> ``` python3 app.py ```

## DCGAN - Deep Convolutional Generative Adversarial Network
DCGAN is one of the popular and successful network design for GAN. It mainly composes of convolution layers without max pooling or fully connected layers. 
<br><br>
They are made of two distinct models, a generator and a discriminator. The job of the generator is to spawn ‘fake’ images that look like the training images. The job of the discriminator is to look at an image and output whether or not it is a real training image or a fake image from the generator. During training, the generator is constantly trying to outsmart the discriminator by generating better and better fakes, while the discriminator is working to become a better detective and correctly classify the real and fake images. The equilibrium of this game is when the generator is generating perfect fakes that look as if they came directly from the training data, and the discriminator is left to always guess at 50% confidence that the generator output is real or fake.

## My Architecuture:
![FaceGAN Architecture](screenshots/facegan_arch.png)

## Authors
* Aditya Jain : [Portfolio](https://adityajain.me)

## Licence
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/adityajn105/FaceGAN-Generating-Random-Faces/blob/master/LICENSE) file for details

## Must Read
1. [Keep Calm and train a GAN. Pitfalls and Tips on training Generative Adversarial Networks](https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9)
2. [How to set descrimator untrainable in GAN](https://stackoverflow.com/questions/51108076/generative-adversarial-networks-in-keras-doesnt-work-like-expected)