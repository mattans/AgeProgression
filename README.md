# PyTorch Implementation of Age Progression/Regression byConditional Adversarial Autoencoder

We developed a few applications based on Jupyter Notebook to test the system with the trained modelsinteractively.  As inputs, users can choose between already labeled images from UTKFace, to observe theresults with regard to parameters such as age, gender, and race.  The applications, referred as Games, canbe seen in Figures 7 to 9.

## Motivation

As participants of the 2018 Workshop in Machine Learning Applications for Computer Graphics (Cohen-Or, Fogel), we were exposed to many interesting ideas in the fields of artificial intelligence and computervision,  such  as  variational  autoencoders  (VAE)  and  deep  convolutional  generative  adversarial  networks(DCGAN).  In  the  latter  part  of  the  course,  we  were  asked  to  choose  a  paper  to  study  and  implement.Skimming through articles, we discovered an interesting paper from 2017 titledAge Progression/Regressionby Conditional Adversarial Autoencoder(Zhang, Song, et al.).  The article presented a method to performage modification on a given face image, with exciting utilization from recreational applications to assistthe searches on missing children.

### System Architecture

The system architecture was written in Python 3.7 and PyTorch 0.4.1, with attempts to keep the code ascompatible as possible with older versions of Python 3 and PyTorch.  Other external packages that wereused are NumPy, scikit-learn, OpenCV, imageio and Matplotlib.

### Encoder

![alt Encoder with 5 convolutional layers and a fully connected layer.  Viewing from left to right, faceimages of dimensions 128×128×3 are transformed into unlabeled Z vectors of size 50 in a latent space.](https://raw.githubusercontent.com/username/projectname/branch/path/to/encoder.png)

### Generator

![alt  Generator with 7 deconvolutional layers and a fully connected layer.  Viewing from left to right,labeled Z vectors of size 70 in a latent space are transformed into face images of dimensions 128×128×3.](https://raw.githubusercontent.com/username/projectname/branch/path/to/generator.png)

### Training

For  training,  we  used  the  UTKFace  dataset,  which  was  collected  by  the  original  authors  of  the  articleand tested in their implementation.  UTKFace contains over 20,000 aligned and cropped face images withtheir appropriate labels.  We wrote a special utility, the UTKFace Labeler, which sorts the dataset imagesto separated folders based on the label, to match with PyTorch demands that classes are determined byfolders structure.
## Authors

Mattan Serry, Hila Balahsan and Dor AltBlavatnik School of Computer Science, Tel Aviv University

## License

This project is licensed under the TAU License 

## Acknowledgments

* https://arxiv.org/abs/1702.08423
* Founded by AWS
