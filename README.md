# SafariNet: Iterative Model Building for Multiclass Regression on Image Data

## Introduction

This project was based on a final project I completed at Princeton for the class [SML 301: Data Intelligence: Modern Data Science Methods](https://csml.princeton.edu/undergraduate/sml-301-data-intelligence-modern-data-science-methods) with Professor Daisy Huang. The dataset is available [here](https://cthulhu.dyn.wildme.io/public/datasets/) (**Note**: This is a download link).

In this repository, I showcase three models. The first was the original model I submitted as my final project, and the next two are iterative improvements I made after the class ended. See the **Models** section for model details.

The dataset consists of images with animals from various species. Our goal is to count the number of animals per species in a given image, a multiclass regression problem. To make our task feasible given ordinary hardware and time scales, we limit ourselves to six species: Reticulated Giraffe, Masai Giraffe, Grevys Zebra, Plains Zebra, Sea Turtle, and Fluke Whale. However, even with a small number of species, a working model must be able to differentiate between two species of giraffes and two species of zebras while also being general enough to differentiate between two distinct marine animals.

## Models

Scores for each of these models can be found below.

**Model 1** 

For the original project, I followed the approach of [Song and Qiu](https://dl.acm.org/doi/abs/10.1145/3191442.3191459) to classify and count the animals simultaneously. A working model must count the animals by species, but in order to do this well, it must first learn to differentiate between the animals. Instead of relying on the model to learn this automatically during training, we add a classification penalty to help the model with this intermediate learning task.

 The model's first few layers resemble the beginning layers of [VGG16](https://arxiv.org/pdf/1409.1556.pdf), a state-of-the-art (and very big) model for image classification. The model's head has two interwoven streams, one predicting the species counts and the other predicting the probability of each species being in the image. 

**Model 2**

Model 2 is an iteration of the first and came many months later. The goal was simply to understand the Squeeze-and-Excitation block of [Hu et al](https://doi.org/10.1109/CVPR.2018.00745) by implementing it in a CNN I had already created.

A Squeeze-and-Excitation block is not itself a model. Rather, it's a small block allowing a multi-channel input to rescale its channels based on their respective importances. In the context of image data, these channels are features that a model has extracted from an image. Vanilla CNNs pass all this information, in parallel, through the convolutional layers. But for any given image, there are certain features that are more or less important. A Squeeze-and-Excitation block can emphasize features that are more useful while masking the rest. This is a basic but powerful example of an "attention mechanism" in deep learning.

I added the S-E block before the last convolutional layer so that it could adjust the extracted features before being processed by the feed-forward head. The performance of the model on the counting task improved by about 10%, but this could be due to an increase in model parameters.

**Model 3**

After Model 2, I was still a bit disappointed in the counting performance for species appearing in large groups, i.e. the giraffes and zebras. So, I normalized the counts by the standard deviation across each class on the training data. This allowed the computed loss to be less biased between classes. I also weighted the counting loss ten times more than the classification loss during training since this is our main task. (**Note**: When the predictions were calculated, I multiplied by the same standard deviations to make the scores between the models fair.)

## Results

The classification task didn't improve between models, but our main counting task did. Model 2 was better than Model 1 at counting (except for the Masai Giraffe), and on average, there was a 7% slash in relative RMSE. 

Model 3 improved on Model 2 in every class, chopping the error down by another 15% on average.

Model 3 is significantly better at counting any species than Model 1; on average, there was an over 20% improvement in error.

*Counting Task - Relative Root MSE*
| Model         | Reticulated Giraffe | Grevys Zebra | Sea Turtle  | Plains Zebra | Masai Giraffe | Fluke Whale  | Average   | 
| ------------- | ------------------- | ------------ | ----------- | ------------ | ------------- | ------------ | --------- |
| Model 1       | 0.37                | 0.58         | 0.09        | 0.91         | 0.42          | 0.16         | 0.42      |
| Model 2       | 0.34                | 0.52         | 0.08        | 0.84         | 0.45          | 0.10         | 0.39      |
| Model 3       | 0.27 *              | 0.46 *       | 0.05 *      | 0.73 *       | 0.40 *        | 0.09 *       | 0.33 *    |

*Classification Task - ROC AUC*
| Model         | Reticulated Giraffe | Grevys Zebra | Sea Turtle  | Plains Zebra | Masai Giraffe | Fluke Whale  | Average   | 
| ------------- | ------------------- | ------------ | ----------- | ------------ | ------------- | ------------ | --------- |
| Model 1       | 0.93                | 0.95         | 0.99 *      | 0.90 *       | 0.87 *        | 0.99         | 0.94 *    |
| Model 2       | 0.94 *              | 0.96 *       | 0.99 *      | 0.87         | 0.86          | 1.00 *       | 0.94 *    |
| Model 3       | 0.93                | 0.96 *       | 0.99 *      | 0.85         | 0.87 *        | 0.99         | 0.93      |

<sub>* denotes best score in column</sub>

## Files

**intro.ipynb**

This is a Jupyter Notebook performing basic Exploratory Data Analysis.

**model{1,2,3}.ipynb**

*model1.ipynb*, *model2.ipynb*, and *model3.ipynb* are the notebooks that train and evaluate each model.

**preprocessing.py**

Each image in the dataset comes with an XML file giving species information and bounding boxes. Running this script creates two folders, count_annotations and class_annotations, and extracts count and class data from the XML files. For example, an image with 2 Masai Giraffes will produce the array $[0, 0, 0, 0, 2, 0]$ for the count_annotations folder and $[0, 0, 0, 0, 1, 0]$ for the class_annotations folder.

**dataset.py**

This contains our custom PyTorch Dataset module. It contains a boolean input `stdevs`, allowing us to specify if we want to normalize count data by standard deviations. This is done for Model 3 but not for Models 1 or 2.

**model.py**

This contains our custom PyTorch model. `add_se` allows us to add the Squeeze-and-Excitation block. This is done for Models 2 and 3 but not for Model 1.

**utils.py**

This contains our training and evalutation functions along with other miscellaneous functions. The evaluation functions allow us to specify if we should denormalize outputs for fair scoring, and the training function allows us to reweight our losses. Both are utilized for Model 3.


## References

[[1]](https://dl.acm.org/doi/abs/10.1145/3191442.3191459) Zichen Song and Qiang Qiu. 2018. Learn to Classify and Count: A Unified Framework for Object Classification and Counting. In Proceedings of the 2018 International Conference on Image and Graphics Processing (ICIGP '18). Association for Computing Machinery, New York, NY, USA, 110–114.

[[2]](https://ieeexplore.ieee.org/document/8354227) J. Parham, C. Stewart, J. Crall, D. Rubenstein, J. Holmberg and T. Berger-Wolf, "An Animal Detection Pipeline for Identification," 2018 IEEE Winter Conference on Applications of Computer Vision (WACV), Lake Tahoe, NV, USA, 2018, pp. 1075-1083

[[3]](https://arxiv.org/pdf/1409.1556.pdf) Karen Simonyan and Andrew Zisserman. 2015. Very Deep Convolutional Networks for Large-Scale Image Recognition.

[[4]](https://doi.org/10.1109/CVPR.2018.00745) J. Hu, L. Shen and G. Sun, "Squeeze-and-Excitation Networks," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, USA, 2018, pp. 7132-7141