## Introduction

This project was based on a final project I completed at Princeton for the class [SML 301: Data Intelligence: Modern Data Science Methods](https://csml.princeton.edu/undergraduate/sml-301-data-intelligence-modern-data-science-methods) with Professor Daisy Huang.

The dataset is an image dataset showing animals across different species. Interestingly, many images contain multiple animals from the same species, and some images contain animals from multiple species. The goal of the project is to count the number of animals per species. In other words, we must perform a classification and regression task simulataneously. To make the problem feasible for ordinary hardware and time scales, we limit ourselves to six species. The dataset is available [here](https://cthulhu.dyn.wildme.io/public/datasets/) (**Note**: This is a download link).

For the original project, we implemented a miniature version of the VGG16 model (i.e., not 16 layers), which performed relatively well on its own. 

After the course had ended, I was reading about the now ubiquitous Squeeze-and-Excitation block from the famous paper of [Hu et al](https://doi.org/10.1109/CVPR.2018.00745). The Squeeze-and-Excitation block allows a multi-channel input to rescale its channels based on their respective importance. This "importance" is judged from a point of view that considers all channels simultaneously. Instead of every high-level feature making it to the model head as-is, the Squeeze-and-Excitation block can emphasize certain features beforehand that it believes will be most beneficial to the model's performance. This can be viewed as an "attention mechanism", wherein the sigmoid activation performs soft-gating on the channels. This addition culminated in "Iteration 2".

From the scores alone, I was happy to see a few things. For one, the original model was already great at classifying sea turtles and fluke whales with nearly perfect accuracy. However, it was still not great at counting these animals, even though they virtually always appear alone in the images. In iteration 2, the relative root mean squred errors for these two classes dropped dramatically without compromising the performance of the classification task.

Here's my guess. Once the Squeeze-and-Excitation block was introduced, the soft-gating gave the model the opportunity to classify animals earlier, allowing distinct regression schemes and count distributions to be learned in the model's head.

After Iteration 2, I was still a bit disappointed in the counting performance for species appearing in large groups, i.e. the giraffes and zebras. So, I normalized the counts by the standard deviation across each class on the training data. This allowed the computed loss to be less biased between the classes. I also weighted the counting loss twice as much as the classification loss since the classification loss was generally ~2x as large as the counting loss. When the predictions were claculated, I multiplied by the same standard deviations to make the scores between the models fair. Across scores, Iteration 3 performed the best much more frequently, and when it didn't have the best score, it was a close second.

## Scoring

Counting Task - Relative Root Mean Squared Errors
| Model        | Reticulated Giraffe | Grevys Zebra | Sea Turtle  | Plains Zebra | Masai Giraffe | Fluke Whale  | Average   | 
| ------------ | ------------------- | ------------ | ----------- | ------------ | ------------- | ------------ | --------- |
| Original     | 0.35                | 0.55         | 0.09        | 1.02         | 0.45          | 0.10         | 0.43      |
| Iteration 2  | 0.34                | 0.52         | 0.08        | 0.84         | 0.45          | 0.10         | 0.39      |
| Iteration 3  | 0.27 *              | 0.46 *       | 0.05 *      | 0.73 *       | 0.40 *        | 0.09 *       | 0.33 *    |

Classification Task - ROC AUC
| Model        | Reticulated Giraffe | Grevys Zebra | Sea Turtle  | Plains Zebra | Masai Giraffe | Fluke Whale  | Average   | 
| ------------ | ------------------- | ------------ | ----------- | ------------ | ------------- | ------------ | --------- |
| Original     | 0.93                | 0.95         | 0.99 *      | 0.84         | 0.87 *        | 0.99         | 0.93      |
| Iteration 2  | 0.94 *              | 0.96 *       | 0.99 *      | 0.87 *       | 0.86          | 1.00 *       | 0.94 *    |
| Iteration 3  | 0.93                | 0.96 *       | 0.99 *      | 0.85         | 0.87 *        | 0.99         | 0.93      |

<sub>* denotes best score in column</sub>

## References
[[1]](https://dl.acm.org/doi/abs/10.1145/3191442.3191459) Zichen Song and Qiang Qiu. 2018. Learn to Classify and Count: A Unified Framework for Object Classification and Counting. In Proceedings of the 2018 International Conference on Image and Graphics Processing (ICIGP '18). Association for Computing Machinery, New York, NY, USA, 110–114.

[[2]](https://ieeexplore.ieee.org/document/8354227) J. Parham, C. Stewart, J. Crall, D. Rubenstein, J. Holmberg and T. Berger-Wolf, "An Animal Detection Pipeline for Identification," 2018 IEEE Winter Conference on Applications of Computer Vision (WACV), Lake Tahoe, NV, USA, 2018, pp. 1075-1083

[[3]](https://doi.org/10.1109/CVPR.2018.00745) J. Hu, L. Shen and G. Sun, "Squeeze-and-Excitation Networks," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, USA, 2018, pp. 7132-7141