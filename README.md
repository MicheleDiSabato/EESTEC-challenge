# EESTEC-challenge
Solution of a machine learning problem during the 12 hours hackathon EESTEC challenge, local round (Milan, April 23, 2022).

The problem concerns the widespread formation of defects in Fused Filament Fabrication (FFF) 3D printer devices such as the presence of filament. The formation of defects, or even critical failures, such as the print object losing adhesion to the print bed will not stop the print process, and printing will continue if not interrupted by the operator, leading to waste of material, power, effective equipment operation time, as well as potentially causing malfunctions to printing parts. To monitor these situations, most of the printer have built in camera features to recognize the presence of defects.

Machine learning models are developed for identifying 3D printing defects during the printing process by analyzing video captured from the process. Defects are likely to occur in 3D printed objects during the printing process, with one of them being stringing; they are mostly correlated to one of the printing parameters or the object’s geometries. The stringing defect can be on a large scale and is usually located in visible parts of the object recorded by a capturing camera.

We tried to developed a neural network to recognize these patterns starting from a collection of images containing anomalies or not.

Firstly we dealt with a binary classification problem. We noticed that the anomalies are not easily detectable by human eye, so we decided to use a black box model. In particular we chose to train a **convolutional neural networks** fine tuning the convolutional part of the **InceptionV3** network and adding a fully connected dense layer with 128 neurons. As loss function we used categorical crossentropy, while we relied on the "Adam" optimizer in order to reduce automatically the learning rate during training.

Secondly we dealt with a unsupervised clustering problem. We used the pretrained Convolutional Neural Network (InceptionV3) to get the latent representation of each image. We keep every layer of InceptionV3, except for the output layer. The latent representation is therefore a vector with 2048 components. Dealing with 2048 components is very computationally expensive, therefore through SVD decomposition we compute the 452 Principal Components. Compute the explained variance: we found out that the first 50 principal components explained more than 90% of the variability of the dataset composed of the latent representations and we applied three different **unsupervised clustering algorithms**, comparing their performances through the inertia metric:

•	Kmeans (k = 3): we used 100 initial starting points, thus averaging on the response of each model out of the 100 (bagging). The inertia reached was: 225.68323.

•	Kmedoids (k = 3): the inertia reached was: 263.6713.

•	Agglomerative Clustering: we tried different linkage and distance metrics, the best (lowest) inertia reached was: 229.8183.

In the end, we select Kmeans algorithm both in terms of inertia and weighted inertia.

**Thank you Michele Di Sabato and Simone Colombara for teaching me so many things about neural networks in just one day!**




