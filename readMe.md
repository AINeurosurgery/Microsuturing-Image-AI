# Suturing Effectualness Evaluation

This is the official repository of the paper **"From feline classification to skills evaluation: A Multitask Learning Framework for Evaluating Micro Suturing Neurosurgical Skills"** accepted in ICIP 2023. 

The focus of this paper is to develop an automated tool for the assessment of trainees for micro-suturing task. The real-life training datasets for the micro-suturing task are often small, with long-tailed distribution, making it difficult to develop machine-learning-based tools for automated assessment. Further, micro-suturing is often performed at various magnifications and suture sizes, which makes the automated assessment more challenging compared to macro-suturing. Hence, currently, assessment is done manually by an expert using the final outcome image. In this paper, we propose a multi-task learning-based convolutional-neural-network regression model to score the effectualness of the micro-suturing task from the final outcome image. We propose a novel equivalent of the logit adjustment (used in classification) applicable to regression formulation, which effectively handles the problems associated with the long-tail distribution of the data. Additionally, we contribute the largest open-access dataset for suturing images and the first dataset pertaining to the micro-suturing task. We also demonstrate that the performance of the proposed algorithm surpasses the performance of human experts and also other state-of-the-art (SOTA) algorithms. 

### Requirements:

Install the following libraries before running the code

* Python 3.9.7 (Downloaded on Sept. 16, 2021 Through Anaconda)
* GCC 7.5.0
* Pytorch 1.10.2+cu113
* Torchvision 0.11.3+cu113
* Tensorboard 2.8.0
* Pillow 8.4.0
* scikit-learn 0.24.2
* Numpy 1.20.3
* Opencv-python 4.5.5.62
* Detailed package information of the conda environment is present in the file "package_info.txt"

### System Settings

The Repository was tested on a single Nvidia A100 GPU of 80 GB. The system details on which the repository was tested are:

* Processor: Intel(R) Xeon(R) Silver 4208 CPU @ 2.10G 
* Architecture: x86_64 
* NVIDIA-SMI version: 470.161.03
* Driver Version: 470.161.03
* CUDA Version: 11.4
* nvcc release: 11.4

### Training

* Create a folder named "data" and Download the dataset from the link given in the paper (http://aiimsnets.org/microsuturing.asp) to that folder.

* Download the pre-trained weights (.pth files) for Res2Net using the following link: (https://csciitd-my.sharepoint.com/:f:/g/personal/anz197518_iitd_ac_in/Ei2b_OdqGY9CrDA3_KQ7AucBWiap9NFz9X_sasv0mx6jWA?e=vnz2ih) into the root folder

* Run the program "preprocess.py" after making all the required changes as per comments to process the dataset

* To train the system, Run "main.py" by making the following changes
	* Change the path in line 173 to training data
	* Change the path in line 174 to the testing data

* It takes around 2 hours to train on the system specifications mentioned above

### Testing

* The testing is to be done once the training is complete.

  -OR-
  
* You can download the trained checkpoint (test_model78.pth.tar) using the following link: (https://csciitd-my.sharepoint.com/:f:/g/personal/anz197518_iitd_ac_in/Ei2b_OdqGY9CrDA3_KQ7AucBWiap9NFz9X_sasv0mx6jWA?e=vnz2ih)

* Download the dataset from the link given in the paper

* Run the program "preprocess.py" after making all the required changes as per comments to process the dataset

* To test the system, Run "test.py" by making the following changes
	* Change the path in line 86 to testing data
	* Change the path in line 92 to the best-trained model (It would have been saved in the "models" directory)

