# Suturing Effectualness Evaluation

In this paper, we propose to evaluate the effectualness of the microsuturing task. 

### Requirements:

Install the following libraries before running the code

* Pytorch
* Torchvision
* Tensorboard
* Pillow
* Sklearn
* Numpy 
* Opencv

### Training

* Download the dataset from the link given in the paper

* Run the program "preprocess.py", after making all the required changes as per comments to process the dataset

* To train the system, Run "main.py" by making the following changes
	* Change the path in line 120 to training data
	* Change the path in line 121 to the testing data


### Testing

* The testing is to be done, once the training is complete.

* Download the dataset from the link given in the paper

* Run the program "preprocess.py", after making all the required changes as per comments to process the dataset

* To test the system, Run "test.py" by making the following changes
	* Change the path in line 86 to testing data
	* Change the path in line 92 to the best-trained model

