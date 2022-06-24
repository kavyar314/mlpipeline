# mlpipeline
By Kavya Ravichandran

This document describes how to use this software and also serves as the report for the Programming Requirement.

## Goal

This pipeline exists to facilitate easy training and evaluation of machine learning models on various datasets. It is designed to be modular: the user can point to a dataset, an architecture, and an optimization algorithm and expect as output a trained model. Customization is available for each of these: data can be preprocessed in several different ways, different kinds of architectures can be specified, and different termination conditions can be explored for the optimizer. In addition, this platform also supports the Viola-Jones Face detector, involving a Haar feature-based weak predictor and AdaBoost to compute a linear predictor over these weak predictors

## Training Models

First, set up a config file with the following attributes if using a neural network:
* `batch_size`
* `loss_args`
* `model_args`
* `optimizer_args`
* `n_epochs`
* `save_path`
* `preproc`

or `img_dim` and `n_learners` if you are using Viola Jones and AdaBoost.

See `config.py` as an example. Note that if you intend to use a custom optimizer, you may need to provide an optimizer config file through the field `path for custom` in the `optimizer_args` variable. That would include the learning rate, the termination type and terminal value, and the names of the logs to use. 

Also, organize the data as described below in `Requirements >> data`. 

With all of this in place, let us discuss how to run the code.
The basic command for training models involves running a command line script:

`python train_model.py --data DATAPATH --model MODELNAME --loss LOSS  --optimizer OPTIMIZER --config config.py `

The following are some specific examples:
* `python train_model.py --data ./sample_data --model vgg16 --loss torch_ce  --optimizer torch_sgd --config config.py`
* `python train_model.py --data ./sample_data --model vgg16 --loss torch_ce  --optimizer logged_gd --config config.py`
* `python train_model.py --data ./sample_data --preproc VJ --model none  --loss none  --optimizer Boosting --config config.py`

## Requirements

### data

*I can send the data if needed via email or DropBox*

* top-level directory has the name of the dataset
    * each sub-directory matches the name of a class
        * files of images from that class are inside the subdirectory

```
dataset_name
	|__ class1
	    |__ img1.jpg
	    |__ img2.jpg
	    |__ ...
	|__ class2
	    |__ img1.jpg
	    |__ img2.jpg
	    |__ ...
	|__ class3
	    |__ img1.jpg
	    |__ img2.jpg
	    |__ ...
	|__ ...
```

### optimizers

Optimizers must have a `.step()` method implemented as in Pytorch.


## Extensions

So far, this platform only has a handful of options implemented for each modular part, but this can be very easily extended. For models and optimizers, simply implement the respective model or optimizer desired and then add it to the respective selector `model_selector` or `optimizer_selector`. Alternate termination conditions can also be added to the class `OptimizerTermination` in the `optimizers.py` file for alternate options.