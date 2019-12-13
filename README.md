# DeepTesla team 
Repository for BME's Deep Learning in practice in Python or Lua subject's home assginment.

## Training, test, validation data for semantics segmentation
We extracted original and annotated video frames from gym_duckietown environment. By using these frames from the video files found under /recordings directory, we plan to train, test, and validate our network.

For more information on how the data collection and preprocessing was done, please refer to [this](https://github.com/DeepTesla/deep_learning_hf/wiki/Data-used-for-training,-testing-and-validating-the-network) wiki page.

### Example training data
Data pair 1:

![orig_1](https://raw.githubusercontent.com/DeepTesla/deep_learning_hf/data/examples/orig_1.jpg)
![annot_1](https://raw.githubusercontent.com/DeepTesla/deep_learning_hf/data/examples/annot_1.jpg)

Data pair 2:

![orig_2](https://raw.githubusercontent.com/DeepTesla/deep_learning_hf/data/examples/orig_2.jpg)
![annot_2](https://raw.githubusercontent.com/DeepTesla/deep_learning_hf/data/examples/annot_2.jpg)

## Simple notebook for basic training of a selected neural network
In the notebook [network.ipynb](https://github.com/DeepTesla/deep_learning_hf/blob/master/network.ipynb) we provide an example how our network training process works. All notebooks were run in Google Colab using default packages.

First, we download the data and divide the video files into separate image files as we are building a single-image classifier. Then, we define a data loader class that will help us load the data dynamically thus saving precious system memory. Next, we define our model, that is a simple encoder-decoder network with 3-3 encoding and decoding blocks. We train this network using Adam optimizer and binary crossentropy as the loss function and the results can be seen at the end of the notebook.

With our optical, subjective evaluation method, the results are satisfying, we plan to optimize the hyperparameters of our network to achieve smaller size and better performance.
