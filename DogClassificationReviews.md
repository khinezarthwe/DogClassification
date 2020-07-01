#### Meets Specifications
```bash
Congratulations!
I appreciate your efforts and devotion. You have coded things correctly. Although a few things still need some improvement, I have added hints to help you out. I encourage you to make the changes before adding the project to your job portfolio.

You have asked a question about computing the output size of convolution layer. This can be done using the formula mentioned here: https://www.quora.com/How-can-I-calculate-the-size-of-output-of-convolutional-layer
Alternatively, you can use size() method to print the size of a tensor: link

I have also answered to the issue that you were facing with the Inception model in the below rubric points.

If you still have any kind of doubts, I encourage you to use these resources:
—> Knowledge
https://knowledge.udacity.com/
Knowledge base is Q&A forum, meaning students-mentors both ask and answer questions here.

Keep Learning! Deep Learning!

```

##### Files Submitted
- The submission includes all required, complete notebook files.

    - The files are rightly submitted.

##### Step 1: Detect Humans
- The submission returns the percentage of the first 100 images in the dog and human face datasets that include a detected, human face.
```
HaarCascade classifier is used correctly. It's performing pretty well detecting human faces.

Suggestion
I noticed that you have mentioned

Correct Percentage of first 100 imges, 17.00%
Total wrong image are 83
It's actually incorrectly predicting the humans in dog images i.e, 17 images out of 100 images of dogs.
It would be better if you change the print statement to this: Detected human faces in dogs images: 17%
```
##### Step 2: Detect Dogs
- Use a pre-trained VGG16 Net to find the predicted class for a given image. Use this to complete a dog_detector function below that returns True if a dog is detected in an image (and False if not).
```
The VGG16 is loaded perfectly. You have coded the VGG16_predict() rightly to load the images and do transformation.
It would be better if you use Normalization transform too.
```
- The submission returns the percentage of the first 100 images in the dog and human face datasets that include a detected dog.
```
The VGG16 is working pretty well detecting the dogs in dog images. To further see some improvement, I encourage you to use Normalization transform.
```

##### Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
- Write three separate data loaders for the training, validation, and test datasets of dog images. These images should be pre-processed to be of the correct size.
```
Amazing!
The training, validation and test transform are well coded. All the steps are precisely coded.
It's great that you have used the normalization transform. 👍
```
- Answer describes how the images were pre-processed and/or augmented.
```
The answer mentions the steps chosen to preprocess the images.
```
- The submission specifies a CNN architecture.
```
The architecture of network is acceptable. You have used the set of 3 convolution layers with ReLU and pooling layers to build the model.
The final fully connected layers seems perfect.

It would be great if you use some weight initialization techniques like Xavier initializer, kaiminguniform etc.
https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_
Try using the batch normalization with the convolution layers too.
```
- Answer describes the reasoning behind the selection of layer types.
```
Great!
The answer is quite explanatory. You have mentioned all your observations while building the model. :v:
```
- Choose appropriate loss and optimization functions for this classification task. Train the model for a number of epochs and save the "best" result.
```
criterion_scratch = nn.CrossEntropyLoss()

### TODO: select optimizer
optimizer_scratch =optim.SGD(model_scratch.parameters(), lr=0.1)
The loss function and SGD optimizer are used precisely.
I encourage you to read about other optimization techniques too: https://cs231n.github.io/neural-networks-3/
``` 
- The trained model attains at least 10% accuracy on the test set.
```
Test Accuracy: 13% (113/836)
The network is correctly trained. It's surpassing the mark of 10%. :clap:
```

##### Step 4: Create a CNN Using Transfer Learning
- The submission specifies a model architecture that uses part of a pre-trained model.
```
Amazing!
It's great that you have used the vgg19 architecture. The final fully connected layer is replaced with a new FC layer.


source: https://www.researchgate.net/figure/Pre-trained-VGG-and-VGG-BN-Architectures-and-DNNs-Top-1-Test-Accuracy-versus-average-log_fig3_330638379
```
- The submission details why the chosen architecture is suitable for this classification task.
```
The answer justifies the reason of using VGG19.

To resolve the inception's out of memory error, you could you have used a lower batch size. That usually helps to resolve the issue. Try using it and if you still face difficulty, you can reach out to us via https://knowledge.udacity.com/
```
- Train your model for a number of epochs and save the result wth the lowest validation loss.
```
Accuracy on the test set is 60% or greater.

Test Accuracy: 81% (685/836)
By training it for 5 epochs and getting 81% seems perfect. I encourage you to try training it for more epochs and observe the change in results.
```
- The submission includes a function that takes a file path to an image as input and returns the dog breed that is predicted by the CNN.

##### Step 5: Write Your Algorithm
- The submission uses the CNN from the previous step to detect dog breed. The submission has different output for each detected image type (dog, human, other) and provides either predicted actual (or resembling) dog breed.
```
The run_app() method takes in the image. It uses the face_detector, dog_detector and transfer learning-based model to classify the image.
```
##### Step 6: Test Your Algorithm
- The submission tests at least 6 images, including at least two human and two dog images.
```
Perfect!
The network is perfectly trained. You have tested the model with a diverse set of images. :star:
```
- Submission provides at least three possible points of improvement for the classification algorithm.
```
In deep learning, we experiment new things and we should always test out approach. I highly recommend you to try working on the all possible options.
```