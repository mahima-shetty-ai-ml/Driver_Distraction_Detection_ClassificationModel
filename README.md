# Driver_Distraction_Detection_ClassificationModel

The provided data set has driver images, each taken in a car with a driver doing something in the car (texting, eating, talking on the phone, makeup, reaching behind, etc). This dataset is obtained from Kaggle(State Farm Distracted Driver Detection competition) for Edvancer Deep Learning Certification.

| Classes |          Predictions         | 
|:-------:|:----------------------------:|
|    C0   |         Safe Driving         |
|    C1   | Talking on the phone - Right |
|    C2   |        Texting - Left        |
|    C3   |  Talking on the phone - Left |
|    C4   |        Texting - Right       |
|    C5   |      Operating the Radio     |
|    C6   |           Drinking           |
|    C7   |        Reaching Behind       |
|    C8   |         Hair & Makeup        |
|    C9   |     Talking to Passenger     |

![driverdistraction](https://user-images.githubusercontent.com/41589522/133131098-7cd072e2-ecbf-44f9-bb30-d439ccb3bfc0.png)

# Model Making BreakDown

From OpenCV2 library, **IMREAD function** : Used to read image from the path given.  
**IMREAD_COLOR function**: It specifies to load a color image.   
**COLOR_BGR2RGB function** Used to convert image into RGB.  
**IMSHOW function** Used to Display the image. 

    import cv2
    import matplotlib.pyplot as plt
    import os
    for i in classes:
        path = os.path.join(directory,i)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_COLOR)
            RGB_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            plt.imshow(RGB_img)
            plt.show()
            break
        break

Same Image processing with Test Data using OpenCV library functions. 
Using **IMREAD function** : Used to read image from the path given.  
**IMREAD_COLOR function**: It specifies to load a color image.   
**COLOR_BGR2RGB function** Used to convert image into RGB.  
**IMSHOW function** Used to Display the image. 


    test_array = []
    for img in os.listdir(test_directory):
        img_array = cv2.imread(os.path.join(test_directory,img),cv2.IMREAD_COLOR)
        RGB_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        test_array = RGB_img
        plt.imshow(RGB_img)
        plt.show()
        break


Getting to know the original size with array by the function shape()
      
    print(img_array.shape) #Output: (480, 640, 3)
     


Since neural networks receive inputs of the same size, all images need to be resized to a fixed size before inputting them to our CNN model. The larger the fixed size, the less shrinking required. Less shrinking means less deformation of features and patterns inside the image.
Changing (480, 640, 3) original size to (240, 240, 3) new size.

    new_img = cv2.resize(test_array,(img_size,img_size))
    print(new_img.shape)
    plt.imshow(new_img)
    plt.show()



Organizing and appending both Training Image Data with class number into a list. This is used later in the training model.  


    training_data = []
    i = 0
    def create_training_data():
        for category in classes:
            path = os.path.join(directory,category)
            class_num = classes.index(category)
            for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_COLOR)
                RGB_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                new_img = cv2.resize(RGB_img,(img_size,img_size))
                training_data.append([new_img,class_num])
    create_training_data()


Random Shuffling the training data. Data Shuffling. Simply put, shuffling techniques aim to mix up the data and can optionally preserve logical relationships between columns. It randomly shuffles data from a dataset within an series of attribute or a set of attributes.

    import random
    random.shuffle(training_data)
    x = []
    y = []

For every feature and label pair in training data. Then we will append those seperately to list1 x and list2 y.

    for features, label in training_data:
        x.append(features)
        y.append(label)
        
Subseting the data First 2242 records.

    x = x[:2242]
    y = y[:2242]
    
Resizing the data into size (240, 240)    
    
    import numpy as np 
    import pandas as pd 
    x = np.array(x).reshape(-1,img_size,img_size,3)
    x[0].shape #Output: (240, 240, 3)
    
# Begin Training the data

Splitting the data using train_test_split from sklearn model_selection module.  Taking Train-Test data ratio 70% - 30%.

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=96)


Importing Packages and modules from Tensorflow, Keras. From these packages, modules like datasets, layers, models, callbacks & Conv2D, MaxPooling2D, Dense, DropOut, Flatten, Activation, Batch Normalization.   

Tensorflow: It is an OPEN SOURCE AI library, using data flow graphs to build models. It allows developers to create large-scale neural networks (NN) with many layers. TensorFlow is mainly used for: 
</br></br>

Classification </br> 
Deep learning NN are an example of an algorithm that natively supports multi label classification problems. Neural network models for multi label classification tasks can be easily defined and evaluated using the Keras deep learning library. </br> </br>
Perception </br> Machine Perception refers to the added functionality in computer systems that enables reaction based on senses, similar to human perception. We imported CV2 ie. OpenCV which does the relatively major job of perception. </br> </br>
Understanding </br> Deep learning works on the principle of extracting features from the raw data by using multiple layers linke CNN, RNN, DropOut layer etc. for identifying different criteria relevant to I/p data. Deep learning techniques include CNN, RNN, and deep NN. </br> </br>
Prediction</br> It refers to the output of an Deep learning algorithm after it has been trained on our image dataset and applied to new testing data which we split during the beginning of deep learning training part when forecasting the likelihood of a particular outcome, such as whether the driver is safe driving, in our case and if not what other activities the driver is doing.  </br></br>
Creation.</br> Creation of Deep learning model for predictions and muulti class classification </br> </br>


    !pip install keras-utils
    import keras
    import keras_utils
    from keras.utils import to_categorical

    from tensorflow.keras import datasets, layers, models, callbacks
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation, BatchNormalization

</br>
Keras</br> Keras is a useful deep learning API written in Python, running on above the machine learning platform TensorFlow TF.</br><?br>

Callbacks</br>A callback function is a function passed into another function as an argument, which is then revoked inside the outermost function to complete some sort of routine or action. In deep learning, a callback is a function that can perform actions at various stages of training (e.g. at the start or end of an epoch, before or after a single batch, etc). </br></br>

Convolution</br> In image processing, convolution is the process of transforming an image by applying a kernel over each pixel and its local neighbors across the entire image.</br></br>

Conv2D</br>Keras Conv2D is a 2D Convolution Layer, the layer create a convolution kernel that is filled with layers input which helps to produce a tensor of outputs.

Kernel: In image processing kernel is a convolution matrix or masks which can be used for blurring, sharpening, embossing, edge detection, and more by doing a convolution between a kernel and an image. </br></br>
![convSobel](https://user-images.githubusercontent.com/41589522/133200140-163ea103-c31f-4a62-8462-c4492696a272.gif)</br>    
Image credit: **https://mlnotebook.github.io/post/CNN1/**
</br>
MaxPooling</br> Max pooling, is a pooling operation that calculates the maximum value in each patch of each feature orattribute map.</br></br>
![Pooling](https://user-images.githubusercontent.com/41589522/133200396-758d1e3f-587a-46d0-b3ca-0d811b565f2b.gif)

Dense</br>Dense layer is the regular deeply connected neural network layer. It is most common and frequently used layer.</br></br>

Image Credit **https://medium.com/analytics-vidhya/your-handbook-to-convolutional-neural-networks-628782b68f7e**</br>

DropOut</br>Drop out layer was added to account for overfitting.</br></br>

<!-- ![image](https://user-images.githubusercontent.com/41589522/133201197-f9f69b91-35cb-4972-bd90-2c57dab656d2.png)</br> -->
![dropout](https://user-images.githubusercontent.com/41589522/133201681-b1546a1f-0f4e-471f-adb4-d67e3357bd54.gif)
</br></br>
Image Credit **https://nagadakos.github.io/2018/09/23/dropout-effect-discussion/**</br>


Making Y tain and test categorical</br>

        Y_train = to_categorical(y_train,num_classes=10)
        Y_test = to_categorical(y_test,num_classes=10)

</br></br>

Building Sequential Model with Convolutional Neural Networks</br>

A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor. We use the 'add()' function to add layers to our model. We will add two layers and an output layer.</br></br>

    model = models.Sequential()
    ## CNN 1
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(240,240,3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization(axis = 3))
    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
    model.add(Dropout(0.3))
    ## CNN 2
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization(axis = 3))
    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
    model.add(Dropout(0.3))
    ## CNN 3
    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization(axis = 3))
    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
    model.add(Dropout(0.5))
    ## Dense & Output
    model.add(Flatten())
    model.add(Dense(units = 512,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units = 128,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10,activation='softmax'))


![giphy](https://user-images.githubusercontent.com/41589522/133202326-9ebf7e9d-83a9-4a2b-b918-e990914d53f9.gif)</br>
Image Credit: **https://www.kaggle.com/abhinand05/in-depth-guide-to-convolutional-neural-networks**




Relu Activation</br> The rectified linear activation function or ReLU for short is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero.</br></br>


![1_w48zY6o9_5W9iesSsNabmQ](https://user-images.githubusercontent.com/41589522/133204317-6704ae05-7874-4977-a087-2c345826eb4c.gif)


Image Credit **https://medium.com/ai%C2%B3-theory-practice-business/magic-behind-activation-function-c6fbc5e36a92**
</br>
Batch Normalization</br> Batch normalization is a layer that allows every layer of the network to do learning more independently. It is used to normalize the output of the previous layers. Using batch normalization learning becomes efficient also it can be used as regularization to avoid overfitting of the model.</br></br>


![AbandonedFrayedAmericanriverotter-size_restricted](https://user-images.githubusercontent.com/41589522/133204939-783c2e58-1958-4778-8993-212bf4b61989.gif)

Image Credit **https://gfycat.com/gifs/search/batch+normalization**

Softmax</brThe softmax function is used as the activation function in the output layer of neural network models that predict a multinomial probability > </br></br> Categorical crossentropy
