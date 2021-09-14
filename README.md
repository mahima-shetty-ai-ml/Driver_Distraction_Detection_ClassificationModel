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


Classification </br> 
Deep learning NN are an example of an algorithm that natively supports multi label classification problems. Neural network models for multi label classification tasks can be easily defined and evaluated using the Keras deep learning library. </br> </br>
Perception </br> Machine Perception refers to the added functionality in computer systems that enables reaction based on senses, similar to human perception. We imported CV2 ie. OpenCV which does the relatively major job of perception. </br> </br>
Understanding </br> Deep learning works on the principle of extracting features from the raw data by using multiple layers for identifying different criteria relevant to I/p data. Deep learning techniques include CNN, RNN, and deep NN. </br> </br>
Discovering
Prediction
Creation.


    !pip install keras-utils
    import keras
    import keras_utils
    from keras.utils import to_categorical

    from tensorflow.keras import datasets, layers, models, callbacks
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation, BatchNormalization

Y_train = to_categorical(y_train,num_classes=10)
Y_test = to_categorical(y_test,num_classes=10)
