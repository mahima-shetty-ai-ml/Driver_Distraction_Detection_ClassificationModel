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
     
Changing 

    new_img = cv2.resize(test_array,(img_size,img_size))
    print(new_img.shape)
    plt.imshow(new_img)
    plt.show()
