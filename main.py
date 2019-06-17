import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def get_path_list(root_path):
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''
    names = os.listdir(root_path) #get subdirectory in test/train folder
    return names 

def get_class_names(root_path, train_names):
    '''
        To get a list of train images path and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        
        Returns
        -------
        list
            List containing all image paths in the train directories
        list
            List containing all image classes id
    '''
    image_path_list = []
    image_classes_list = []

    for i, name in enumerate(train_names) :
        images = os.listdir(root_path+'/'+train_names[i]) #dataset/train/Aaron_Eckhart
        for image in images :
            image_path_list.append(root_path+'/'+name+'/'+image) #dataset/train/Aaron_Eckhart/Aaron_Eckhart_0001.jpg
            image_classes_list.append(i) #label each image in image_path_list to corresponding classes
    return image_path_list, image_classes_list

def get_train_images_data(image_path_list):
    '''
        To load a list of train images from given path list

        Parameters
        ----------
        image_path_list : list
            List containing all image paths in the train directories
        
        Returns
        -------
        list
            List containing all loaded train images
    '''
    train_image_list = []
    for name in image_path_list :
        train_image_list.append(cv2.imread(name)) #load image
    return train_image_list

def detect_faces_and_filter(image_list, image_classes_list=None):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is more or less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    '''
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cropped_faces = []
    cropped_faces_location = []
    filtered_classes_list = []
    for i, image in enumerate(image_list) :
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #convert to grayscale
        detected_faces = face_cascade.detectMultiScale(image, scaleFactor = 1.2, minNeighbors = 5)
        # print(len(detected_faces)) #show number of faces detected in a picture
        if len(detected_faces)<1 :
            continue
        for face in detected_faces :
            x,y,w,h = face
            cropped_faces_location.append(face)
            face_rect = image[y:y+h, x:x+w] #crop detected faces from image
            cropped_faces.append(face_rect) #append cropped images to list
            if image_classes_list is not None :
                filtered_classes_list.append(image_classes_list[i]) #label each cropped images
            else :
                filtered_classes_list = [''] #prevent None return value error
        
        # debug cropped faces
        # if i<6 :
        #     cv2.imshow("show", cropped_faces[i])
        #     print(filtered_classes_list[i])
        #     print(cropped_faces_location[i])
        #     cv2.waitKey(0)

    return cropped_faces,cropped_faces_location,filtered_classes_list

def train(train_face_grays, image_classes_list):
    '''
        To create and train classifier object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Classifier object after being trained with cropped face images
    '''
    classifier = cv2.face.LBPHFaceRecognizer_create()
    classifier.train(train_face_grays, np.array(image_classes_list))
    return classifier

def get_test_images_data(test_root_path, image_path_list):
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        image_path_list : list
            List containing all image paths in the test directories
        
        Returns
        -------
        list
            List containing all loaded test images
    '''
    test_image_list = []
    for name in image_path_list :
        test_image_list.append(cv2.imread(test_root_path+'/'+name)) #load image
    return test_image_list

def predict(classifier, test_faces_gray):
    '''
        To predict the test image with classifier

        Parameters
        ----------
        classifier : object
            Classifier object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''
    # print(len(test_faces_gray))
    result = []
    for face in test_faces_gray :
        res, _ = classifier.predict(face)
        result.append(res)
        print(result) #debug
    return result

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    '''
        To draw prediction results on the given test images

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            prediction result
    '''
    predicted_test_image_list = []
    for i, result in enumerate(predict_results) :
        # print(train_names[result]) #debug
        x,y,w,h = test_faces_rects[i]
        person_name = train_names[result].replace("_"," ")
        cv2.rectangle(test_image_list[i], (x,y), (x+w,y+h), (0,255,0), 1) #add rectangle
        cv2.putText(test_image_list[i], person_name, (x,y-10), 0, 0.5, (0,255,0)) #add person name
        predicted_test_image_list.append(test_image_list[i])
    return predicted_test_image_list

def combine_results(predicted_test_image_list):
    '''
        To combine all predicted test image result into one image

        Parameters
        ----------
        predicted_test_image_list : list
            List containing all test images after being drawn with
            prediction result

        Returns
        -------
        ndarray
            Array containing image data after being combined
    '''
    final_image_result = np.array(predicted_test_image_list)
    print(final_image_result.shape)
    return final_image_result

def show_result(image):
    '''
        To show the given image

        Parameters
        ----------
        image : ndarray
            Array containing image data
    '''
    image = np.hstack(image)
    cv2.imshow("Result", image)
    cv2.waitKey(0)

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == "__main__":
    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "dataset/train"
    '''
        -------------------
        End of modifiable
        -------------------
    '''
    
    train_names = get_path_list(train_root_path)
    image_path_list, image_classes_list = get_class_names(train_root_path, train_names)
    train_image_list = get_train_images_data(image_path_list)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    classifier = train(train_face_grays, filtered_classes_list)

    '''
        Please modify test_image_path value according to the location of
        your data test root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "dataset/test"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_names = get_path_list(test_root_path)
    test_image_list = get_test_images_data(test_root_path, test_names)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(classifier, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)
    final_image_result = combine_results(predicted_test_image_list)
    show_result(final_image_result)