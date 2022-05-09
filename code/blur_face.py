import os
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
  
# plotting images
def plotImages(img):
    plt.imshow(img, cmap="gray")
    plt.axis('off')
    plt.style.use('seaborn')
    plt.show()
  
def blur(cascade_path, image_path):  
    # Reading an image using OpenCV
    image = cv2.imread(image_path)
    # Converting BGR image into a RGB image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # get the height and width of the image   
    h, w = image.shape[:2]
    # gaussian blur kernel size depends on width and height of original image
    kernel_width = (w // 7) | 1
    kernel_height = (h // 7) | 1 
    # Display the original image
    plotImages(image)
    
    # Use the Haar cascade classifier to detect the faces in Images
    face_cascade = cv2.CascadeClassifier(cascade_path)
    face_data = face_cascade.detectMultiScale(image, 1.3, 5)

    try:
        # Draw rectangle around the faces
        for (x, y, w, h) in face_data:
            # Enclose the detected faces inside a rectangular box
            start_point = (x, y)
            end_point = (x + w, y + h)
            color = (255, 0, 0)
            thickness = 0
            cv2.rectangle(image, start_point, end_point, color, thickness)
            # Extract the region of the image that contains the face
            ROI = image[y:y+h, x:x+w]
            # applying a gaussian blur over the new rectangle area  
            ROI = cv2.GaussianBlur(ROI, (kernel_width, kernel_height), 10)
            # put the blurred face into the original image
            image[y:y+ROI.shape[0], x:x+ROI.shape[1]] = ROI
        
        # save the bluarred image in the predictions folder
        # use cvtColor to convert from image from RGB to BGR
        filename, _ =os.path.splitext(os.path.basename(image_path))
        prediction_path = os.path.join('predictions', f'{filename}.jpg')
        cv2.imwrite(prediction_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
    except Exception as e:
        print(f'excption : {e}')
        pass

    # Display the Blurred Faces Image
    plotImages(image)

if __name__ == "__main__":
    
    default_model_path = os.path.join('models', 'haarcascade_frontalface_alt.xml')
    default_image_path = os.path.join('data', 'andrew_ng.jpeg')
    parser = argparse.ArgumentParser(description='Face Bluring.')
    parser.add_argument('--face_cascade', help='Path to face cascade.', default = default_model_path)
    parser.add_argument('--image', help='path of the image.', default= default_image_path)
    args = parser.parse_args()
    cascade = args.face_cascade 
    image = args.image
    blur(cascade, image)