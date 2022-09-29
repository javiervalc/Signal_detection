import cv2
import os
from sqlite3 import paramstyle
import tensorflow as tf
import numpy as np


from tensorflow.keras.preprocessing.image import load_img, img_to_array


import cv2



###################
#######params######
###################


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CLASSES = ['100', '110', '120', '50', '60', '70', '80', '90']
OUTPUT_DIR = 'static'
SIGN_DIR = 'signs2'
MODEL_DIR = 'model'
MODEL_FILE = 'model_signs_v2.h5'
SIZE = 150






print('Loading Model..')
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, MODEL_FILE))
model.trainable = False





def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS






def make_prediction(imgen_array):
    """
    Predicts a given `filename` file
    """
    #print('Filename is ', filename)
    #full_path = os.path.join(lc.OUTPUT_DIR, filename)
    #test_data = prepare_image(full_path)
    predictions = model.predict(imgen_array)

    output=CLASSES[np.argmax(predictions[0])]
    final_score=100 * np.max(tf.nn.softmax(predictions[0]))
    return output, final_score




def circles_detection(image):
    image = cv2.imread("static/original.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('static/gray.jpg', gray)


    roi_corners = np.array([[(500, 600), (450, 750), (1280, 600), (1280, 300)]], dtype=np.int32)
    mask = np.ones(image.shape, dtype=np.uint8)
    mask.fill(255)
    cv2.fillPoly(mask, roi_corners, 0)
    masked_image = cv2.bitwise_or(mask,image) #a la imagen original le quito el trapecio



    #circle detection
    gray2 = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray2, 5)
    circles_0 = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 300, param1=50,param2=40,minRadius=5,maxRadius=60)
    global circles
    circles = np.round(circles_0[0, :]).astype("int") #los redondeo para poderlos recorrer con un for













    #for each circle cut the rectangle
    for i in range(circles.shape[0]):
        x1=circles[i][0]
        x2=circles[i][2]
        y1=circles[i][1]
        y2=circles[i][2]
        r=circles[i][2]
        x_crop=x1-x2
        y_crop=y1-y2


        # crop image as a square
        img = blur[y_crop:y_crop+r*2, x_crop:x_crop+r*2]


        # create a mask
        mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)


        # create circle mask, center, radius, fill color, size of the border
        cv2.circle(mask,(r,r), r, (255,255,255),-1)


        # get only the inside pixels
        fg = cv2.bitwise_or(img, img, mask=mask)

        mask = cv2.bitwise_not(mask)
        background = np.full(img.shape, 255, dtype=np.uint8)
        bk = cv2.bitwise_or(background, background, mask=mask)
        final = cv2.bitwise_or(fg, bk)


        cv2.imwrite(f'static/signs/sign_{i}.jpg', final)


    return circles




def circles_drawing(x,y,r): #dibujar un solo circulo
    image = cv2.imread("static/gray.jpg")
    cv2.circle(image, (x, y), r, (0, 255, 0), 4)
    cv2.imwrite('static/gray2.jpg', image)


