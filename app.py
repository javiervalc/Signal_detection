#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import time
import utils
import tensorflow as tf
from flask import Flask, flash, request, redirect, url_for, render_template
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename

# Setting up environment
if not os.path.isdir(utils.OUTPUT_DIR):
    os.mkdir(utils.OUTPUT_DIR)

if not os.path.isdir(utils.SIGN_DIR):
    os.mkdir(utils.SIGN_DIR)

app = Flask(__name__)
Bootstrap(app)
app.config['UPLOAD_FOLDER'] = utils.OUTPUT_DIR








@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and utils.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))  #para guardar el archivo en static.
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'original.jpg'))  #para guardar el archivo en static.

            utils.circles_detection(filename)

            time.sleep(5)


            for i in range((utils.circles.shape[0])):
                imagen=tf.keras.preprocessing.image.load_img(f'static/signs/sign_{i}.jpg',target_size=(utils.SIZE,utils.SIZE))



                imgen_array = tf.keras.utils.img_to_array(imagen)
                imgen_array = tf.expand_dims(imgen_array, 0)



                output,final_score = utils.make_prediction(imgen_array)

                print(f"xxxxxxxxxSCORE{final_score}xxxxxxxxxxxxxxxxxxx")

                if final_score>95:

                    x=utils.circles[i][0]
                    y=utils.circles[i][1]
                    r=utils.circles[i][2]


                    utils.circles_drawing(x,y,r)






                    path_to_image = url_for('static', filename=('signs/sign_'+str(i)+'.jpg'))
                    path_to_image2 = url_for('static', filename='gray2.jpg')

                    result = {
                        'output': output,
                        'final_score': final_score,
                        'path_to_image': path_to_image,
                        'path_to_image2': path_to_image2,
                        'size': utils.SIZE
                        }









                    return render_template('show.html', result=result)
    return render_template('index.html')







if __name__ == "__main__":
    app.run(debug=True)
