# Importing Kivy Dependencies
from kivy.app import App 
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
Window.clearcolor = (1,1,1,1)
from kivy.logger import Logger

# Importing Recognizer dependencies
import cv2
import tensorflow as tf
from tensorflow.python.keras.models import load_model as load_model
from layers import L1Dist
import os
import numpy as np

# Building App Layout
class Facial_RecognitionApp(App):

    def build(self):
        # Main Layout Components 
        self.web_cam = Image(size_hint=(1, .8))
        self.button = Button(
            text="VERIFY", 
            size_hint=(1, .1), 
            font_size=48,
            bold = True,
           border =  (30,30,30,30),
            background_normal = '',
            background_color = '#e5e5e5',
            color ='#000000',
            on_press = self.verify 
            )
        self.verification_label = Label(text="Verfiication Un-Initiated", size_hint=(1,.1 ), color ='#000000', valign = 'middle')

        # Adding Layout Components
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Loading Tensorflow/Keras Model
        self.model = load_model('PUT YOUR MACHINE LEARNING MODEL', custom_objects={'L1Dist':L1Dist})

        # Setting up video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout

    # Run Continously to get webcam feed
    def update(self, *args):
        # Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[60:60+250, 600:600+250, :]

        # Flip Horizontal and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt = 'bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Preprocessing - Scale and Resize
    def preprocess(self, file_path):
    
        # Read(load) in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image (turning image into tensor (n-dimintional array))
        img = tf.io.decode_jpeg(byte_img)
    
        # Preprocessing steps - resizing the image to 100x100x3 (100 px to 100px in 3 channel)
        img = tf.image.resize(img, (100,100))
        # Scaling image to be inbetween 0 and 1(takes pixel value and divides it between 0 and 1)
        # traditional image pixel value 0-255
        img = img / 255.0
    
        # Returning image
        return img

        # Verification function
    def verify(self, *args):
        detection_threshold = 0.5
        verification_threshold = 0.8

        # Capture input image from our webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[60:60+250, 600:600+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))

            # Make Predictions 
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)

        detection = np.sum(np.array(results) > detection_threshold)
    
        # Verification Threshold: Proportion of positive predictions / total positive samples 
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
        verified = verification > verification_threshold
    

        # Set Verification Text
        self.verification_label.text = 'verified' if verified == True else 'Un - Verified'

        # Log out details
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        return results, verified

if __name__ == "__main__":
    Facial_RecognitionApp().run()

