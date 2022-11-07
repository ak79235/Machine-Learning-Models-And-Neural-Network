from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json
from PIL import Image
import cv2
import keras
from joblib import Parallel, delayed
import joblib
import tensorflow as tf
from keras.models import load_model
  


# app = Flask(__name__)
# print("Loading model...")
# new_model = load_model('C:\\Users\\ak792\\Desktop\\Yethi\\captcha_OCR\\captcha_ocr.h5')
# new_model.summary()
# I need to load the captcha_ocr,h5 model here.
new_model = load_model('C:\\Users\\ak792\\Desktop\\Yethi\\captcha_OCR\\saved_model\\my_model')
new_model.summary()
