from app import app
from flask import render_template
from flask import request, redirect, url_for, send_from_directory, send_file
import os
from werkzeug.utils import secure_filename
# import doInference


import keras
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

import cv2
import matplotlib.pyplot as plt
import copy
# import os

# @app.route("/")
# def index():
#   return render_template("public/index.html")

# @app.route("/about")
# def about():
#   return render_template("public/about.html")


curPath = os.getcwd()
app.config["IMAGE_UPLOADS"] = curPath + "/app/static/img/"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
app.config["MAX_IMAGE_FILESIZE"] = 2 * 1024 * 1024


@app.route("/")
def index():
  return render_template("public/upload_image.html")


def allowed_image(filename):
  if not "." in filename:
    return False

  ext = filename.rsplit(".", 1)[1]

  if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
    return True
  else:
    return False


def allowed_image_filesize(filesize):
  if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
    return True
    
  else:
    return False


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():

  print("here")
  if request.method == "POST":

    if request.files:

      print("request is: \n")
      print(request.files)
      image = request.files["image"]

      if image.filename == "":
        print("No filename")
        return redirect(request.url)

      if allowed_image(image.filename):
        filename = secure_filename(image.filename)

        imageSavePath = os.path.join(app.config["IMAGE_UPLOADS"])
        print("imageSavePath: "  + imageSavePath)
        image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

        print("Image saved")

        return redirect(url_for("uploaded_file", filename=filename))

      else:
        print("That file extension is not allowed")
        return redirect(request.url)

  return render_template("public/upload_image.html")



def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


# In[40]:


def readImageAsArray(imagePath):
    img = cv2.imread(imagePath)
    return img

def showImage(imageArray):
    # plots image array as colored image 
    plt.imshow(cv2.cvtColor(imageArray, cv2.COLOR_BGR2RGB))


def makeInference():
    print("called to make inference\n")
    model = keras.models.load_model("model-save.h5", 
                                custom_objects={'bce_dice_loss': bce_dice_loss,
                                                'dice_coef': dice_coef,
                                               })

    test_img_path = os.getcwd() + "/static/img/00dbd3c.jpg"
    print("image path: ")
    print(test_img_path)
    # test_img_path = "/app/static/img/00dbd3c.jpg"
    imageAsArray = readImageAsArray(test_img_path)    
    imageAsArrayCropped = cv2.resize(imageAsArray, (480, 320))   
    imageAsArrayExpanded = np.expand_dims(imageAsArrayCropped, axis=0)
    modelPrediction = model.predict(imageAsArrayExpanded)
    predictionMasks = modelPrediction[0, ].round().astype(int)

    
    maskedImage = copy.deepcopy(imageAsArrayCropped)
    print(maskedImage.shape)

    for row in range(320):
        for col in range(480):
            if(predictionMasks[row][col][0]==1):
                maskedImage[row][col] = [255,0,0]
            elif(predictionMasks[row][col][1]==1):
                maskedImage[row][col] = [0,255,0]
            elif(predictionMasks[row][col][2]==1):
                maskedImage[row][col] = [0,0,255]
            elif(predictionMasks[row][col][3]==1):
                maskedImage[row][col] = [100,100,100]
            else:
                maskedImage[row][col] = [0,0,0]

    
    plt.imshow(imageAsArrayCropped, cmap='gray')
    plt.imshow(maskedImage, cmap='gist_rainbow_r', alpha=0.5)
    plt.savefig('segmented-cloud.jpeg')


@app.route("/show-image/<filename>")
def uploaded_file(filename):
  print("call here with filename: " + filename)
  print("calling inference file")
#   segmentedFileName = "segmented-cloud.jpeg"
# #   doInference.makeInference()
#   makeInference()
  return render_template("public/show_image.html", filename=filename)

@app.route("/upload-image/<filename>")
def send_file(filename):
#   segmentedFileName = "segmented-cloud.jpeg"
# #   doInference.makeInference()
#   makeInference()
  return send_from_directory(app.config["IMAGE_UPLOADS"], filename)























@app.route("/sign-up", methods=["GET", "POST"])
def sign_up():
  if request.method == "POST":

    req = request.form

    username = req.get("username")
    email = request.form.get("email")
    password = request.form["password"]

    missing = list()

    for k, v in req.items():
      if v == "":
        missing.append(k)

    if missing:
      feedback = f"Missing fields for {', '.join(missing)}"
      return render_template("public/sign_up.html", feedback=feedback)

    return redirect(request.url)

  return render_template("public/sign_up.html")

@app.route("/profile/<username>")
def profile(username):

  users = {
    "mitsuhiko": {
        "name": "Armin Ronacher",
        "bio": "Creatof of the Flask framework",
        "twitter_handle": "@mitsuhiko"
    },
    "gvanrossum": {
        "name": "Guido Van Rossum",
        "bio": "Creator of the Python programming language",
        "twitter_handle": "@gvanrossum"
    },
    "elonmusk": {
        "name": "Elon Musk",
        "bio": "technology entrepreneur, investor, and engineer",
        "twitter_handle": "@elonmusk"
    }
  }

  user = None

  if username in users:
    user = users[username]

  return render_template("public/profile.html", username=username, user=user)

@app.route("/multiple/<foo>/<bar>/<baz>")
def multiple(foo, bar, baz):

    print(f"foo is {foo}")
    print(f"bar is {bar}")
    print(f"baz is {baz}")


    return f"foo is {foo}, bar is {bar}, baz is {baz}"

@app.route("/jinja")
def jinja():
  # Strings
  my_name = "Julian"

  # Integers
  my_age = 26

  # Lists
  langs = ["Python", "JavaScript", "Bash", "Ruby", "C", "Rust"]

  # Dictionaries
  friends = {
    "Tony": 43,
    "Cody": 28,
    "Amy": 26,
    "Clarissa": 23,
    "Wendell": 39
  }

  # Tuples
  colors = ("Red", "Blue")

  # Booleans
  cool = False

  # Classes
  class GitRemote:
    def __init__(self, name, description, domain):
      self.name = name
      self.description = description 
      self.domain = domain

    def clone(self, repo):
        return f"Cloning into {repo}"

  my_remote = GitRemote(
    name="Learning Flask",
    description="Learn the Flask web framework for Python",
    domain="https://github.com/Julian-Nash/learning-flask.git"
  )

# Functions
  def repeat(x, qty=1):
    return x * qty

  return render_template(
    "public/jinja.html", my_name=my_name, my_age=my_age, langs=langs,
    friends=friends, colors=colors, cool=cool, GitRemote=GitRemote, 
    my_remote=my_remote, repeat=repeat
  )