from app import app
from flask import render_template
from flask import request, redirect
import os
from werkzeug.utils import secure_filename

@app.route("/")
def index():
  return render_template("public/index.html")

@app.route("/about")
def about():
  return render_template("public/about.html")

app.config["IMAGE_UPLOADS"] = "/home/ece-student/app/app/static/img/"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
app.config["MAX_IMAGE_FILESIZE"] = 0.5 * 1024 * 1024

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

  if request.method == "POST":

    if request.files:

      if "filesize" in request.cookies:

        if not allowed_image_filesize(request.cookies["filesize"]):
          print("Filesize exceeded maximum limit")
          return redirect(request.url)

        image = request.files["image"]

        if image.filename == "":
          print("No filename")
          return redirect(request.url)

        if allowed_image(image.filename):
          filename = secure_filename(image.filename)

          image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

          print("Image saved")

          return redirect(request.url)

        else:
          print("That file extension is not allowed")
          return redirect(request.url)

  return render_template("public/upload_image.html")

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