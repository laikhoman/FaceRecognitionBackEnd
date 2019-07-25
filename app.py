from flask import Flask, request, jsonify, g, url_for
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask import make_response
from flask_sqlalchemy import SQLAlchemy
from flask_httpauth import HTTPBasicAuth
from flask_marshmallow import Marshmallow
#import turicreate as tc
import sys
from queue import Queue
import os
import uuid
import logging
from flask import send_from_directory
import threading
from marshmallow import fields
from marshmallow import post_load
from passlib.apps import custom_app_context as pwd_context
import jwt
import datetime
import cv2
import numpy as np
import imutils

app = Flask(__name__)

#configure images destination folder
app.config['UPLOADED_IMAGES_DEST'] = 'images'
images = UploadSet('images', IMAGES)
configure_uploads(app, images)

#configure sqlite database
DATABASE_NAME = 'facerecognition1.sqlite'

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, DATABASE_NAME)
app.config['SECRET_KEY'] = 'test1234'
db = SQLAlchemy(app)
ma = Marshmallow(app)
auth = HTTPBasicAuth()
extList = []
fileNameList = []

#model / users is a many to many relationship, that means there's a third #table containing user id and model id

users_models = db.Table('users_models',
                        db.Column("user_id", db.Integer, db.ForeignKey('user.id')),
                        db.Column("model_id", db.Integer, db.ForeignKey('model.version'))
                        )


# model table
class Model(db.Model):
    version = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(100))
    users = db.relationship('User', secondary=users_models)


# user table
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    name = db.Column(db.String(100))
    position = db.Column(db.String(50))

    password_hash = db.Column(db.String(128))

    def __init__(self, username, name, position):
        self.username = username
        self.name = name
        self.position = position

    def hash_password(self, password):
        self.password_hash = pwd_context.encrypt(password)

    def verify_password(self, password):
        return pwd_context.verify(password, self.password_hash)

    def generate_auth_token(self, user_id, expiration = 3600):
        """
        Generates the Auth Token
        :return: string
        """
        try:
            payload = {
                'exp': datetime.datetime.utcnow() + datetime.timedelta(days=0, seconds=expiration),
                'iat': datetime.datetime.utcnow(),
                'sub': user_id
            }
            return jwt.encode(
                payload,
                app.config.get('SECRET_KEY'),
                algorithm='HS256'
            )
        except Exception as e:
            return e

    @staticmethod
    def verify_auth_token(token):
        """
        Validates the auth token
        :param auth_token:
        :return: integer|string
        """
        try:
            payload = jwt.decode(token, app.config.get('SECRET_KEY'))
            is_blacklisted_token = BlacklistToken.check_blacklist(token)
            if is_blacklisted_token:
                return 'Token blacklisted. Please log in again.'
            else:
                return payload['sub']
        except jwt.ExpiredSignatureError:
            return 'Signature expired. Please log in again.'
        except jwt.InvalidTokenError:
            return 'Invalid token. Please log in again.'


@auth.verify_password
def verify_password(username_or_token, password):
    # first try to authenticate by token
    user = User.verify_auth_token(username_or_token)
    if not user:
        # try to authenticate with username/password
        user = User.query.filter_by(username=username_or_token).first()
        if not user or not user.verify_password(password):
            return False
    g.user = user
    return True

class BlacklistToken(db.Model):
    """
    Token Model for storing JWT tokens
    """
    __tablename__ = 'blacklist_tokens'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    token = db.Column(db.String(500), unique=True, nullable=False)
    blacklisted_on = db.Column(db.DateTime, nullable=False)

    def __init__(self, token):
        self.token = token
        self.blacklisted_on = datetime.datetime.now()

    def __repr__(self):
        return '<id: token: {}'.format(self.token)

    @staticmethod
    def check_blacklist(auth_token):
        # check whether auth token has been blacklisted
        res = BlacklistToken.query.filter_by(token=str(auth_token)).first()
        if res:
            return True
        else:
            return False

# user schema
class UserSchema(ma.Schema):
    class Meta:
        fields = ('id', 'username', 'name', 'position')


# model schema
class ModelSchema(ma.Schema):
    version = fields.Int()
    url = fields.Method("add_host_to_url")
    users = ma.Nested(UserSchema, many=True)

    # this is necessary because we need to append the current host to the model url
    def add_host_to_url(self, obj):
        return request.host_url + obj.url


# initialize everything
user_schema = UserSchema()
users_schema = UserSchema(many=True)
model_schema = ModelSchema()
models_schema = ModelSchema(many=True)

if(not os.path.exists(DATABASE_NAME)):
    db.create_all()
# db.drop_all()

#error handlers
@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'not found'}), 404)

@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': 'bad request'}), 400)


@app.route("/face-recognition/api/v1.0/login", methods=['POST'])
def login():
    if not request.form:
        return make_response(jsonify({'status': 'failed', 'error': 'bad request', 'message:': 'All Fields are empty'}), 400)
    elif not 'username' in request.form.keys():
        return make_response(jsonify({'status': 'failed', 'error': 'bad request', 'message:': 'Username is required'}), 400)
    elif not 'password' in request.form.keys():
        return make_response(jsonify({'status': 'failed', 'error': 'bad request', 'message:': 'Password is required'}), 400)
    else:
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        # check if user actually exists
        # take the user supplied password, hash it, and compare it to the hashed password in database
        if not user :
            return make_response(jsonify({'status': 'failed', 'message:': 'Username not exists'}), 400)
        elif not user.verify_password(password):
            return make_response(jsonify({'status': 'failed', 'message:': 'Password is wrong'}), 400)
        # if the above check passes, then we know the user has the right credentials
        g.user = user
        token = g.user.generate_auth_token(user.id)
        return jsonify({'token': token.decode('ascii')})



@app.route('/face-recognition/api/v1.0/token')
@auth.login_required
def get_auth_token():
    token = g.user.generate_auth_token()
    return jsonify({ 'token': token.decode('ascii') })

@app.route("/face-recognition/api/v1.0/user/register", methods=['POST'])
def register_user():
    if not request.form:
        return make_response(jsonify({'status': 'failed', 'error': 'bad request', 'message:': 'All Fields are empty'}), 400)
    elif not 'username' in request.form.keys():
        return make_response(jsonify({'status': 'failed', 'error': 'bad request', 'message:': 'Username is required'}), 400)
    elif not 'name' in request.form.keys():
        return make_response(jsonify({'status': 'failed', 'error': 'bad request', 'message:': 'Name is required'}), 400)
    elif not 'password' in request.form.keys():
        return make_response(jsonify({'status': 'failed', 'error': 'bad request', 'message:': 'Password is required'}), 400)
    elif not 'confirmPassword' in request.form.keys():
        return make_response(jsonify({'status': 'failed', 'error': 'bad request', 'message:': 'Confirm Password is required'}), 400)
    else:
        username = request.form['username']
        name = request.form['name']
        password = request.form['password']
        confirmPassword = request.form['confirmPassword']
        position = request.form.get('position')
        if User.query.filter_by(username=username).first() is not None:
            return make_response(jsonify({'status': 'failed', 'message:': 'Username or email has been used'}), 400)
        if password != confirmPassword :
            return make_response(jsonify({'status': 'failed', 'message:': 'Password is not same with confirmation password'}), 400)
        if position is None:
            position = ""
        newuser = User(username, name, position)
        newuser.hash_password(password)
        db.session.add(newuser)
        db.session.commit()
        if 'photos[]' in request.files.keys():
            uploaded_images = request.files.getlist('photos[]')
            save_images_to_folder(uploaded_images, newuser, "register")
        return jsonify({'status': 'success', 'user': user_schema.dump(newuser).data})


# function to save images to image directory
def save_images_to_folder(images_to_save, user, mode):
    extList = []
    fileNameList = []
    for a_file in images_to_save:
        # save images to images folder using user id as a subfolder name
        uniqueUUID = str(uuid.uuid4())
        images.save(a_file, str(user.id), uniqueUUID + '.')

        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, 'images/' + str(user.id))
        # get extension
        for filename in os.listdir(path):
            if os.path.splitext(filename)[-2].lower() == uniqueUUID:
                ext = os.path.splitext(filename)[-1].lower()
                extList.append(ext)
                fileNameList.append(filename)


    # detect & crop image

    if mode == "register":
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, 'images/' + str(user.id))
        for filename in os.listdir(path):
            if filename.endswith('.png') or filename.endswith('.jpeg') or filename.endswith('.jpg'):
                ext = os.path.splitext(filename)[-1].lower()
                detectAndCropFaceImage('images/' + str(user.id) + '/' + filename, ext, 300, 300, 0.6, user.id, '')
    elif mode == "addPhotos":
        j = 0
        for image in images_to_save:
            #detectAndCropFaceImage('', '', image, 300, 300, 0.6, user.id, extList[j])
            detectAndCropFaceImage('images/' + str(user.id) + '/' + fileNameList[j], '', 300, 300, 0.6, user.id, extList[j])
            j = j+1

    # get the last trained model
    model = Model.query.order_by(Model.version.desc()).first()
    if model is not None:
        # increment the version
        Queue().put(model.version + 1)
    else:
        # create first version
        Queue().put(1)

@app.route('/face-recognition/api/v1.0/user/add_more_photo', methods=['POST'])
@auth.login_required
def add_more_photo():
    if not request.form:
        return make_response(jsonify({'status': 'failed', 'message:': 'All Fields are empty'}), 400)
    elif not 'username' in request.form.keys():
        return make_response(jsonify({'status': 'failed', 'message:': 'Username is required'}), 400)
    elif not 'photos[]' in request.files.keys():
        return make_response(jsonify({'status': 'failed', 'message': 'No photos uploaded'}), 400)
    else:
        username = request.form['username']
        user = User.query.filter_by(username=username).first()
        uploaded_images = request.files.getlist('photos[]')
        save_images_to_folder(uploaded_images, user, "addPhotos")
        return jsonify({'status': 'success', 'info': 'photo(s) added'})


@app.route('/face-recognition/api/v1.0/resource')
@auth.login_required
def get_resource():
    return jsonify({ 'data': 'Hello, %s!' % g.user.username })

@app.route("/face-recognition/api/v1.0/model/info" , methods=['GET'])
def get_model_info():
    models_schema.context['request'] = request
    model = Model.query.order_by(Model.version.desc()).first()
    if model is None:
        return make_response(jsonify({'status': 'failed', 'error': 'model is not ready'}), 400)
    else:
        return jsonify({'status' : 'success', 'model' : model_schema.dump(model).data})

# serve models
@app.route('/models/')
def download(filename):
    return send_from_directory('models', filename, as_attachment=True)

def detectAndCropFaceImage(imagePath, ext, width, height, confidenceValue, userID, extension):
    imageData = None
    if extension != '':
        imageData = cv2.imread(imagePath)
    else:
        imageData = cv2.imread(imagePath)

    image = imutils.resize(imageData, width=width)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (width, height)), 1.0, (width, height),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image

    protoPath = os.path.join(basedir, "deploy.prototxt")
    modelPath = os.path.join(basedir, "res10_300x300_ssd_iter_140000.caffemodel")
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    detector.setInput(imageBlob)
    detections = detector.forward()

    # ensure at least one face was found
    if len(detections) > 0:
        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        # ensure that the detection with the largest probability also
        # means our minimum probability test (thus helping filter out
        # weak detections)
        if confidence > confidenceValue:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI and grab the ROI dimensions
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                return

            if extension != '':
                saveDetectedCropImage(userID, extension, face)
            else:
                saveDetectedCropImage(userID, ext, face)
            return face

def saveDetectedCropImage(userID, ext, face):
    dirname = os.path.dirname(__file__)
    #path = os.path.join(dirname, 'detectedCropImage/')
    if not os.path.exists('detectedCropImage/' + str(userID)):
        os.makedirs('detectedCropImage/' + str(userID))
    cv2.imwrite(os.path.join(dirname, 'detectedCropImage/' + str(userID) + '/' + str(uuid.uuid4()) + ext), face)
    cv2.waitKey(0)

def train_model():
    while True:
        #get the next version
        version = queue.get()
        logging.debug('loading images')
        data = tc.image_analysis.load_images('images', with_path=True)

        # From the path-name, create a label column
        data['label'] = data['path'].apply(lambda path: path.split('/')[-2])

        # use the model version to construct a filename
        filename = 'Faces_v' + str(version)
        mlmodel_filename = filename + '.mlmodel'
        models_folder = 'models/'

        # Save the data for future use
        data.save(models_folder + filename + '.sframe')

        result_data = tc.SFrame( models_folder + filename +'.sframe')
        train_data = result_data.random_split(0.8)

        #the next line starts the training process
        model = tc.image_classifier.create(train_data, target='label', model='resnet-50', verbose=True)

        db.session.commit()
        logging.debug('saving model')
        model.save( models_folder + filename + '.model')
        logging.debug('saving coremlmodel')
        model.export_coreml(models_folder + mlmodel_filename)

        # save model data in database
        modelData = Model()
        modelData.url = models_folder + mlmodel_filename
        classes = model.classes
        for userId in classes:
            user = User.query.get(userId)
            if user is not None:
                modelData.users.append(user)
        db.session.add(modelData)
        db.session.commit()
        logging.debug('done creating model')
        # mark this task as done
        queue.task_done()

@app.route('/')
def hello_world():
    return 'Hello World!'


#configure queue for training models
queue = Queue(maxsize=0)
thread = threading.Thread(target=train_model, name='TrainingDaemon')
thread.setDaemon(False)
thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)

# configure logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] - %(threadName)-10s : %(message)s')



