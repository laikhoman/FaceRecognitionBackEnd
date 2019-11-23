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
import dataset
import tensorflow as tf
from tensorflow import set_random_seed
import train
set_random_seed(2)
import time

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
testingFileName = ""

train_path = 'detectedCropImage'
img_size = 128
num_channels = 3

validation_size = 0.2
batch_size = 2
total_iterations = 0

##Network graph params
filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 128

PADDING = 50

#model / users is a many to many relationship, that means there's a third #table containing user id and model id

users_models = db.Table('users_models',
                        db.Column("user_id", db.Integer, db.ForeignKey('user.id'),nullable=False),
                        db.Column("model_id", db.Integer, db.ForeignKey('model.version'),nullable=False),
                        db.PrimaryKeyConstraint('user_id', 'model_id'))

# class UsersModel(db.Model):
#     user_id = db.Column("user_id", db.Integer, db.ForeignKey('user.id'))
#     model_id = db.Column("model_id", db.Integer, db.ForeignKey('model.version'))

# model table
class Model(db.Model):
    version = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(100))
    users = db.relationship('User', secondary=users_models, backref = 'models')

    def __init__(self, version, url):
        self.version = version
        self.url = url

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
    testingFileName = ""
    for a_file in images_to_save:
        # save images to images folder using user id as a subfolder name
        path = ""
        if mode == "register" or mode == "addPhotos" :
            uniqueUUID = str(uuid.uuid4())
            images.save(a_file, str(user.id), uniqueUUID + '.')
            dirname = os.path.dirname(__file__)
            path = os.path.join(dirname, 'images/' + str(user.id))
        elif mode == "test":
            uniqueUUID = str(uuid.uuid4())
            ext = os.path.splitext(a_file.filename)[1][1:]
            extList.append(ext)
            dirname = os.path.dirname(__file__)
            path = os.path.join(dirname, 'testingImages/'+ str(user.id))
            if not os.path.exists('testingImages/' + str(user.id)):
                os.makedirs('testingImages/' + str(user.id))
            a_file.save(os.path.join(dirname, 'testingImages/'+ str(user.id) + '/' + uniqueUUID + '.' + ext))

        # get extension and filenameList
        for filename in os.listdir(path):
            if os.path.splitext(filename)[-2].lower() == uniqueUUID:
                ext = os.path.splitext(filename)[-1].lower()
                if mode == "register" or mode == "addPhotos" :
                    extList.append(ext)
                fileNameList.append(filename)

    # detect & crop image

    if mode == "register":
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, 'images/' + str(user.id))
        for filename in os.listdir(path):
            if filename.endswith('.png') or filename.endswith('.jpeg') or filename.endswith('.jpg'):
                ext = os.path.splitext(filename)[-1].lower()
                detectAndCropFaceImage('images/' + str(user.id) + '/' + filename, ext, 300, 300, 0.6, user.id, '', mode)
        allUsers = User.query.order_by(User.id.asc()).all()
        num_classes = len(allUsers)
        print(num_classes)

        if num_classes > 1:
            model = Model.query.order_by(Model.version.desc()).first()
            print("model", model)
            if model is not None:
                # increment the version
                latestModelVersion = (Model.query.order_by(Model.version.desc()).first()).version
                print("version up: ", latestModelVersion + 1)
                queue.put(latestModelVersion + 1)
            else:
                # create first version
                queue.put(1)
            # print("queue size: ", queue.qsize())
            # if queue.qsize() == 0:
            #     queue.put(1)
            # else:
            #     latestModelVersion = queue.get()
            #     print("version up: ", latestModelVersion + 1)
            #     queue.put(latestModelVersion + 1)
    elif mode == "addPhotos":
        j = 0
        for image in images_to_save:
            detectAndCropFaceImage('images/' + str(user.id) + '/' + fileNameList[j], '', 300, 300, 0.6, user.id, extList[j], mode)
            j = j + 1
        allUsers = User.query.order_by(User.id.asc()).all()
        num_classes = len(allUsers)
        print(num_classes)
        if num_classes > 1:
            model = Model.query.order_by(Model.version.desc()).first()
            print("model", model)
            if model is not None:
                # increment the version
                latestModelVersion = (Model.query.order_by(Model.version.desc()).first()).version
                print("version up: ", latestModelVersion)
                queue.put(latestModelVersion + 1)
            else:
                # create first version
                queue.put(1)
            # if queue.qsize() == 0:
            #     queue.put(1)
            # else:
            #     latestModelVersion = queue.get()
            #     print("version up: ", latestModelVersion + 1)
            #     queue.put(latestModelVersion + 1)
    elif mode == "test":
        j = 0
        for image in images_to_save:
            detectAndCropFaceImage('testingImages/' + str(user.id) + '/' + fileNameList[j], '', 300, 300, 0.6, user.id,extList[j], mode)
            j = j + 1

@app.route('/private')
@auth.login_required
def private_page():
    return "Only for authorized people!"

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

def detectAndCropFaceImage(imagePath, ext, width, height, confidenceValue, userID, extension, mode):
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
            x1 = startX - PADDING
            y1 = startY - PADDING
            x2 = endX + PADDING
            y2 = endY + PADDING
            # extract the face ROI and grab the ROI dimensions
            #face = image[startY:endY, startX:endX]
            face = image[max(0, startY):min(endY, y2), max(0, startX):min(endX, x2)]
            (fH, fW) = face.shape[:2]
            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                return
            if mode == "test":
                saveDetectedCropTestingImage(userID, extension, face)
            else :
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

def saveDetectedCropTestingImage(userID, ext, face):
    global testingFileName
    dirname = os.path.dirname(__file__)
    if not os.path.exists('detectedCropTestingImage/' + str(userID)):
        os.makedirs('detectedCropTestingImage/' + str(userID))
    strUUID = str(uuid.uuid4())
    cv2.imwrite(os.path.join(dirname, 'detectedCropTestingImage/' + str(userID) + '/' + strUUID + '.' + ext), face)
    cv2.waitKey(0)

    # continue testingProcess
    testingFileName = 'detectedCropTestingImage/' + str(userID) + '/' + strUUID + '.' + ext

@app.route("/face-recognition/api/v1.0/training" , methods=['GET'])
def trainingProcess(num_iteration = 1000):
    while True:
        version = queue.get()
        print("version ", version)
        allUsers = User.query.order_by(User.id.asc()).all()
        num_classes = len(allUsers)
        classes = []
        for i in range(num_classes):
            classes.append(str(allUsers[i].id))
        print(classes)
        print(len(classes))

        # convolutional process
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
        ## labels
        y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
        y_true_cls = tf.argmax(y_true, dimension=1)

        layer_conv1 = train.create_convolutional_layer(input=x,
                                                       num_input_channels=num_channels,
                                                       conv_filter_size=filter_size_conv1,
                                                       num_filters=num_filters_conv1)
        layer_conv2 = train.create_convolutional_layer(input=layer_conv1,
                                                       num_input_channels=num_filters_conv1,
                                                       conv_filter_size=filter_size_conv2,
                                                       num_filters=num_filters_conv2)

        layer_conv3 = train.create_convolutional_layer(input=layer_conv2,
                                                       num_input_channels=num_filters_conv2,
                                                       conv_filter_size=filter_size_conv3,
                                                       num_filters=num_filters_conv3)

        layer_flat = train.create_flatten_layer(layer_conv3)

        layer_fc1 = train.create_fc_layer(input=layer_flat,
                                          num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                                          num_outputs=fc_layer_size,
                                          use_relu=True)

        layer_fc2 = train.create_fc_layer(input=layer_fc1,
                                          num_inputs=fc_layer_size,
                                          num_outputs=num_classes,
                                          use_relu=False)

        y_pred = tf.nn.softmax(layer_fc2, name='y_pred')
        y_pred_cls = tf.argmax(y_pred, dimension=1)
        session.run(tf.global_variables_initializer())
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
        cost = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
        global total_iterations

        # use the model version to construct a filename
        filename = 'Faces_v' + str(version)
        if not os.path.exists('models/'):
            os.makedirs('models/')
        models_folder = 'models/'

        # save model data in database
        modelData = Model(version, models_folder + filename)
        for userId in classes:
            if userId != '.DS_Store':
                user = User.query.filter_by(id=int(userId)).first()
                if user is not None:
                    modelData.users.append(user)
        db.session.add(modelData)
        db.session.commit()

        for i in range(total_iterations, total_iterations + num_iteration):
            x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
            x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)
            feed_dict_tr = {x: x_batch, y_true: y_true_batch}
            feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}
            session.run(optimizer, feed_dict=feed_dict_tr)
            if i % int(data.train.num_examples / batch_size) == 0:
                val_loss = session.run(cost, feed_dict=feed_dict_val)
                epoch = int(i / int(data.train.num_examples / batch_size))
                # show progress
                acc = session.run(accuracy, feed_dict=feed_dict_tr)
                val_acc = session.run(accuracy, feed_dict=feed_dict_val)
                msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
                print(msg.format(epoch + 1, acc, val_acc, val_loss))
                saver.save(session, models_folder + filename)

        total_iterations += num_iteration
        logging.debug('done creating model')
        queue.task_done()
    return jsonify({'status': 'success'})

@app.route("/face-recognition/api/v1.0/user/testing" , methods=['POST'])
@auth.login_required
def testing():
    if 'photos[]' in request.files.keys():
        username = request.form['username']
        user = User.query.filter_by(username=username).first()
        uploaded_images = request.files.getlist('photos[]')
        save_images_to_folder(uploaded_images, user, "test")
        result = testingProcess(testingFileName)
        if result[1] != username :
            return jsonify({'status': result[0], 'predicted': result[1], 'confidence': result[2], 'isRecognized': 'false'})
        return jsonify({'status': result[0], 'predicted': result[1], 'confidence': result[2], 'isRecognized': 'true'})
    return jsonify({'status': 'failed'})

def testingProcess(pathFile):
    # First, pass the path of the image
    dir_path = os.path.dirname(__file__)
    image_path = pathFile
    filename = os.path.join(dir_path, image_path)
    image_size = 128
    num_channels = 3
    images = []
    # Reading the image using OpenCV
    image = cv2.imread(filename)
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0 / 255.0)
    # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size, image_size, num_channels)

    ## Let us restore the saved model
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    latestModelVersion = (Model.query.order_by(Model.version.desc()).first()).version
    saver = tf.train.import_meta_graph('models/' + 'Faces_v' + str(latestModelVersion) + '.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('models/./'))

    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")
    ## Let's feed the images to the input placeholders
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")

    allUsers = User.query.order_by(User.id.asc()).all()
    num_classes = len(allUsers)
    classes = []
    for i in range(num_classes):
        classes.append(str(allUsers[i].id))

    y_test_images = np.zeros((1, num_classes))
    ### Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    maxIndex = np.argmax(result)
    id = classes[maxIndex]
    predicted = User.query.filter_by(id=int(id)).first()
    return ["success", predicted.username, str(result[0,maxIndex])]

@app.route('/')
def hello_world():
    return 'Hello World!'

# configure queue for training models
queue = Queue(maxsize=0)
thread = threading.Thread(target=trainingProcess, name='TrainingDaemon')
thread.setDaemon(False)
thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)

# configure logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] - %(threadName)-10s : %(message)s')
