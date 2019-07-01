from flask import Flask, request, jsonify, g, url_for
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask import make_response
from flask_sqlalchemy import SQLAlchemy
from flask_httpauth import HTTPBasicAuth
from flask_marshmallow import Marshmallow
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
from itsdangerous import (TimedJSONWebSignatureSerializer
                          as Serializer, BadSignature, SignatureExpired)

app = Flask(__name__)

#configure images destination folder
app.config['UPLOADED_IMAGES_DEST'] = 'images'
images = UploadSet('images', IMAGES)
configure_uploads(app, images)

#configure sqlite database
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'facerecognition1.sqlite')
app.config['SECRET_KEY'] = 'test1234'
db = SQLAlchemy(app)
ma = Marshmallow(app)
auth = HTTPBasicAuth()

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

    def generate_auth_token(self, expiration = 3600):
        s = Serializer(app.config['SECRET_KEY'], expires_in=expiration)
        return s.dumps({'id': self.id})

    @staticmethod
    def verify_auth_token(token):
        s = Serializer(app.config['SECRET_KEY'], expires_in=3600)
        try:
            data = s.loads(token)
        except SignatureExpired:
            print('Signature Expired')
            return None
        except BadSignature:
            print('Bad Signature')
            return None
        user = User.query.get(data['id'])
        return user


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
db.create_all()
# db.drop_all()

#error handlers
@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'not found'}), 404)

@app.errorhandler(400)
def not_found(error):
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
        token = g.user.generate_auth_token()
        return jsonify({'token': token.decode('ascii')})



@app.route('/face-recognition/api/v1.0/token')
@auth.login_required
def get_auth_token():
    token = g.user.generate_auth_token()
    return jsonify({ 'token': token.decode('ascii') })

@app.route("/face-recognition/api/v1.0/user/register", methods=['POST'])
def register_user():
    print(request.form.keys())
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
            save_images_to_folder(uploaded_images, newuser)
        return jsonify({'status': 'success', 'user': user_schema.dump(newuser).data})


# function to save images to image directory
def save_images_to_folder(images_to_save, user):
    for a_file in images_to_save:
        # save images to images folder using user id as a subfolder name
        images.save(a_file, str(user.id), str(uuid.uuid4()) + '.')

    # get the last trained model
    model = Model.query.order_by(Model.version.desc()).first()
    if model is not None:
        # increment the version
        Queue().put(model.version + 1)
    else:
        # create first version
        Queue().put(1)

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

@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)

# configure logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] - %(threadName)-10s : %(message)s')



