#!/usr/bin/python3
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from json import dumps
from classify import Classify

app = Flask(__name__)
api = Api(app)


class Employees(Resource):
    classify = Classify()
    def get(self):
        return self.classify.processQuery()
    
    def post(self):
        return {'status':'POST'}


api.add_resource(Employees, '/query') # Route_1


if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5000)
