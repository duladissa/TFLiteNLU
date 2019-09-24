#!/usr/bin/python3
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from json import dumps
from classify import Classify
import os
import datetime

app = Flask(__name__)
api = Api(app)


class NLU(Resource):
    classify = Classify()
    def get(self):
        text = request.args.get('text')
        startTime = datetime.datetime.now()
        intent = self.classify.classify(text)
        endTime = datetime.datetime.now()

        delta = endTime - startTime
        deltaMil = int(delta.total_seconds() * 1000)
        return {'intent': intent, 'time': ("%s%s" % (deltaMil, "ms"))}
    
    def post(self):
        return {'status':'POST'}


api.add_resource(NLU, '/query') # Route_1

port = int(os.environ.get("PORT", 5000))

print("%s%s" % ("PORT =", port))
if __name__ == '__main__':
     app.run(host='0.0.0.0', port=port)
