import json

class JsonHandler(object):

    @staticmethod
    def read_json(filepath):
        with open(filepath) as jsonFile:    
            return json.load(jsonFile)