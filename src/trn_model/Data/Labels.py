from utils.JsonHandler import JsonHandler

class Labels():

    classes = JsonHandler.read_json("../../conf/labels.json")["classes"]

    @staticmethod
    def get_classes():
        return Labels.classes
