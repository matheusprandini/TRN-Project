from Models.TRNCNN import TRNCNN
from Models.TRNConvLSTM import TRNConvLSTM
from Models.TRNLSTM import TRNLSTM
from Models.TRNGRU import TRNGRU

class TRNFactory():
    typeToModel = {
        "lstm": TRNLSTM,
        "gru": TRNGRU,
        "cnn": TRNCNN,
        "convlstm": TRNConvLSTM
    }

    @staticmethod
    def get_model(typeName, featureExtractorName):
        if typeName not in TRNFactory.typeToModel:
            return None
        return TRNFactory.typeToModel[typeName](featureExtractorName)