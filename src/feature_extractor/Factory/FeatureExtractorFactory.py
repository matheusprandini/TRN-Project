from FeatureExtractorEntity.MobileNetV2 import MobileNetV2Model
from FeatureExtractorEntity.ResNet50V2Model import ResNet50V2Model
from FeatureExtractorEntity.VGG16Model import VGG16Model

class FeatureExtractorFactory():
    modelNameToFeatureExtractorModel = {
        "MobileNetV2": MobileNetV2Model(),
        "ResNet50V2": ResNet50V2Model(),
        "VGG16": VGG16Model()
    }

    def get_model(self, name, weights="imagenet", includeTop=True, inputShape=(224,224,3)):
        if name not in self.modelNameToFeatureExtractorModel:
            return None
        return self.modelNameToFeatureExtractorModel[name].build_model(weights, includeTop, inputShape)