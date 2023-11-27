from models.vgg import vgg16

MODELS = {
    'vgg16': vgg16,
}


def build_model(model_name, num_classes):
    assert model_name in MODELS.keys()
    model = MODELS[model_name](num_classes)
    return model
