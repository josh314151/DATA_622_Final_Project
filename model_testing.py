import torchvision
from torchvision.models import list_models, get_model_weights, get_model

classification_models = list_models(module=torchvision.models)

i = 0
for model in classification_models:
    test_model = get_model(model)
    model_weights = get_model_weights(model)
    print(f"Testing model {++i} of {len(classification_models)}Model: {model}, Weights: {model_weights}")
    test_model.eval()

    #TODO: test each model against the dataset