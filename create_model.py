from torch import nn
from torch import optim
from torchvision import models

def create_model(arch, hidden_units1, hidden_units2, flower_categories, learning_rate):
    pretrained_model = arch

    model = getattr(models, pretrained_model)(pretrained=True)

    #Freeze pre-trained model parameters
    for param in model.parameters():
        param.requires_grad = False

    classifier_input_features = model.classifier[0].in_features
    hidden_units1 = hidden_units1
    hidden_units2 = hidden_units2
    flower_categories = flower_categories
    learning_rate = learning_rate

    #Model classifier architecture
    classifier = nn.Sequential(nn.Linear(classifier_input_features, hidden_units1),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(hidden_units1, hidden_units2),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(hidden_units2, flower_categories),
                               nn.LogSoftmax(dim=1))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer