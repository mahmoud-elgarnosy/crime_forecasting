from torchinfo import summary
import torchvision.models as models

model = models.vgg16()
summary(model, (3, 224, 224))
