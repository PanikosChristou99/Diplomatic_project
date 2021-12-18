# Use this code do invoke pytorch to download all the models we are using to this image so its not re downloaded on building child docker images


from torchvision import models

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = models.detection.retinanet_resnet50_fpn(pretrained=True)
