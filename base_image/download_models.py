# Use this code do invoke pytorch to download all the models we are using to this image so its not re downloaded on building child docker images

# models = ["fasterrcnn_mobilenet_v3_large_320_fpn", "fasterrcnn_mobilenet_v3_large_fpn", "fasterrcnn_resnet50_fpn", "retinanet_resnet50_fpn", "maskrcnn_resnet50_fpn" , "fcos_resnet50_fpn" ,"keypointrcnn_resnet50_fpn","ssdlite320_mobilenet_v3_large" , "ssd300_vgg16" ]
from torchvision import models

model1 = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model1 = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
model1 = models.detection.retinanet_resnet50_fpn(pretrained=True)
model1 = models.detection.fcos_resnet50_fpn(pretrained=True)
model1 = models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model1 = models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
model1 = models.detection.ssd300_vgg16(pretrained=True)