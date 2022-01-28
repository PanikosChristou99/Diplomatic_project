# Use this code do invoke pytorch to download all the models we are using to this image so its not re downloaded on building child docker images

# # fasterrcnn_mobilenet_v3_large_320_fpn, fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn, maskrcnn_resnet50_fpn

from torchvision import models

fasterrcnn_resnet50_fpn = models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True)
retinanet_resnet50_fpn = models.detection.retinanet_resnet50_fpn(
    pretrained=True)
fasterrcnn_mobilenet_v3_large_320_fpn = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
    pretrained=True)
fasterrcnn_mobilenet_v3_large_fpn = models.detection.fasterrcnn_mobilenet_v3_large_fpn(
    pretrained=True)
maskrcnn_resnet50_fpn = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
