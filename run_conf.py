
# Ti theloume
#
# DIffrent ML - Ola idia
# idio ML alla precpr kai pos antepskelthe to cloud
# disable on not disable preproccesing
# arithmos eikonon ola idia
# Sleep time sixnotita dipalsia tou edge1
# Tets image reduction vs accuarcy
# How quality effects accuracy kai metrcis

models = ["fasterrcnn_mobilenet_v3_large_320_fpn", "fasterrcnn_mobilenet_v3_large_fpn",
          "fasterrcnn_resnet50_fpn", "retinanet_resnet50_fpn", "maskrcnn_resnet50_fpn"]

prepproccessing_parameters = {
    'BW': [0, 1],
    "resize": ['25%', '75%'],
    "resize": ['25%', '50%', "75%"], "quality": ['25%', '50%', "75%"]}

write_file_name = "./conf_details.txt"

f = open("write_file_name.txt", "w")

# fasterrcnn is the fastes

# One
# Cloud has fasterrcnn_mobilenet_v3_large_320_fpn

# And all others have all with same time and images num


# Two
# Edges and cloud have same ML but each has a different prepprocessing 0-25 / 50-75


# Three
# Different ML for cloud with different 2 edges 0 percent and 75 percent prepro with no ML

# Four
# All have ffaster MLs but we try with many image nums 1-5 / 10-20 / 40-80


f.close()
