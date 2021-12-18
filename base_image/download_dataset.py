# Use this code do invoke fiftyone to download all the coco dataset to this image so its not re downloaded on building child docker images

import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    dataset_name="coco-2017-val",
    max_samples=500,

)

dataset.persistent = True
