import fiftyone as fo
import fiftyone.zoo as foz
import torch
import torchvision
import use_model_to_detect
from fiftyone import ViewField as F

from read_my_coco import  read_my_coco




# A name for the dataset
name = "vehicles"

# The directory containing the dataset to import
dataset_dir = "/home/borisef/data/vehicles/test/"
json_path = "/home/borisef/data/vehicles/test/annotations_small.json"


# The type of the dataset being imported
dataset_type = fo.types.COCODetectionDataset

dataset = fo.Dataset.from_dir(
    #dataset_dir=dataset_dir,
    data_path=dataset_dir,
    dataset_type=dataset_type,
    labels_path=json_path,
    #name=name,
    dynamic=True
)

dataset = read_my_coco(dataset_dir,json_path,[],[])
# Compute metadata so we can reference image height/width in our view
dataset.compute_metadata()

#RUN DETECTION MODEL (see https://docs.voxel51.com/recipes/adding_detections.html)
map_labels_coco2vehicles = {6:2,3:3,4:4,8:5}
dataset = use_model_to_detect.add_detect_to_dataset(dataset,num_samples=126, map_labels=map_labels_coco2vehicles,predictions_label="predictions_faster")

#RUN MMDETECTION MODEL NOT WORKING YET
# config = "/home/borisef/projects/mm/mmdetection/boris3/config/faster-rcnn_r50-caffe_fpn_ms-1x_coco_FULL.py"
# ckpt = "/home/borisef/projects/mm/mmdetection/checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-5324cff8.pth"
# use_model_to_detect.add_mmdetect_to_dataset(dataset,config, ckpt, num_samples=None,map_labels = map_labels_coco2vehicles,predictions_label="mm_predictions_faster")



#evaluate
high_conf_view = dataset.filter_labels("predictions_faster", F("confidence") > 0.75, only_matches=False)
eval_results = fo.evaluate_detections(high_conf_view,pred_field="predictions_faster",gt_field = "detections",eval_key = 'eval',method=None,classwise=True)
eval_results.print_report()

# compute uniqness
import fiftyone.brain as fob
fob.compute_uniqueness(dataset)
fob.compute_similarity(dataset, brain_key="image_sim") # will be able to find similar images by image_sim
# Image embeddings
fob.compute_visualization(dataset, brain_key="img_viz")
# Object patch embeddings
fob.compute_visualization(
    dataset, patches_field="detections", brain_key="gt_viz"
)

# compute duplicates
duplicates = fob.compute_exact_duplicates(dataset)

#TODO: add tag has_duplicate to each sample




# Only contains detections with confidence >= 0.75
#high_conf_view = dataset.filter_labels("predictions_faster", F("confidence") > 0.75)


session = fo.launch_app(dataset, auto=False, desktop=True)
pass #put breakpoint here

#view of all
all_view = dataset.view()


#EXPORT DATASET
#dataset.export("/home/borisef/temp",dataset_type = fo.types.CVATImageDataset) # or any format

#PLOTS
cm = eval_results.plot_confusion_matrix()
session.plots.attach(cm) # only jupyter
cm.show() #won't work
cm.save("/home/borisef/temp/cm.png")

