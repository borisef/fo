import json
import fiftyone as fo

def read_my_coco(dataset_dir, json_path, img_fields = [], anno_fields = []):
    # read all attributes from coco , assumption - same order
    dataset_type = fo.types.COCODetectionDataset

    with open(json_path) as json_file:
        data = json.load(json_file)

    dataset = fo.Dataset.from_dir(
        # dataset_dir=dataset_dir,
        data_path=dataset_dir,
        dataset_type=dataset_type,
        labels_path=json_path,
        # name=name,
        dynamic=True
    )

    annos = data['annotations']
    imgs = data['images']

    for i,s in enumerate(dataset.view()):
        for imf in img_fields:
            s['metadata'][imf] = imgs[i][imf]
            s.save()
            for af in anno_fields:
                ann_indexes = []
                for j,ann in enumerate(annos):
                    if(ann["image_id"] == imgs[i]["id"]):
                        ann_indexes.append(j)
                #copy af
                for j, d in enumerate(ann_indexes):
                    s['detections']['detections'][j][af] = annos[ann_indexes[j]][af]
                    s.save()
                #pass

    return dataset




if __name__=="__main__":
    # The directory containing the dataset to import
    dataset_dir = "/home/borisef/data/vehicles/test/"
    json_path = "/home/borisef/data/vehicles/test/annotations_small.json"

    dataset = read_my_coco(dataset_dir,json_path,['r_2_t'],['area'])
    dataset.add_dynamic_sample_fields()


