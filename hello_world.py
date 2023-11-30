import fiftyone as fo
import fiftyone.zoo as foz

foz.list_zoo_models("detection")

len(foz.list_zoo_datasets())

list_zoo_datasets_detection = foz.list_zoo_datasets('detection')

dataset = foz.load_zoo_dataset('quickstart')

session = fo.launch_app(dataset, auto=False, desktop=True)

pass #put breakpoint here