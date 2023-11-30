import torch
import torchvision
import fiftyone as fo
from PIL import Image
from torchvision.transforms import functional as func




def add_detect_to_dataset(dataset, num_samples = None, map_labels={}, predictions_label = "predictions"):
    if(num_samples is None):
        num_samples = len(dataset)

    # Run the model on GPU if it is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load a pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()

    print("Model ready")

    # Choose a random subset of 100 samples to add predictions to
    predictions_view = dataset.take(num_samples, seed=51)

    # Get class list
    classes = dataset.default_classes

    # Add predictions to samples
    with fo.ProgressBar() as pb:
        for sample in pb(predictions_view):
            # Load image
            image = Image.open(sample.filepath)
            image = func.to_tensor(image).to(device)
            c, h, w = image.shape

            # Perform inference
            preds = model([image])[0]
            labels = preds["labels"].cpu().detach().numpy()
            scores = preds["scores"].cpu().detach().numpy()
            boxes = preds["boxes"].cpu().detach().numpy()

            # Convert detections to FiftyOne format
            detections = []

            keys = map_labels.keys()
            true_labels = []
            true_scores = []
            true_boxes = []
            if(len(keys) == 0):
                true_labels = labels
                true_scores = scores
                true_boxes = boxes
            else:
                for label, score, box in zip(labels, scores, boxes):
                    if(label in keys):
                        true_labels.append(map_labels[label])
                        true_scores.append(score)
                        true_boxes.append(box)

            for label, score, box in zip(true_labels, true_scores, true_boxes):
                # Convert to [top-left-x, top-left-y, width, height]
                # in relative coordinates in [0, 1] x [0, 1]
                x1, y1, x2, y2 = box
                rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

                detections.append(
                    fo.Detection(
                        label=classes[label],
                        bounding_box=rel_box,
                        confidence=score
                    )
                )

            # Save predictions to dataset
            sample[predictions_label] = fo.Detections(detections=detections)
            sample.save()

        return dataset
