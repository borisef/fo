import fiftyone as fo
import fiftyone.zoo as foz
import  fiftyone.brain as fob

def annotate(dataset, classes = None, prompt = "A photo of a"):
    if(classes is None):
        classes = dataset.default_classes

    model = foz.load_zoo_model("clip-vit-base32-torch")
    dataset.apply_model(model, label_field="predictions_clip")

    # Make zero-shot predictions with custom classes
    model = foz.load_zoo_model(
        name="clip-vit-base32-torch",
        text_prompt=prompt,
        classes=classes,
    )
    dataset.apply_model(model, label_field="predictions_clip_prompt")

    return dataset