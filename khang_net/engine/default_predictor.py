import torch

from detectron2.config.instantiate import instantiate
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

class DefaultPredictor:
    """
    Perform as same as DefaultPredictor in Detectron2 but used for Lazy config
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = instantiate(self.cfg.model)
        self.model.eval()
        self.model.to(self.cfg.train.device)

        checkpointer =  DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.train.init_checkpoint)

        # Hard code this line: aug = T.ResizeShortestEdge, see DefaultPredictor in detectron2 for more details
        self.aug = instantiate(self.cfg.dataloader.test.mapper.augmentations[0])
        self.input_format = self.cfg.model.input_format
        assert self.input_format in ["RGB", "BGR"]

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image.to(self.cfg.train.device)

            inputs = {"image": image, "height": height, "width": width}

            predictions = self.model([inputs])[0]
            return predictions