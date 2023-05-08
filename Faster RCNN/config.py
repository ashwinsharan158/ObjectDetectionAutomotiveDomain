import torch
BATCH_SIZE = 16
RESIZE_TO = 416
NUM_EPOCHS = 200
NUM_WORKERS = 4
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAIN_DIR = './train'
VALID_DIR = './val'
CLASSES = [
    '__background__', 'vehicle', 'person', 'stop sign'
]
NUM_CLASSES = len(CLASSES)

VISUALIZE_TRANSFORMED_IMAGES = True

OUT_DIR = 'outputs'