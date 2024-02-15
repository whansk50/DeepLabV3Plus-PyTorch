import warnings
warnings.filterwarnings("ignore")

#SPA classes
CLASSES = {
    0:'background',
    1:'aeroplane',
    2:'bicycle',
    3:'bird',
    4:'boat',
    5:'bottle',
    6:'bus',
    7:'car',
    8:'cat',
    9:'chair',
    10:'cow',
    11:'diningtable',
    12:'dog',
    13:'horse',
    14:'motorbike',
    15:'person',
    16:'pottedplant',
    17:'sheep',
    18:'sofa',
    19:'train',
    20:'tvmonitor'
}

batch=2
lr = 1e-5
n_classes = len(CLASSES) #must be 1 more than the number of classes in the dataset because of background(class 0)

# Set flag to train the model or not. If set to 'False', only prediction is performed (using an older model checkpoint)
TRAINING = True

#setting parameter
HEIGHT=WIDTH=256

# Set num of epochs
EPOCHS = 100

class_dict = CLASSES
# Get class names
class_names = list(class_dict.keys())
# Get class RGB values
class_rgb_values= list(class_dict.values())

val_term = 3