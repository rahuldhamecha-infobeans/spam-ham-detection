import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from ib_aitool.admin.interview_analyzer.metrics import dice_loss, dice_coef, iou
from ib_tool import BASE_DIR

""" Global parameters """
H = 512
W = 512

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def remove_bg(image_folder_path):
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    # create_dir("remove_bg")

    model_path = os.path.join(BASE_DIR,'ib_aitool/admin/interview_analyzer','model.h5')

    """ Loading model: DeepLabV3+ """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model(model_path)

    # model.summary()
    image_folder_final_path = os.path.join(image_folder_path,'*')

    print(image_folder_final_path)
    """ Load the dataset """
    data_x = glob(image_folder_final_path)

    for path in tqdm(data_x, total=len(data_x)):

        print(path)
        """ Extracting name """
        name = path.split("/")[-1].split(".")[0]

        """ Read the image """
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        x = cv2.resize(image, (W, H))
        x = x / 255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        """ Prediction """
        y = model.predict(x)[0]
        y = cv2.resize(y, (w, h))
        y = np.expand_dims(y, axis=-1)
        y = y > 0.5

        photo_mask = y
        background_mask = np.abs(1 - y)

        # cv2.imwrite(f"remove_bg/{name}.png", photo_mask*255)
        # cv2.imwrite(f"remove_bg/{name}.png", background_mask*255)

        # cv2.imwrite(f"remove_bg/{name}.png", image * photo_mask)
        # cv2.imwrite(f"remove_bg/{name}.png", image * background_mask)

        masked_photo = image * photo_mask
        background_mask = np.concatenate([background_mask, background_mask, background_mask], axis=-1)
        background_mask = background_mask * [255, 255, 255]
        final_photo = masked_photo + background_mask
        cv2.imwrite(f"{image_folder_path}/{name}.jpg", final_photo)