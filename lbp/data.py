import os
import random
import numpy as np
import cv2


def is_image_file(filename):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    return any(filename.lower().endswith(ext) for ext in valid_extensions)


def load_images_from_folder(folder_path, num_samples, label):

    images = []
    labels = []

    all_files = [f for f in os.listdir(folder_path) if is_image_file(f)]
    sampled_files = random.sample(all_files, min(num_samples, len(all_files)))

    for file in sampled_files:
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            labels.append(label)

    return images, labels


def prepare_dataset(base_folder, num_samples_per_class, num_live, num_spoof):
    X = []
    y = []

    people_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

    for person in people_folders:
        person_folder = os.path.join(base_folder, person)
        live_folder = os.path.join(person_folder, 'live')
        spoof_folder = os.path.join(person_folder, 'spoof')

        # Load live images
        live_images, live_labels = load_images_from_folder(live_folder, num_samples_per_class, label=1)
        X.extend(live_images[:num_live])
        y.extend(live_labels[:num_live])

        # Load spoof images
        spoof_images, spoof_labels = load_images_from_folder(spoof_folder, num_samples_per_class, label=-1)
        X.extend(spoof_images[:num_spoof])
        y.extend(spoof_labels[:num_spoof])

    return np.array(X), np.array(y)
