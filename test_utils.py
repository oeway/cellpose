import os
from utils import geojson_to_label, load_train_test_data

def test_geojson_to_label():
    geojson_to_label('./data/hpa_dataset_v2/train/3213_1239_D9_1/annotation.json')
    assert os.path.exists('./data/hpa_dataset_v2/train/3213_1239_D9_1/cell_masks.png')

def test_load_train_dataset():
    images, labels, image_names, test_images, test_labels, image_names_test = load_train_test_data('./data/hpa_dataset_v2/train', './data/hpa_dataset_v2/test', ['er.png', 'nuclei.png'], 'cell_masks.png')
    print(len(images), len(test_images), images[0].shape)
    assert len(images) == len(labels) == 20 and len(test_images) == len(test_labels)== 143
    assert images[0].shape == (512, 512, 2)

    images, labels, image_names, test_images, test_labels, image_names_test = load_train_test_data('./data/hpa_dataset_v2/train', './data/hpa_dataset_v2/test', ['er.png', 'nuclei.png'], 'cell_masks.png', 0.5)
    assert images[0].shape == (256, 256, 2)

if __name__ == '__main__':
    test_load_train_dataset()
