import json
from keras.preprocessing.image import ImageDataGenerator

original_img_width = 640
original_img_height = 480


class Preprocessor(object):
    def __init__(self, target_width=None, target_height=None, width_shift_range=0.15, height_shift_range=0.15,
                 rotation_range=30, shear_range=0.1, zoom_range=0.05, fill_mode='nearest'):

        self.params = {
            'th': target_height,
            'tw': target_width,
            'wsr': width_shift_range,
            'hsr': height_shift_range,
            'rr': rotation_range,
            'sr': shear_range,
            'zr': zoom_range,
            'fm': fill_mode
        }

        self.target_size = (target_width, target_height)
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            rotation_range=rotation_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            fill_mode=fill_mode
        )

        self.test_datagen = ImageDataGenerator(rescale=1./255)

    def get_train_generator(self, directory, batch_size):
        return self.train_datagen.flow_from_directory(
            directory,
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )

    def get_test_generator(self, directory, batch_size):
        return self.test_datagen.flow_from_directory(
            directory,
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

    def __str__(self):
        return 'Preprocessor(%s)' % json.dumps(self.params)

    def path(self):
        return (
            json.dumps(self.params)
                .replace(' ', '')
                .replace('"', '')
                .replace('{', '')
                .replace('}', '')
        )




def make_submission(model, path, class_indices):
    test_generator = get_test_generator()
    filenames = test_generator.filenames
    n_test = len(filenames)
    predictions = model.predict_generator(test_generator)

    import pandas as pd

    data_for_frame = {
        'filenames': filenames
    }

    for c, i in class_indices.items():
        i = int(i)
        data_for_frame[c] = predictions[:, i]

    pdf = pd.DataFrame(data_for_frame)
    pdf['img'] = pdf.filenames.map(lambda x: x.split('/')[1])
    pdf = pdf[['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']]
    pdf.to_csv(path, header=True, index=False)
