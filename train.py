import pandas as pd
import pickle
import argparse
import shutil
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

from m0_naive import Model0
from m1_vgg16 import Model1
from m2_resnet50 import Model2
from m0_5_conv import Model0_5

from preprocessing import Preprocessor
import validation_split as vs

model_dict = {
    'm0': Model0,
    'm05': Model0_5,
    'm1': Model1,
    'm2': Model2
}


def make_submission(model, data_generator, path):
    class_indices = {'c%s' % i: i for i in range(10)}
    filenames = data_generator.filenames
    predictions = model.predict_generator(data_generator)

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


def main(model_name, batch_size=64, model_params=None, preprocessing_params=None, test_run=False):
    model_wrapper = model_dict[model_name](*model_params)
    preprocessor = Preprocessor(**preprocessing_params)

    model_dir = os.path.join(
        'test_runs' if test_run else 'output',
        model_wrapper.path(),
        preprocessor.path()
    )
    success_path = os.path.join(model_dir, 'history')

    if not os.path.exists(success_path):

        best_model_path = os.path.join(model_dir, 'model-best.hdf5')
        shutil.rmtree(model_dir, ignore_errors=True)
        os.makedirs(model_dir)
        model = model_wrapper.get_model()

        callbacks = [
            ModelCheckpoint(
                best_model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
            EarlyStopping(patience=1 if test_run else 5, verbose=1)
        ]

        if test_run:
            print 'test run'
            train_generator = preprocessor.get_train_generator(vs.TINY_TRAIN_DIR, batch_size)
            val_generator = preprocessor.get_test_generator(vs.TINY_VALIDATION_DIR, batch_size)
            test_generator = preprocessor.get_test_generator(vs.TINY_TEST_DIR, batch_size)
            print 'class indices'
            print train_generator.class_indices
        else:
            print 'true run'
            train_generator = preprocessor.get_train_generator(vs.TRAIN_DIR, batch_size)
            val_generator = preprocessor.get_test_generator(vs.VALIDATION_DIR, batch_size)
            test_generator = preprocessor.get_test_generator(vs.TEST_DIR, batch_size)

        train_samples = train_generator.samples
        val_samples = val_generator.samples
        history = model.fit_generator(
            train_generator,
            train_samples // batch_size,
            validation_data=val_generator,
            validation_steps=val_samples // batch_size + 1,
            epochs=100,
            callbacks=callbacks
        ).history
        with open(success_path, 'wb') as f:
            pickle.dump(history, f)

        model = load_model(best_model_path)
        make_submission(model, test_generator, os.path.join(model_dir, 'submission.csv'))

    with open(success_path, 'rb') as f:
        history = pickle.load(f)

    print model_wrapper
    print preprocessor
    print 'best loss %.3f' % min(history['val_loss'])
    print 'best accuracy %.3f' % max(history['val_acc'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trains a model and creates a submission.'
        'Saves the best version of the model (according to validation loss).')
    parser.add_argument('model', help='which model to use. Currently available - m0, m05, m1, m2')
    parser.add_argument('target_width', type=int, help='width in pixels to which the image will be resized.')
    parser.add_argument('target_height', type=int, help='height in pixels to which the image will be resized.')
    parser.add_argument('batch_size', type=int, help='number of records in a batch. This is a common parameter of all models.')
    parser.add_argument('width_shift_range', type=float, help='range of horizontal shift in data augmentation (as a fraction).')
    parser.add_argument('height_shift_range', type=float, help='range of vertical shift in data augmentation (as a fraction).')
    parser.add_argument('rotation_range', type=int, help='range of rotation in data augmentation, in degrees.')
    parser.add_argument('shear_range', type=float, help='range of shear angle in data augmentation, in degrees.')
    parser.add_argument('zoom_range', type=float, help='range of zoom in data augmentation. If zoom_range = z, zoom will be in the range [1-z, 1+z]')
    parser.add_argument('fill_mode', help='points outside the boundaries are filled according to the mode')

    parser.add_argument('model_additional_params', nargs='*', help='parameters specific to the given model')
    parser.add_argument('--test', action='store_true', help='Test run on tiny data')

    args = parser.parse_args()
    model_params = [
        args.target_width,
        args.target_height
    ] + args.model_additional_params

    preprocessing_params = {
        'target_width': args.target_width,
        'target_height': args.target_height,
        'width_shift_range': args.width_shift_range,
        'height_shift_range': args.height_shift_range,
        'rotation_range': args.rotation_range,
        'shear_range': args.shear_range,
        'zoom_range': args.zoom_range,
        'fill_mode': args.fill_mode
    }

    main(args.model, args.batch_size, model_params, preprocessing_params, args.test)
