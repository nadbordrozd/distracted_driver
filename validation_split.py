import pandas as pd
import os
import shutil
import glob

VAL_SIZE = 0.2

RAW_TRAIN_DIR = 'data/train/'
TEST_DIR = 'data/test/'
TRAIN_DIR = 'data/train_split'
VALIDATION_DIR = 'data/validation_split'
TINY_TRAIN_DIR = 'data/tiny_train_split'
TINY_VALIDATION_DIR = 'data/tiny_validation_split'
TINY_TEST_DIR = 'data/tiny_test'


def make_train_dev_directories():
    os.mkdir(TRAIN_DIR)
    os.mkdir(VALIDATION_DIR)

    os.mkdir(TINY_TRAIN_DIR)
    os.mkdir(TINY_VALIDATION_DIR)

    for i in range(10):
        os.mkdir(os.path.join(TRAIN_DIR, 'c' + str(i)))
        os.mkdir(os.path.join(VALIDATION_DIR, 'c' + str(i)))
        os.mkdir(os.path.join(TINY_TRAIN_DIR, 'c' + str(i)))
        os.mkdir(os.path.join(TINY_VALIDATION_DIR, 'c' + str(i)))


def copy_train_dev_data():
    df = pd.read_csv('data/driver_imgs_list.csv')
    all_drivers = sorted(list(set(df.subject)))
    val_size = int(VAL_SIZE * len(all_drivers))
    validation_drivers = all_drivers[:val_size]
    print val_size, validation_drivers
    tiny_train_driver, tiny_validation_driver = all_drivers[:2]

    for _, subject, classname, filename in df.itertuples():
        origin_path = os.path.join(RAW_TRAIN_DIR, classname, filename)
        if subject in validation_drivers:
            shutil.copy(
                origin_path,
                os.path.join(VALIDATION_DIR, classname, filename))
        else:
            shutil.copy(
                origin_path,
                os.path.join(TRAIN_DIR, classname, filename))

        if subject == tiny_train_driver:
            shutil.copy(
                origin_path,
                os.path.join(TINY_TRAIN_DIR, classname, filename)
            )
        elif subject == tiny_validation_driver:
            shutil.copy(
                origin_path,
                os.path.join(TINY_VALIDATION_DIR, classname, filename)
            )


def fix_test_directory():
    # due to keras flow_from_directory semantics, we have to put
    # all the test images in a subdirectory like this
    # TEST_DIR/all_imgs/img*.jpg
    # so that it mirrors the setup of the train directory
    # TRAIN_DIR/c0/img*.jpg
    # TRAIN_DIR/c1/img*.jpg
    # and so on
    shutil.move(TEST_DIR, 'all_imgs')
    os.mkdir(TEST_DIR)
    shutil.move('all_imgs', TEST_DIR)
    os.mkdir(TINY_TEST_DIR)
    os.mkdir(os.path.join(TINY_TEST_DIR, 'all_imgs'))

    test_imgs = glob.glob(os.path.join(TEST_DIR, 'all_imgs', '*'))
    for img_path in test_imgs[:100]:
        shutil.copy(
            img_path,
            os.path.join(TINY_TEST_DIR, 'all_imgs', os.path.basename(img_path))
        )

if __name__ == '__main__':
    if os.path.exists(TRAIN_DIR):
        print 'nothing to do here'
    else:
        make_train_dev_directories()
        copy_train_dev_data()
        fix_test_directory()
