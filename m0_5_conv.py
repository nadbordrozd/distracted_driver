from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, BatchNormalization, MaxPool2D


class Model0_5(object):
    def __init__(self, img_width, img_height):
        self.model = None
        self.img_width = img_width
        self.img_height = img_height

    def get_model(self):
        self.model = Sequential([
            BatchNormalization(axis=3, input_shape=(self.img_width, self.img_height, 3)),
            Conv2D(32, 3, 3, activation='relu'),
            BatchNormalization(axis=3),
            MaxPool2D((3, 3)),
            Conv2D(64, 3, 3, activation='relu'),
            BatchNormalization(axis=3),
            MaxPool2D((3, 3)),
            Conv2D(128, 3, 3, activation='relu'),
            BatchNormalization(axis=3),
            MaxPool2D((3, 3)),
            Flatten(),
            Dense(200, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(200, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.model

    def path(self):
        return 'm0_5_conv/%s_%s' % (self.img_width, self.img_height)

    def __str__(self):
        return 'Model0_5(%s, %s)' % (self.img_width, self.img_height)
