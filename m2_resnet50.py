from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras import applications


class Model2(object):
    def __init__(self, img_width, img_height, dense_units):
        self.dense_units = int(dense_units)
        self.model = None
        self.img_width = img_width
        self.img_height = img_height

    def get_model(self):
        model = applications.ResNet50(include_top=False, weights='imagenet',
                                   input_shape=(self.img_width, self.img_height, 3))

        for layer in model.layers:
            layer.trainable = False

        x = model.output
        x = Flatten()(x)
        x = Dropout(0.4)(x)
        x = Dense(self.dense_units, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(self.dense_units, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.8)(x)
        predictions = Dense(10, activation='softmax')(x)

        model_final = Model(input=model.input, output=predictions)
        model_final.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['categorical_accuracy'])

        self.model = model_final
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model_final

    def path(self):
        return 'models/m2_resnet50/%s_%s_du_%s' % (self.img_width, self.img_height, self.dense_units)

    def __str__(self):
        return 'Model2(%s, %s, %s)' % (self.img_width, self.img_height, self.dense_units)
