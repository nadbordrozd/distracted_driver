from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras import applications


class Model1(object):
    def __init__(self, img_width, img_height, layers_to_freeze, dense_units):
        self.layers_to_freeze = int(layers_to_freeze)
        self.dense_units = int(dense_units)
        self.model = None
        self.img_width = img_width
        self.img_height = img_height

        self.params = (img_width, img_height, self.layers_to_freeze, self.dense_units)

    def get_model(self):
        model = applications.VGG16(include_top=False, weights='imagenet',
                                   input_shape=(self.img_width, self.img_height, 3))

        # print 'I HAVE THIS MANY? LAYERS', len(model.layers)
        for layer in model.layers[:self.layers_to_freeze]:
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
        return 'm1_vgg16/%s_%s_%s_%s' % self.params

    def __str__(self):
        return 'Model1(%s, %s, %s, %s)' % self.params
