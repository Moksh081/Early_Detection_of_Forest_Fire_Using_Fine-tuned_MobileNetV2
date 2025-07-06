from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Input
import matplotlib.pyplot as plt

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load dataset
train_data = train_datagen.flow_from_directory(
    '/kaggle/input/the-wildfire-dataset/the_wildfire_dataset_2n_version/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_data = val_test_datagen.flow_from_directory(
    '/kaggle/input/the-wildfire-dataset/the_wildfire_dataset_2n_version/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_data = val_test_datagen.flow_from_directory(
    '/kaggle/input/the-wildfire-dataset/the_wildfire_dataset_2n_version/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# SqueezeNet model
def SqueezeNet(input_shape=(224, 224, 3), num_classes=1000):
    inputs = Input(shape=input_shape)
    
    # Initial Conv Layer
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Fire Modules (Squeeze + Expand)
    x = Conv2D(16, (1, 1), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    
    # Output Layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

# Create Model
model = SqueezeNet(input_shape=(224, 224, 3), num_classes=train_data.num_classes)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train model
history = model.fit(
    train_data,
    epochs=10,
    validation_data=validation_data,
    batch_size=32,
    verbose=1
)

# Plot training results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Evaluate model
test_loss, test_acc = model.evaluate(test_data)
print(f'Test accuracy: {test_acc}')


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Input, DepthwiseConv2D, BatchNormalization, ReLU, Add
import matplotlib.pyplot as plt

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load dataset
train_data = train_datagen.flow_from_directory(
    '/kaggle/input/the-wildfire-dataset/the_wildfire_dataset_2n_version/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_data = val_test_datagen.flow_from_directory(
    '/kaggle/input/the-wildfire-dataset/the_wildfire_dataset_2n_version/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_data = val_test_datagen.flow_from_directory(
    '/kaggle/input/the-wildfire-dataset/the_wildfire_dataset_2n_version/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# ShuffleNet model
def ShuffleNet(input_shape=(224, 224, 3), num_classes=1000):
    inputs = Input(shape=input_shape)
    
    # Initial Conv Layer
    x = Conv2D(64, (3, 3), strides=2, activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    
    # ShuffleNet Block
    def shuffle_unit(x, out_channels):
        branch = x
        x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(out_channels, (1, 1), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x, branch])
        return x
    
    x = shuffle_unit(x, 128)
    x = shuffle_unit(x, 256)
    x = shuffle_unit(x, 512)
    x = GlobalAveragePooling2D()(x)
    
    # Output Layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

# Create Model
model = ShuffleNet(input_shape=(224, 224, 3), num_classes=train_data.num_classes)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train model
history = model.fit(
    train_data,
    epochs=10,
    validation_data=validation_data,
    batch_size=32,
    verbose=1
)

# Plot training results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Evaluate model
test_loss, test_acc = model.evaluate(test_data)
print(f'Test accuracy: {test_acc}')