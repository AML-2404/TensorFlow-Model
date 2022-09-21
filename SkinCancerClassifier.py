import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

img_heitgh,img_width = 32,32
batch_size = 20

train_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size= (img_heitgh,img_width),
    batch_size = 20
)

val_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/validation",
    image_size= (img_heitgh,img_width),
    batch_size = 20
)

test_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size= (img_heitgh,img_width),
    batch_size = 20
)

class_names = ['BCC','Melanoma']

# plt.figure(figsize=(10,10))
#
# for image,labels in train_data.take(1):
#   for i in range(9):
#     ax = plt.subplot(3,3,i+1)
#     plt.imshow(image[i].numpy().astype('uint8'))
#     plt.title(class_names[labels[i]])
#
# plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32,3,activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32,3,activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32,3,activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dense(3)
    ]
)

model.compile(
    optimizer = 'adam',
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['accuracy']
    )

model.fit(
    train_data,
    validation_data = val_data,
    epochs = 20
)
#
plt.figure(figsize=(10,10))

pred = []
real = []

for image,labels in test_data.take(1):
  classification = model(image)
  # print(classification)
  for i in range(9):
    ax = plt.subplot(3,3,i+1)
    plt.imshow(image[i].numpy().astype('uint8'))
    index = np.argmax(classification[i])
    plt.title("Pred: "+ class_names[index] +" | Real: "+class_names[labels[i]])
    pred.append(class_names[index])
    real.append(class_names[labels[i]])

# for image,labels in test_data.take(1):
#   classification = model(image)
#   print(classification)
#   # for i in range(9):
#   #   ax = plt.subplot(3,3,i+1)
#   #   plt.imshow(image[i].numpy().astype('uint8'))
#   #   index = np.argmax(classification[i])
#   #   plt.title("Pred: "+ class_names[index] +" | Real: "+class_names[labels[i]])



# test_results = {}
#
# test_results['model'] = model.evaluate(
#     test_data, class_names, verbose=0)
#
# print(f" Accuracy: {test_results}")

# plt.show()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# y_pred_2 = model.predict(test_data)
# y_pred_2
# print(y_pred_2)

conconfusion = metrics.confusion_matrix(real, pred)
print(f"Confusion matrix:\n{conconfusion}")

# print(pred)
# print(real)