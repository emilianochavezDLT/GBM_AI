import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

#Reference https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python
#arr is now a numpy array of shape (60000, 28, 28), with each element an integer from 0 to 255
'''
# Plot the first 10 images
for i in range(3):
    plt.imshow(arr[i], cmap='gray')
    plt.show()
'''

# File paths
train_images_file = 'Dataset/train-images-idx3-ubyte'
train_labels_file = 'Dataset/train-labels-idx1-ubyte'
test_images_file = 'Dataset/t10k-images-idx3-ubyte'
test_labels_file = 'Dataset/t10k-labels-idx1-ubyte'

# Load the data
train_images = idx2numpy.convert_from_file(train_images_file)
train_labels = idx2numpy.convert_from_file(train_labels_file)
test_images = idx2numpy.convert_from_file(test_images_file)
test_labels = idx2numpy.convert_from_file(test_labels_file)


'''
# Display a few images and labels
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(train_images[i], cmap='gray')
    plt.title("Label: " + str(train_labels[i]))
    plt.axis('off')
plt.show()

#Count the number of images
print(arr.shape[0]) # 60000
'''

#Data Preprocessing

#Reshape the images, meaning flatten the 28x28 images into 1D arrays of length 28*28=784


#Get the unique labels
#The .unique() function returns the unique values in the array, and the .shape attribute returns the number of unique values.
unique_labels = np.unique(train_labels)
print(unique_labels) # [0 1 2 3 4 5 6 7 8 9]


# Flatten the images
train_images = train_images.reshape((train_images.shape[0], -1))
test_images = test_images.reshape((test_images.shape[0], -1))

'''
The flattening of images is done by taking the 28x28 images and 
reshaping them into 1D arrays of length 28*28=784.

28 columns and 28 rows are flattened into a single row of 784 elements.

This is useful because many machine learning algorithms require the input data to be 1D arrays.
Each pixel in the 28x28 images is a feature, and the machine learning algorithms will treat each pixel as a separate feature.

'''

#Normalize the images
#The pixel values in the images range from 0 to 255. 
#We can normalize the pixel values to be between 0 and 1 by dividing the pixel values by 255.
train_images = train_images / 255.0
test_images = test_images / 255.0



