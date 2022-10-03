from PIL import Image
import numpy as np
import sqlite3 as sl
import pickle

con = sl.connect('test-data.db')
with con:
    con.execute("""
        CREATE TABLE DATA (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            image TEXT,
            label INTEGER
        );
    """)

labels = open("t10k-labels.idx1-ubyte", "rb")
images = open("t10k-images.idx3-ubyte", "rb")

labelMagicNum = labels.read(4)
imageMagicNum = images.read(4)

print(int.from_bytes(labelMagicNum, byteorder="big"))
print(int.from_bytes(imageMagicNum, byteorder="big"))

numOfLabels = int.from_bytes(labels.read(4), byteorder="big")
numOfImages = int.from_bytes(images.read(4), byteorder="big")
numOfImageRows = int.from_bytes(images.read(4), byteorder="big")
numOfImageCols = int.from_bytes(images.read(4), byteorder="big")

print(numOfLabels)
print(numOfImages)
print(numOfImageRows)
print(numOfImageCols)

data = []

for i in range(numOfImages):
    label = int.from_bytes(labels.read(1), byteorder="big")
    image = []

    for j in range(numOfImageRows):
        image.append([])
        for k in range(numOfImageCols):
            pixelVal = int.from_bytes(images.read(1), byteorder="big")
            image[j].append(pixelVal)

    sql = 'INSERT INTO DATA (image, label) values(?, ?)'
    sql_data = [
        (pickle.dumps((np.array(image, dtype=np.uint8)).flatten()), label)
    ]
    with con:
        con.executemany(sql, sql_data)
    # data.append((image, label))

# array = np.array(data[0][0], dtype=np.uint8)
#
# new_image = Image.fromarray(array)
# new_image.save(str(data[0][1]) + '.png')
#
# print(array.flatten())
#
# array = np.array(data[1][0], dtype=np.uint8)
#
# new_image = Image.fromarray(array)
# new_image.save(str(data[1][1]) + '.png')
#
# array = np.array(data[2][0], dtype=np.uint8)
#
# new_image = Image.fromarray(array)
# new_image.save(str(data[2][1]) + '.png')
