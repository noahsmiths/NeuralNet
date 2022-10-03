import sqlite3 as sl
import pickle
from PIL import Image

con = sl.connect('training-data.db')

res = con.execute("""SELECT * FROM DATA WHERE label = 1 and id > 7""")
tp = res.fetchone()

arr = pickle.loads(tp[1])

new_image = Image.fromarray(arr.reshape(28, 28))
new_image.save(str(tp[0]) + "_" + str(tp[2]) + '.png')