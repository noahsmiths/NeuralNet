import sqlite3 as sl
import random


class TrainingData:
    def __init__(self):
        self.con = sl.connect('training-data.db')

    def get_image(self, index):
        res = self.con.execute("""SELECT * FROM DATA WHERE id = ?""", (index, ))
        return res.fetchone()

    def get_random_image(self):
        index = random.randint(1, 60000)
        return self.get_image(index)
