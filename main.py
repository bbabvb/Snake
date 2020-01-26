import csv

import matplotlib.pyplot as plt
from functions1 import Snake, Learner
import numpy as np
import random
import time
from tkinter import*


def main():
    learner = Learner(400)
    input_size = 12
    layer_size = 24
    out_size = 4
    pop_size = 1500
    max_eval = 350
    apocs = 15
    amount_of_islands = 50
    island_size = 10
    # x, val = learner.ES(pop_size, max_eval, input_size, layer_size, out_size, 1)
    x, val = learner.island_optimaizer(apocs ,amount_of_islands, island_size , max_eval, input_size, layer_size, out_size)
    with open('best.csv', 'a', newline='\n') as file:
        writer = csv.writer(file)
        writer.writerow([val, input_size, layer_size, out_size, x])
    # tk = Tk()
    # canvas = Canvas(tk , width=500, height=400)
    # tk.title("Drawing")
    # canvas.pack()
    #
    # ball = canvas.create_oval(0, 0, 60, 60, fill="red")
    # xspeed = 1,
    # yspeed = 2,
    #
    # while True:
    #     canvas.move(ball, xspeed, yspeed)
    #     tk.update()
    #     time.sleep(0.01)
    # tk.mainloop()
if __name__ == "__main__":
    main()



