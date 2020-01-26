import matplotlib.pyplot as plt
import random
import numpy as np
import time
from tkinter import*
from PIL import ImageGrab
import csv
import pandas as pd
import re
import copy
class Board:

    def __init__(self, size, speed, shapesize):
        self.size = size
        posx= int(size/20)*10
        self.snake = [(int(size/20)*10, int(size/20)*10)]
        self.speed = speed
        self.apple = (abs((random.randint(0, (self.size / 10)) * 10 - shapesize)),
                     abs((random.randint(0, (self.size / 10)) * 10 - shapesize)))
        self.state = True
        while(self.apple == self.snake):
            (abs((random.randint(0, (self.size / 10)) * 10 - shapesize)),
             abs((random.randint(0, (self.size / 10)) * 10 - shapesize)))

    def toImage(self, canvas, shapesize):

        canvas.delete("all")
        for pos in self.snake:
            x = pos[0]
            y = pos[1]
            canvas.create_rectangle(x, y, x+shapesize, y+shapesize ,fill="white")
        canvas.create_rectangle(self.apple[0], self.apple[1],
                    self.apple[0] + shapesize, self.apple[1] + shapesize, fill="red")
        time.sleep(0.01)

class Snake:
    def __init__(self, tk, canvas, size, speed):
        self.shapesize = 10
        self.board = Board(size, speed, self.shapesize)
        self.size = size
        self.speed = speed
        self.tk = tk
        self.canvas = canvas
        self.score = 0
        # self.direction = self.get_direction()
        self.direction = "left"

    def plot_board(self, generetion , id, itr, frame ):
        self.board.toImage(self.canvas , self.shapesize)
        self.tk.update()
        self.tk.title(f"Snake gen: {generetion} child: {id} score: {self.score} movment left: {itr}")


    def clear(self):
        self.board = Board(self.size, self.speed)

        self.score = 0
        self.direction = "left"

    def check_State(self):
        if self.board.snake[0][0] >= self.board.size:
            return False
        if  self.board.snake[0][1] >= self.board.size:
            return False
        if self.board.snake[0][0] < 0:
            return False
        if  self.board.snake[0][1] < 0:
            return False
        if self.board.snake[0]  in self.board.snake[1:]:
            return False
        return True

    def move(self, dir):
        self.direction = dir
        before = self.board.snake[len(self.board.snake) - 1]


        if dir == "left":
            temp1 = self.board.snake[0]
            self.board.snake[0] = (self.board.snake[0][0], self.board.snake[0][1] - 10)
            for i in range(1, len(self.board.snake)):
                temp2 = self.board.snake[i]
                self.board.snake[i] = temp1
                temp1 = temp2
            if self.board.snake[0][0] == self.board.apple[0] and self.board.snake[0][1] == self.board.apple[1]:
                self.board.snake.append(before)
                self.board.apple = (abs((random.randint(0, (self.size / 10)) * 10 - self.shapesize)),
                     abs((random.randint(0, (self.size / 10)) * 10 - self.shapesize)))
                while self.board.apple in self.board.snake:
                    self.board.apple = (abs((random.randint(0, (self.size / 10)) * 10 - self.shapesize)),
                                        abs((random.randint(0, (self.size / 10)) * 10 - self.shapesize)))
                self.score +=1

        if dir == "right":

            temp1 = self.board.snake[0]
            self.board.snake[0] = (self.board.snake[0][0], self.board.snake[0][1] + 10)
            for i in range(1, len(self.board.snake)):
                temp2 = self.board.snake[i]
                self.board.snake[i] = temp1
                temp1 = temp2
            if self.board.snake[0][0] == self.board.apple[0] and self.board.snake[0][1] == self.board.apple[1]:
                self.board.snake.append(before)
                self.board.apple = (abs((random.randint(0, (self.size / 10)) * 10 - self.shapesize)),
                                    abs((random.randint(0, (self.size / 10)) * 10 - self.shapesize)))
                while self.board.apple in self.board.snake:
                    self.board.apple = (abs((random.randint(0, (self.size / 10)) * 10 - self.shapesize)),
                                        abs((random.randint(0, (self.size / 10)) * 10 - self.shapesize)))
                self.score += 1

        if dir == "up":
            temp1 = self.board.snake[0]
            self.board.snake[0] = (self.board.snake[0][0] - 10, self.board.snake[0][1])
            for i in range(1, len(self.board.snake)):
                temp2 = self.board.snake[i]
                self.board.snake[i] = temp1
                temp1 = temp2
            if self.board.snake[0][0] == self.board.apple[0] and self.board.snake[0][1] == self.board.apple[1]:
                self.board.snake.append(before)
                self.board.apple = (abs((random.randint(0, (self.size / 10)) * 10 - self.shapesize)),
                                    abs((random.randint(0, (self.size / 10)) * 10 - self.shapesize)))
                while self.board.apple in self.board.snake:
                    self.board.apple = (abs((random.randint(0, (self.size / 10)) * 10 - self.shapesize)),
                                        abs((random.randint(0, (self.size / 10)) * 10 - self.shapesize)))
                self.score += 1

        if dir == "down":
            temp1 = self.board.snake[0]
            self.board.snake[0] = (self.board.snake[0][0] + 10, self.board.snake[0][1])
            for i in range(1, len(self.board.snake)):
                temp2 = self.board.snake[i]
                self.board.snake[i] = temp1
                temp1 = temp2
            if self.board.snake[0][0] == self.board.apple[0] and self.board.snake[0][1] == self.board.apple[1]:
                self.board.snake.append(before)
                self.board.apple = (abs((random.randint(0, (self.size / 10)) * 10 - self.shapesize)),
                                    abs((random.randint(0, (self.size / 10)) * 10 - self.shapesize)))
                while self.board.apple in self.board.snake:
                    self.board.apple = (abs((random.randint(0, (self.size / 10)) * 10 - self.shapesize)),
                                        abs((random.randint(0, (self.size / 10)) * 10 - self.shapesize)))
                self.score += 1

    def move_snake(self, dir ):

            new_direction = dir
            if new_direction == "left" and self.direction !="right":
              self.move("left")

            elif  new_direction == "right" and self.direction !="left":
                self.move("right")

            elif  new_direction == "up"  and self.direction !="down":
                self.move("up")

            elif  new_direction == "down" and self.direction !="up":
                self.move("down")

            else: self.move(self.direction)

            if self.check_State() == True:
                return True
            return False

    def game_play(self):
        # self.plot_board()
        while(self.move_snake()):
            # print(self.create_x())
            self.plot_board()

    def create_x(self):
        X = []
        size = self.size
        x_head, y_head = self.board.snake[0][0], self.board.snake[0][1]
        x_apple, y_apple = self.board.apple[0], self.board.apple[1]
        snake_y = [self.board.snake[i][1] for i in range(1, len(self.board.snake))]
        snake_x = [self.board.snake[i][0] for i in range(1, len(self.board.snake))]
        # up
        if x_head <= 0:
            X.append(1e6)
        else:
            X.append(x_head)  # wall
        if y_head == y_apple and x_head < x_apple:
            X.append((x_apple - x_head))  # apple
        else:
            X.append(1e6)
        if (x_head - 10, y_head) in self.board.snake[1:]:
            X.append(1e6)
        elif y_head in snake_y:
            if x_head > snake_x[snake_y.index(y_head)]:
                X.append((size - x_head + snake_x[snake_y.index(y_head)]))
            else:
                X.append(x_head)
        else:
            X.append(x_head)

        # down
        if x_head >= size - self.shapesize or (x_head + 10, y_head) in self.board.snake[1:]:
            X.append(1e6)
        else:
            X.append((size - x_head))  # wall
        if y_head == y_apple and x_head > x_apple:  # apple
            X.append((x_head - x_apple))
        else:
            X.append(1e06)
        if (x_head + 10, y_head) in self.board.snake[1:]:
            X.append(1e6)
        elif y_head in snake_y:
            if x_head < snake_x[snake_y.index(y_head)]:
                X.append((size - snake_x[snake_y.index(y_head)] + x_head))
            else:
                X.append((size - x_head))
        else:
            X.append((size - x_head))

        # right
        if y_head >= size - self.shapesize or (x_head, y_head + 10) in self.board.snake[1:]:
            X.append(1e6)
        else:
            X.append((size - y_head))  # wall
        if x_head == x_apple and y_head < y_apple:  # apple
            X.append((y_apple - y_head))
        else:
            X.append(1e6)

        if (x_head, y_head + 10) in self.board.snake[1:]:
            X.append(1e6)
        elif x_head in snake_x:
            if y_head < snake_y[snake_x.index(x_head)]:
                X.append((size - snake_y[snake_x.index(x_head)] + y_head))
            else:
                X.append((size - y_head))
        else:
            X.append((size - y_head))

        # left
        if y_head <= 0 or (x_head, y_head - 10) in self.board.snake[1:]:
            X.append(1e6)
        else:
            X.append(y_head)  # wall
        if x_head == x_apple and y_head > y_apple:  # apple
            X.append((y_head - y_apple))
        else:
            X.append(1e6)
        if (x_head, y_head - 10) in self.board.snake[1:]:
            X.append(1e6)
        elif x_head in snake_x:
            if y_head > snake_y[snake_x.index(x_head)]:
                X.append((size - snake_y[snake_x.index(x_head)] + y_head))
            else:
                X.append(y_head)
        else:
            X.append(y_head)
        return X


class Learner:

    def __init__(self, size):
        self.snake = None
        self.size = size
        self.tk = Tk()
        self.canvas = Canvas(self.tk, width=size,bg="black", height=size)
        self.canvas.pack()

    def on_closing(self):
        self.tk.destroy()

    def eval(self, weights, generetion, id):
        weights1 = weights[0]
        weight2 = weights[1]
        self.snake = Snake(self.tk, self.canvas, self.size, self.size)
        x = self.snake.create_x()

        dir = NeuralNetwork.feedforward(x, weights1, weight2)
        cnt = 0
        itr = 500
        prev_s_size = len(self.snake.board.snake)
        while (self.snake.move_snake(dir) and itr > 0 ):
            start = time.perf_counter()
            s_size = len(self.snake.board.snake)
            if s_size > prev_s_size:
                itr += 300
            if self.snake.score >= 350:
                self.snake.plot_board(generetion , id, itr, cnt)
            x = self.snake.create_x()
            dir = NeuralNetwork.feedforward(x, weights1, weight2)
            itr -=1
            if itr > 750:  itr = 750
            cnt += 1
            prev_s_size = s_size
        # print(f"{time.perf_counter() - start} sconds")
        if  self.snake.score < 1:
            res = 0.01
        else:
            res = self.snake.score
        # print(res)
        return res

    def sexual_selection(pop_size, vals):
        vals = vals[:int(pop_size/10)]
        l = int(pop_size/10)
        couples = []
        grid = np.linspace(0, l - 1, l)

        for _ in range(pop_size):
            male = np.random.choice(grid, size=1, p= vals/np.sum(vals))
            female = np.random.choice(grid, size=1, p= vals/np.sum(vals))
            couples.append([int(male), int(female)])

        return couples

    def crossover(x, y, childs):
        childs.append((x+y)/2)
    def crossover2(self, x, y, alpha, beta, childs):
        childs.append((x + y)/2)
        childs.append(x - y)
        childs.append(y - x)


    def mutate(childs, p, pop_size, start):
        for i in range(start, pop_size):
            if np.random.uniform(0, 1) < p:
                childs[i] = childs[i] + np.random.normal(0, 1, np.shape(childs[i]))



    def merge_pop(parents, childs, parents_vals, childs_vals):
        parents_cnt, childs_cnt = 0, 0

        for i in range(len(parents)):
            if parents_vals[parents_cnt] > childs_vals[childs_cnt]:
                parents_cnt += 1
            else:
                childs_cnt += 1

        new_pop = parents[0:parents_cnt] + childs[0:childs_cnt]
        new_vals = parents_vals[0:parents_cnt] + childs_vals[0:childs_cnt]
        indicates = np.array(new_vals).argsort()
        new_pop = [new_pop[indicate] for indicate in indicates[::-1]]
        new_vals = sorted(new_vals)[::-1]
        return new_pop, new_vals


    def GA2(self, pop_size, max_evals, input_size, layer1_size, out_size, sexual_selection=sexual_selection, eval=eval,
           cross_over=crossover, mutate=mutate, merge_pop=merge_pop):

        # build graph
        start = time.perf_counter()
        iter_cnt =  0
        history = []
        # create randon population
        pop = []
        pop_vals = []
        childs = []
        for _ in range(pop_size):
            pop.append(np.array((np.random.uniform(-10, 10,(((input_size ) * layer1_size) + ((layer1_size)*out_size), 1)))))
        pop_wights = self.vect2matGA(pop, input_size, layer1_size, out_size, len(pop))
        parents_vals = [self.eval(pop_wights[i], iter_cnt, i) for i in range(len(pop))]
        pop = [pop[pos] for pos in np.argsort(parents_vals)[::-1]]
        parents_vals = sorted(parents_vals)[::-1]
        # circle of life
        while (iter_cnt < max_evals):
            print(f"gen {iter_cnt} has passed f_max: {parents_vals[0]}")
            # print(f"sd1: {pop[5][3]}     sd2: {pop[5][4]}   sd3: {pop[5][5]}")
            iter_cnt += 1
            childs.clear()
            couples = sexual_selection(pop_size , parents_vals)
            [cross_over(pop[couples[i][0]], pop[couples[i][1]], childs) for i in range(int(pop_size))]
            mutate(childs, 0.1 , len(childs), 0 )
            childs_wights = self.vect2matGA(childs, input_size, layer1_size, out_size, len(childs))
            childs_vals = [self.eval(childs_wights[i],iter_cnt, i) for i in range(pop_size)]
            childs = [childs[pos] for pos in np.argsort(childs_vals)[::-1]]
            childs_vals = sorted(childs_vals)[::-1]
            pop, parents_vals = merge_pop(pop, childs , parents_vals, childs_vals)
            history.append(parents_vals[0])
            print(childs_vals[:10])
            # time.sleep(2)
        print(f"GA took {(time.perf_counter() - start)/60} minutes for {max_evals} iteretion")

        self.snake.tk.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.snake.tk.mainloop()
        return history, parents_vals[0], pop[0]

    def mutate_es(self, pop, lamda, lamda_vect, pop_size):
        childs = []
        Zs = []
        for i in range(pop_size):
            Z = np.random.normal(0, 1, np.shape(pop))
            childs.append(pop[0] + (lamda * lamda_vect * Z)[0])
            Zs.append(Z)
        return np.array(childs) ,Zs

    def ES(self, pop_size, max_evals, input_size, layer1_size, out_size,path = "best.csv", eval=eval,
           mutate=mutate_es):
        start = time.perf_counter()
        iter_cnt = 0
        history = []
        childs = []
        pop = np.array([(np.random.uniform(-10, 10,(((input_size) * layer1_size) + ((layer1_size)*out_size), 1)))])
        # df = pd.read_csv('best.csv', sep=',')
        # x = df.values[0,4]
        # x = re.findall(r'\d+.\d+', x)
        # pop = np.empty([1,len(x),1])
        # x = np.array([x])
        # x = x.transpose()
        # pop[0] = x
        n = len(pop[0])
        whights = self.vect2mat(pop, input_size, layer1_size, out_size, 1)
        pop_val = self.eval(whights[0], iter_cnt, 0)
        Z = np.zeros(np.shape(pop))
        beta = np.sqrt(1 / n)
        beta_scale = 1/n
        lamda_g = 10
        lamda_g_scale = np.ones(np.shape(pop[0]))
        c = np.sqrt(1/n)
        # circle of life
        while (iter_cnt < max_evals):
            print(f"gen {iter_cnt} has passed f_max: {pop_val} lamda_g: {lamda_g}")
            # print(f"sd1: {pop[5][3]}     sd2: {pop[5][4]}   sd3: {pop[5][5]}")
            iter_cnt += 1
            Zs = []
            lamda_g = lamda_g*np.exp((np.linalg.norm(Z)/np.sqrt(n)*np.sqrt((c/(2-c)))) -1 + 1/(5*n))**(beta)
            lamda_g_scale = lamda_g_scale*(np.abs(Z) / np.sqrt(c/(2-c)) + 0.35)**beta_scale
            childs, Zs = self.mutate_es(pop,lamda_g, lamda_g_scale, pop_size)
            childs_wights = self.vect2mat(childs, input_size, layer1_size, out_size, len(childs))
            childs_vals = [self.eval(childs_wights[i],iter_cnt, i) for i in range(pop_size)]
            index = np.argmax(childs_vals)
            # if childs_vals[index] > pop_val:
            pop_val = childs_vals[index]
            pop[0] = childs[index]
            Z = (1-c)*Z +c*Zs[index]
        return pop[0], pop_val

    def vect2matGA(self, pop, input_size, layer1_size, out_size, pop_size):
        whights = []
        for i in range(pop_size):
            whights.append([list(np.reshape(pop[i][:(input_size ) * layer1_size], (input_size , layer1_size))),
                   list(np.reshape(pop[i][(input_size) * layer1_size:], ((layer1_size  ), out_size)))])
        return whights

    def vect2mat(self, pop, input_size, layer1_size, out_size, pop_size):
        whights = []
        for i in range(pop_size):
            temp = np.array(pop[i])
            whights.append([list(np.reshape(temp[:(input_size) * layer1_size], (input_size, layer1_size))),
                   list(np.reshape(temp[(input_size) * layer1_size:], ((layer1_size), out_size)))])
        return whights


    def dr2unpack(self, n, Z, beta, beta_scale, lamda_g, lamda_g_scale, c, input_size, layer1_size, out_size):
        return n, Z, beta, beta_scale, lamda_g, lamda_g_scale, c, input_size, layer1_size, out_size
    def dr2circle(self, island, island_size, apocs, dr2_parameters, island_number):
        n, Z, beta, beta_scale, lamda_g, lamda_g_scale, c, input_size, layer1_size, out_size= self.dr2unpack(*dr2_parameters)
        # print(f"            dr2 num : {island_number} ", island[:2])

        childs = []
        itr_cnt = 0
        whights = self.vect2mat(np.array([island]), input_size, layer1_size, out_size, 1)
        island_val = self.eval(whights[0], itr_cnt, 0)
        first_val = copy.deepcopy(island_val)
        best = island
        best_val = island_val
        while (itr_cnt < apocs):
            itr_cnt += 1
            Zs = []
            lamda_g = lamda_g * np.exp((np.linalg.norm(Z) / np.sqrt(n) * np.sqrt((c / (2 - c)))) - 1 + 1 / (5 * n)) ** (
                beta)
            lamda_g_scale = lamda_g_scale * (np.abs(Z) / np.sqrt(c / (2 - c)) + 0.35) ** beta_scale
            childs, Zs = self.mutate_es(np.array([island]), lamda_g, lamda_g_scale, island_size)
            childs_wights = self.vect2mat(childs, input_size, layer1_size, out_size, len(childs))
            childs_vals = [self.eval(childs_wights[i], itr_cnt, i) for i in range(island_size)]
            index = np.argmax(childs_vals)
            if childs_vals[index] > best_val:
                best = copy.deepcopy(childs[index])
                best_val = childs_vals[index]
                Z_best = Zs[index]
            island_val = childs_vals[index]
            island = childs[index]
            Z = (1 - c) * Z + c * Zs[index]
        print(f"island number: {island_number} has finished with best score of: {best_val}, first val: {first_val}")
        best_whight = self.vect2mat([best], input_size, layer1_size, out_size, 1)
        # for i in range(5):
        #     print("eval best: ", self.eval(best_whight[0], itr_cnt, 0) )
        #     print("eval topdrs: ", self.eval(childs_wights[index], itr_cnt, 0))

        return [best, best_val, Z]
    def island_optimaizer(self,apocs ,amount_of_islands, island_size,max_evals, input_size, layer1_size, out_size, sexual_selection=sexual_selection, eval=eval,
           cross_over=crossover, mutate=mutate, merge_pop=merge_pop):
        iter_cnt = 0
        # create randon population
        islands = []
        childs = []
        # initialize islands
        for i in range(amount_of_islands):
            islands.append(
                np.array((np.random.normal(0, 1, (((input_size) * layer1_size) + ((layer1_size) * out_size), 1)))))
            # print(f"island number: {i} ",islands[i][:2])

        # initialize dr2 parameters
        n = len(islands[0])
        Z = np.zeros(np.shape(islands))
        beta = np.sqrt(1 / n)
        beta_scale = 1/n
        lamda_g = 1
        lamda_g_scale = np.ones(np.shape(islands[0]))
        c = np.sqrt(1/n)
        dr2_parameters = [n, Z, beta, beta_scale, lamda_g, lamda_g_scale ,c, input_size, layer1_size, out_size]
        best_val = 0
        # circle of life
        while (iter_cnt < max_evals):
            # create new islands and evaluation
            islands = [self.dr2circle(islands[i], island_size, apocs, dr2_parameters, i) for i in range(amount_of_islands)]
            islands_val = [islands[i][1] for i in range(amount_of_islands)]
            islands_Z =  [islands[i][2] for i in range(amount_of_islands)]
            islands = [islands[i][0] for i in range(amount_of_islands)]

            islands = [islands[pos] for pos in np.argsort(islands_val)[::-1]]
            islands_val = sorted(islands_val)[::-1]
            print(f"islands untie number {iter_cnt} has passed wtih f_max: {islands_val[0]}")
            iter_cnt += 1
            childs.clear()
            couples = sexual_selection(amount_of_islands, islands_val)
            [self.crossover2(islands[couples[i][0]], islands[couples[i][1]], 0.5, 0.5, childs) for i in range(int(amount_of_islands))]
            childs_wights = self.vect2mat(childs, input_size, layer1_size, out_size, len(childs))
            childs_vals = [self.eval(childs_wights[i], iter_cnt, i) for i in range(amount_of_islands)]
            childs = [childs[pos] for pos in np.argsort(childs_vals)[::-1]]
            childs_vals = sorted(childs_vals)[::-1]
            islands, islands_val = merge_pop(islands, childs, islands_val, childs_vals)
            print(islands_val[:10], f"amount of islands: {len(islands)}")
            print(childs_vals[:10])
            if islands_val[0] > best_val:
                best = islands[0]
                best_val = islands_val[0]
            if     islands_val[0] > 80:
                bla = 1
            dr2_parameters = [n, islands_Z[0], beta, beta_scale, lamda_g, lamda_g_scale, c, input_size, layer1_size, out_size]
        return best, best_val


class NeuralNetwork:

    def __init__(self, weights1, weights2):
        self.weights1 = weights1
        self.weights2 = weights2

    def feedforward(x, weights1, weights2, ):
        x = x/np.max(x)
        layer1 = NeuralNetwork.sigmoid(np.dot(x, weights1))
        output =  NeuralNetwork.sigmoid(np.dot(layer1, weights2))
        # print(f"layer1 : {self.layer1}")
        # print(f"output layer : {self.output}")
        num = np.argmax(output)
        if num % 4 == 0:
            dir = "left"
        if num % 4 == 1:
            dir = "right"
        if num % 4 == 2:
            dir = "down"
        if num % 4 == 3:
            dir = "up"
        # print(f"net output: {self.output}")
        return dir


    def sigmoid(x):
        res = 1 / (1 + np.exp(-x))
        return res
    def get_weights(self):
        return (self.weights1 , self.weights2)
