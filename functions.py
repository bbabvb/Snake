import matplotlib.pyplot as plt
import random
import numpy as np
import time
from tkinter import*

class Board:

    def __init__(self, size, speed):
        self.size = size
        self.snake = [(int(size/2) , int(size/2))]
        self.speed = speed
        self.apple = (random.randint(1,size-2) , random.randint(1,size-2))
        self.state = True
        while(self.apple == self.snake):
            self.apple = (random.randint(1, size-2), random.randint(1, size-2))

    def toImage(self):
        # board = [[255] * self.size] * self.size
        board = [[255 for i in range(self.size)] for j in range(self.size)]

        for pos in self.snake:
            x = pos[0]
            y = pos[1]
            board[x][y] = 0

        board[self.apple[0]][self.apple[1]] = 125
        board = np.array(board)
        return board[1:self.size-1 , 1:self.size -1]


class Snake:
    def __init__(self, size, speed ):
        self.board = Board(size, speed)
        self.size = size
        self.speed = speed
        self.pl = plt
        self.tk = Tk()
        self.canvas = Canvas(self.tk, size, size)
        self.canvas.pack()
        self.score = 0
        # self.direction = self.get_direction()
        self.direction = "left"

    def plot_board(self, generetion , id ):
        self.pl.imshow(self.board.toImage())
        self.pl.pause(0.05)
        self.pl.title(f"generetion: {generetion} child: {id} score: {self.score}")

    def clear(self):
        self.board = Board(self.size, self.speed)
        self.pl.close()
        self.pl = plt
        self.score = 0
        self.direction = "left"

    def check_State(self):
        if self.board.snake[0][0] == self.board.size - 1:
            return False
        if  self.board.snake[0][1] == self.board.size - 1:
            return False
        if self.board.snake[0][0] < 1:
            return False
        if  self.board.snake[0][1] < 1:
            return False
        if self.board.snake[0]  in self.board.snake[1:]:
            return False
        return True


    def get_direction(self):
        dir = ["left" , "right" , "up" , "down"]
        return (dir[random.randint(0,100)% 4 ])

    def move(self, dir):
        self.direction = dir
        before = self.board.snake[len(self.board.snake) - 1]
        if dir == "left":
            temp1 = self.board.snake[0]
            self.board.snake[0] = (self.board.snake[0][0], self.board.snake[0][1] - 1)
            for i in range(1, len(self.board.snake)):
                temp2 = self.board.snake[i]
                self.board.snake[i] = temp1
                temp1 = temp2
            if self.board.snake[0][0] == self.board.apple[0] and self.board.snake[0][1] == self.board.apple[1]:
                self.board.snake.append(before)
                self.board.apple = (random.randint(1, self.board.size - 2), random.randint(1, self.board.size - 2))
                while self.board.apple in self.board.snake:
                    self.board.apple = (random.randint(1, self.board.size - 2), random.randint(1, self.board.size - 2))
                self.score +=1

        if dir == "right":

            temp1 = self.board.snake[0]
            self.board.snake[0] = (self.board.snake[0][0], self.board.snake[0][1] + 1)
            for i in range(1, len(self.board.snake)):
                temp2 = self.board.snake[i]
                self.board.snake[i] = temp1
                temp1 = temp2
            if self.board.snake[0][0] == self.board.apple[0] and self.board.snake[0][1] == self.board.apple[1]:
                self.board.snake.append(before)
                self.board.apple = (random.randint(1, self.board.size - 2), random.randint(1, self.board.size - 2))
                while self.board.apple in self.board.snake:
                    self.board.apple = (random.randint(1, self.board.size - 2), random.randint(1, self.board.size - 2))
                self.score += 1

        if dir == "up":
            temp1 = self.board.snake[0]
            self.board.snake[0] = (self.board.snake[0][0] - 1, self.board.snake[0][1])
            for i in range(1, len(self.board.snake)):
                temp2 = self.board.snake[i]
                self.board.snake[i] = temp1
                temp1 = temp2
            if self.board.snake[0][0] == self.board.apple[0] and self.board.snake[0][1] == self.board.apple[1]:
                self.board.snake.append(before)
                self.board.apple = (random.randint(1, self.board.size - 2), random.randint(1, self.board.size - 2))
                while self.board.apple in self.board.snake:
                    self.board.apple = (random.randint(1, self.board.size - 2), random.randint(1, self.board.size - 2))
                self.score += 1

        if dir == "down":
            temp1 = self.board.snake[0]
            self.board.snake[0] = (self.board.snake[0][0] + 1, self.board.snake[0][1])
            for i in range(1, len(self.board.snake)):
                temp2 = self.board.snake[i]
                self.board.snake[i] = temp1
                temp1 = temp2
            if self.board.snake[0][0] == self.board.apple[0] and self.board.snake[0][1] == self.board.apple[1]:
                self.board.snake.append(before)
                self.board.apple = (random.randint(1, self.board.size - 2), random.randint(1, self.board.size - 2))
                while self.board.apple in self.board.snake:
                    self.board.apple = (random.randint(1, self.board.size - 2), random.randint(1, self.board.size - 2))
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
        self.pl.ion()
        # self.plot_board()
        while(self.move_snake()):
            # print(self.create_x())
            self.plot_board()
            self.pl.pause(10)
            self.pl.clf()


    def create_x(self):
        X = []
        size = self.size
        x_head,y_head = self.board.snake[0][0], self.board.snake[0][1]
        x_apple, y_apple = self.board.apple[0], self.board.apple[1]

        # up
        if x_head == 1:
            X.append(1e06)
        else: X.append( x_head -1) # wall
        if y_head == y_apple and x_head < x_apple:
            X.append(x_apple - x_head) # apple
        else:
            X.append(1e06)

        # down
        if x_head == size - 1:
            X.append(1e06)
        else:
            X.append(size - x_head -1) # wall
        if y_head == y_apple and x_head > x_apple: # apple
            X.append(x_head - x_apple)
        else:
            X.append(1e06)

        # right
        if y_head == size - 1  :
            X.append(1e06)
        else:
            X.append(size - y_head - 1) # wall
        if x_head == x_apple and y_head < y_apple: # apple
            X.append(y_apple - y_head)
        else:
            X.append(1e06)

        # left
        if y_head == 1 :
            X.append(1e06)
        else:
            X.append(y_head - 1)  # wall
        if x_head == x_apple and y_head > y_apple:  # apple
            X.append(y_head - y_apple)
        else:
            X.append(1e06)


        # # up right
        # if x_head == 1 or y_head == size -1 :
        #     X.append(99999)
        # else: X.append(x_head + size - y_head)  # wall
        # if x_head - y_head == x_apple - y_apple - \
        #         (x_apple - x_head) :  # apple
        #     X.append( abs(x_apple - y_apple) + abs(x_apple - x_head)*2)
        # else:
        #     X.append(1e06)
        #
        # # up left
        # if x_head == 1 or y_head == 1:
        #     X.append(1e06)
        # else: X.append(x_head + y_head)  # wall
        # if x_head - y_head == x_apple - y_apple :  # apple
        #     X.append(abs(x_head - x_apple)*2)
        # else:
        #     X.append(1e06)
        #
        # # down right
        # if x_head == size -1 or y_head== size-1:
        #     X.append(1e06)
        # else:    X.append(x_head + size - y_head)  # wall
        # if x_head - y_head == x_apple - y_apple - \
        #         (x_apple - x_head):  # apple
        #     X.append(abs(x_head - x_apple) - abs(y_head - y_apple))
        # else:
        #     X.append(1e06)
        #
        # # down left
        # if x_head == size-1 or y_head == 1:
        #     X.append(1e06)
        # else: X.append(x_head + size - y_head)  # wall
        # if x_head - y_head == y_apple - x_apple + \
        #         (x_apple - x_head):  # apple
        #     X.append(abs(y_apple - x_apple - (x_apple - x_head)))
        # else:
        #     X.append(1e06)
        #
        # # print(X)
        return X

class Learner:

    def __init__(self, snake, pop_size):
        self.snake = snake
        self.pop_size = pop_size
        self.imgs = []



    def eval(self, weights , generetion , id , layer_size , out_size):
        weights1 = weights[0]
        weight2 = weights[1]
        x = self.snake.create_x()
        nn = NeuralNetwork(len(x), layer_size, out_size, weights1, weight2)

        dir = nn.feedforward(x)
        self.snake.pl.ion()
        tk = Tk()
        canvas = Canvas(tk, width=500, height=500)
        tk.title(f"Snake gen{generetion}")
        canvas.pack()
        cnt = 0
        itr = 150
        prev_s_size = len(self.snake.board.snake)
        while (self.snake.move_snake(dir) and itr > 0 ):
            start = time.perf_counter()
            s_size = len(self.snake.board.snake)
            if s_size > prev_s_size:
                itr += 150
            if self.snake.score >= 30:
                self.snake.plot_board(generetion , id, window)
            x = self.snake.create_x()
            dir = nn.feedforward(x)
            itr -=1
            cnt += 1
            prev_s_size = s_size
        # print(f"{time.perf_counter() - start} sconds")
        if  self.snake.score < 1:
            res = 0.01
        else:
            res = self.snake.score
        self.snake.clear()
        # print(res)
        return res

    def sexual_selection(x, vals):
        l1 = len(x)
        x = x[:15]
        vals = vals[:15]
        l = len(x)
        couples = []
        grid = np.linspace(0, l - 1, l)

        for _ in range(l1):
            male = np.random.choice(grid, size=1, p= vals/np.sum(vals))
            female = np.random.choice(grid, size=1, p= vals/np.sum(vals))
            couples.append([int(male), int(female)])

        return couples

    def crossover(x, y , childs):
        childs.append( ( (x[0]+y[0])/2 , (x[1]+y[1])/2 ) )

    def mutate(x , i ,p):
        if np.random.uniform(0, 1) < p:
            x[i] = (x[i][0] + np.random.normal(0, 0.1, np.shape(x[i][0])),
                    x[i][1] + np.random.normal(0, 0.1, np.shape(x[i][1])))

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


    def GA2(self, pop_size, max_evals, input_size, layer_size, out_size, sexual_selection=sexual_selection, eval=eval,
           cross_over=crossover, mutate=mutate, merge_pop=merge_pop):

        # build graph
        start = time.perf_counter()
        iter_cnt =  0
        history = []
        sd = 75

        # create randon population
        pop = []
        pop_vals = []

        childs = []
        for _ in range(pop_size):
            pop.append((np.random.uniform(-1, 1, (input_size, layer_size)), np.random.uniform(-1, 1, (layer_size, out_size))) )

        parents_vals = [self.eval(pop[i], iter_cnt, i, layer_size, out_size ) for i in range(len(pop))]
        pop = [pop[pos] for pos in np.argsort(parents_vals)[::-1]]
        parents_vals = sorted(parents_vals)[::-1]
        # circle of life
        while (iter_cnt < max_evals):
            print(f"gen {iter_cnt} has passed f_max: {parents_vals[0]}")
            iter_cnt += 1
            childs.clear()
            couples = sexual_selection(pop , parents_vals)
            [cross_over(pop[couples[i][0]], pop[couples[i][1]], childs) for i in range(int(pop_size))]
            [mutate(childs, i, 0.7) for i in range(len(childs))]
            childs_vals = [self.eval(childs[i],iter_cnt, i, layer_size, out_size) for i in range(pop_size)]
            childs = [childs[pos] for pos in np.argsort(childs_vals)[::-1]]
            childs_vals = sorted(childs_vals)[::-1]
            pop, parents_vals = merge_pop(pop, childs , parents_vals, childs_vals)
            history.append(parents_vals[0])
            sd *= 0.99999
            if iter_cnt % 5000 == 0:
                print(f"processes {id}    iter: {iter_cnt}    f: {parents_vals[0]} sd: {sd}")
        print(f"GA took {(time.perf_counter() - start)/60} minutes for {max_evals} iteretion")
        return history, parents_vals[0], pop[0]























class NeuralNetwork:

    def __init__(self, input_size, layer_size, out_size, weights1, weights2):
        self.weights1 = weights1
        self.weights2 = weights2


    def feedforward(self, x):
        self.layer1 = self.sigmoid(np.dot(x, self.weights1))
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2))
        # print(f"layer1 : {self.layer1}")
        # print(f"output layer : {self.output}")
        num = np.argmax(self.output)
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


    def sigmoid(self, x):
        res =  1 / (1 +np.exp(-x))
        return res
    def get_weights(self):
        return (self.weights1 , self.weights2)