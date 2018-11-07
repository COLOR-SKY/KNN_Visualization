# Author: Height Yan
# Date: 2018/11/05
import random
random.seed(0)
global m
global theta; theta = 0
global testing_data; testing_data = []
global actual_predicted_pairs; actual_predicted_pairs = []
global pause, boundary; pause = True; boundary = False
global predicted; predicted = None

# Costomize globals
global side_length; side_length = 400 #Initial size of the graph
global graphX, graphY; graphX, graphY = 500, 400 #Initial location of the KNN graph
global test_num; test_num = 0 #Keep the record of how many testcases were been tested
global train_num; train_num = 800 # The number of training data
global nb_num; nb_num = 5 # The number of neighbors

def rule(test_case):
    r,g,b = test_case[0], test_case[1], test_case[2]
    return r < 100 and b > 100 or g < 120 and b < 80
def generate_data(number_of_test_cases):
    data = []
    for _ in range(number_of_test_cases):
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        data.append(random_color)
    return data


def setup():
    size(1400, 800, P3D)
    background(255)
    actual_function = rule
    global train_num 
    training_data = generate_data(train_num)
    test_case_to_actual_category = {test_case: actual_function(test_case)
        for test_case in training_data}
    global m
    m = Model()
    m.train(test_case_to_actual_category)


def draw():
    background(255)
    #Visualizing training_data
    pushMatrix()
    global theta,m,side_length
    theta = map(mouseX, 0, width, 0, 4*PI)
    #theta += 0.005
    global graphX, graphY, boundary
    translate(graphX,graphY)
    rotateY(theta)
    
    #draw coordinate lines
    strokeWeight(3)
    linex, liney, linez = - side_length/2, side_length/2, -side_length/2
    stroke(side_length,0,0);line(linex, liney, linex, linex + side_length, liney, linez)
    stroke(0,side_length,0);line(linex, liney, linex, linex, liney - side_length, linez)
    stroke(0,0,side_length);line(linex, liney, linex, linex, liney, linez + side_length)
    strokeWeight(1);stroke(0)
    #draw training data
    m.display_train(side_length, 100, 200, 8, boundary)
    #Draw results
    offsetX, offsetY = 800, 200
    
    global pause, nb_num, test_num
    global actual_predicted_pairs, testing_data, predicted
    testing_data = (generate_data(1)) if pause == False else testing_data 
    testing_data = [[0,0,0]] if testing_data == [] else testing_data
    for test_case in testing_data:
        m.predict(test_case)
        if pause == False:
            predicted = m.predict(test_case)
            actual = rule(test_case)
            actual_predicted_pairs.append((actual, predicted))
    accuracy, recall, precision = get_accuracy_recall_precision(actual_predicted_pairs)
    m.display_KNN()
    popMatrix()
    pushMatrix()
    fill(0,0,0,255); yGap = 25; textSize(20); textAlign(LEFT)
    text("neighbor_num  = " + str(nb_num), offsetX, offsetY - yGap*2)
    text("training_size = " + str(train_num), offsetX, offsetY - yGap)
    text("accuracy = " + str(accuracy), offsetX, offsetY)
    text("recall = " + str(recall), offsetX, offsetY + yGap)
    text("precision = " + str(precision),offsetX, offsetY + yGap*2)
    text("Test : R(" + str(testing_data[0][0]) + ") G(" + str(testing_data[0][1]) + ") B(" + str(testing_data[0][2]) + ") Prediction : " + str(predicted), offsetX, offsetY + yGap*3)
    text("Test# : " + str(test_num), offsetX, offsetY + yGap*4)
    popMatrix()

    test_num = test_num + 1 if pause == False else test_num
        
    
def get_accuracy_recall_precision(actual_predicted_pairs):
    truth_table = {
        (True, True) : 'tp',
        (True, None) : 'tp',
        (True, False) : 'fn',
        (False, False) : 'tn',
        (False, None): 'tn',
        (False, True) : 'fp'
    }
    result_table = {'tp' : 0, 'tn' : 0, 'fp' : 0, 'fn' : 0}
    for pair in actual_predicted_pairs:
        key = truth_table[pair]
        result_table[key] += 1
    accuraccy = float((result_table['tp'] + result_table['tn'])*1.0 / len(actual_predicted_pairs)) if len(actual_predicted_pairs) > 0 else 0.0
    recall = result_table['tp']*1.0 / (result_table['tp'] + result_table['fn']) if (result_table['tp'] + result_table['fn']) != 0 else 0.0
    precision = result_table['tp']*1.0 / (result_table['tp'] + result_table['fp']) if (result_table['tp'] + result_table['fp']) !=0 else 0.0
    return round(accuraccy,3), round(recall,3), round(precision,3)

class Model:
    def train(self, test_case_to_actual_category):
        k = 10
        self.training_data = test_case_to_actual_category
        self.boundary = []
        self.neighbors = []
        self.test_case = [10000,10000,10000]
        for train_case in self.training_data:
            neighbors = []
            for other_case in self.training_data:
                if other_case != train_case:
                    neighbors.append([other_case, self.training_data[other_case], self.Dist(other_case, train_case)])
            neighbors = sorted(neighbors, key = lambda x:x[2])[:k]
            for closest in [n[0] for n in neighbors]:
                like = self.training_data[train_case]
                nlike = self.training_data[closest]
                if like != nlike:
                    difference = PVector(train_case[0]+closest[0], train_case[1]+closest[1], train_case[2]+closest[2])
                    difference.mult(0.5)
                    self.boundary.append([difference.x, difference.y, difference.z])
            
            
    def display_train(self, side_length, offsetX, offsetY, radius, boundary = False):
        for data in self.training_data:
            x,y = -side_length/2, -side_length/2
            newx,newy, newz = map(data[0],0,255,0,side_length), map(data[1],0,255,0,side_length), map(data[2],0,255,0,side_length)
            result = self.training_data[data]
            pushMatrix()
            translate(x, y, newz + x)
            translate(newx, side_length - newy, 0)
            stroke(0, 50)
            fill(0, 50) if result == False else fill(255, 50)
            ellipseMode(CENTER)
            global theta
            rotateY(-theta)
            ellipse(0,0,radius,radius)
            popMatrix()
        
        if boundary is True:
            for data in self.boundary:
                x,y = -side_length/2, -side_length/2
                newx,newy, newz = map(data[0],0,255,0,side_length), map(data[1],0,255,0,side_length), map(data[2],0,255,0,side_length)
                pushMatrix()
                translate(x, y, newz + x)
                translate(newx, side_length - newy, 0)
                stroke(0, 50)
                fill(0)
                ellipseMode(CENTER)
                global theta
                rotateY(-theta)
                ellipse(0,0,radius,radius)
                popMatrix()
        
        
    def predict(self, test_case):
        global nb_num
        return self.KNN(test_case, k = nb_num)

    def Dist(self, test_case, train_case):
        return sqrt((test_case[0] - train_case[0])**2 + (test_case[1] - train_case[1])**2 + (test_case[2] - train_case[2])**2)

    def KNN(self, test_case, k):
        self.test_case = test_case
        neighbors = []
        for train_case in self.training_data:
            neighbors.append([train_case, self.training_data[train_case],self.Dist(test_case, train_case)])
        self.neighbors = sorted(neighbors, key = lambda x:x[2])[:k]
        import random
        roll = random.randint(0,k-1)
        #print(len(neighbors), roll, neighbors[roll][1])
        return self.neighbors[roll][1]
        
    def display_KNN(self):
        test_case = self.test_case
        neighbors = self.neighbors
        #draw train data
        global side_length
        x,y = -side_length/2, -side_length/2
        newx, newy, newz = map(test_case[0],0,255,0,side_length), map(test_case[1],0,255,0,side_length), map(test_case[2],0,255,0,side_length)
        pushMatrix()
        translate(x, y, newz + x)
        translate(newx, newy, 0)
        stroke(0, 50); ellipseMode(CENTER)
        global theta
        rotateY(-theta)
        fill(0,0,255); ellipse(0,0,10,10)
        popMatrix()
        #Draw Neighbors
        for n in neighbors:
            nx, ny, nz = map(n[0][0],0,255,0,side_length), map(n[0][1],0,255,0,side_length), map(n[0][2],0,255,0,side_length)
            #display neighbor's data
            #Display neighbors
            pushMatrix()
            strokeWeight(3)
            translate(x, y, nz + x)
            translate(nx, ny, 0)
            stroke(0, 50); ellipseMode(CENTER)
            global theta
            rotateY(-theta)
            textAlign(CENTER);
            fill(255,200,0,90); ellipse(0,0,13,13); fill(0); textSize(10);text(str(n[0]), 0, -10)
            textSize(8); text(" Distance: " + str(round(n[2],2)),0,-20)
            popMatrix()
        
def mousePressed():
    global pause
    pause = False if pause == True else True
def mouseWheel(event):
    global side_length
    side_length += event.getCount()*50
def keyPressed():
    global nb_num, graphX, graphY, pause, boundary
    if keyCode == UP:
        nb_num += 1
    if keyCode == DOWN:
        nb_num = nb_num - 1  if nb_num > 1 else 1
    if key == 'a':
        graphX -= 50
    if key == 'd':
        graphX += 50
    if key == 'w':
        graphY -= 50
    if key == 's':
        graphY += 50
    if key == ' ':
        pause = False if pause == True else True
    if key == 'b':
        boundary = False if boundary == True else True


    
