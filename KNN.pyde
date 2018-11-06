# Author: Height Yan
# Date: 2018/11/05
import random
random.seed(0)
global m
global theta; theta = 0
global testing_data; testing_data = []
global actual_predicted_pairs; actual_predicted_pairs = []
global pause; pause = True

# Costomize globals
global side_length; side_length = 400 #Initial size of the graph
global graphX, graphY; graphX, graphY = 500, 400 #Initial location of the KNN graph
global test_num; test_num = 0 #Keep the record of how many testcases were been tested
global train_num; train_num = 800 # The number of training data
def rule(test_case):
    r,g,b = test_case[0], test_case[1], test_case[2]
    return r < 100 and b > 100 or g < 120 and b < 80


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
    global graphX, graphY
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
    m.display_train(side_length, 100, 200)
    
    
    #Draw results
    offsetX, offsetY = 800, 200
    
    global pause
    global actual_predicted_pairs, testing_data
    testing_data = (generate_data(1)) if pause == False else testing_data
    for test_case in testing_data:
        actual = rule(test_case)
        predicted = m.predict(test_case)
        if pause == False:
            actual_predicted_pairs.append((actual, predicted))
    accuracy, recall, precision = get_accuracy_recall_precision(actual_predicted_pairs)
    popMatrix()
    pushMatrix()
    fill(0); yGap = 25; textSize(20); textAlign(LEFT)
    text("training_size = " + str(train_num), offsetX, offsetY - yGap)
    text("accuracy = " + str(accuracy), offsetX, offsetY)
    text("recall = " + str(recall), offsetX, offsetY + yGap)
    text("precision = " + str(precision),offsetX, offsetY + yGap*2)
    text("Test :" + str(testing_data), offsetX, offsetY + yGap*3)
    text("Test #:" + str(test_num), offsetX, offsetY + yGap*4)
    popMatrix()
    
    global test_num
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
        self.training_data = test_case_to_actual_category
        
    def display_train(self, side_length, offsetX, offsetY):
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
            ellipse(0,0,10,10)
            popMatrix()
        
    def predict(self, test_case):
        return self.KNN(test_case, k = 5)

    def Dist(self, test_case, train_case):
        return sqrt((test_case[0] - train_case[0])**2 + (test_case[1] - train_case[1])**2 + (test_case[2] - train_case[2])**2)

    def KNN(self, test_case, k):
        neighbors = []
        for train_case in self.training_data:
            neighbors.append([train_case, self.training_data[train_case],self.Dist(test_case, train_case)])
        neighbors = sorted(neighbors, key = lambda x:x[2])[:k]
        import random
        roll = random.randint(0,k-1)
        #print(len(neighbors), roll, neighbors[roll][1])
        
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
        fill(0); ellipse(0,0,10,10)
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
        return neighbors[roll][1]
def mousePressed():
    global pause
    pause = False if pause == True else True
def mouseWheel(event):
    global side_length
    side_length += event.getCount()*50
    
def generate_data(number_of_test_cases):
    data = []
    for _ in range(number_of_test_cases):
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        data.append(random_color)
    return data
    
