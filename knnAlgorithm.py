import sys
import csv
from queue import PriorityQueue
import math
import random
import numpy as np 
import matplotlib.pyplot as plt
import array
import time
import copy



def kNN(k_value, training_set, testing_set):

    # (c) Train the k-NN algorithm using only the data in the training set
    # Obtain list of classifications for training set.
    list_of_classifications = classify(k_value, training_set, training_set)

    # Obtain list of classifications for testing set, given the training set's classifications.
    list_of_classifications_testing = classify(k_value, training_set, testing_set)


    # (d) Compute the accuracy of the k-NN model when used to make predic for instances in training set.
    accuracy_of_training_set = accuracy_of_set(list_of_classifications)

    
    # (e) Compute the accuracy of the k-NN model when used to make predic for instances in testing set.
    accuracy_of_testing_set = accuracy_of_set(list_of_classifications_testing)


    return (accuracy_of_training_set, accuracy_of_testing_set)

def classify(k_value, training_set, testing_set):
    list_of_classifications = []
    sample_index = 0
    # For each instance:
    # Use Euclidean Distance and priority queue with smallest distance being first to order neighbors of the sample
    for sample in testing_set:
        neighbors = PriorityQueue()
        compared_sample_index = 0
        for compared_sample in training_set:
            attributes_summed = 0
            for i in range(4): # 0, 1, 2, 3
                attributes_summed += (math.pow(((float)(sample[i]) - ((float)(compared_sample[i]))), 2))
            
            euclidean_dist = math.sqrt(attributes_summed)
            neighbors.put((euclidean_dist, compared_sample))
            
            compared_sample_index += 1
        

        species = {
            "Iris-setosa": 0,
            "Iris-versicolor": 0,
            "Iris-virginica": 0
        }
        # Increment each species type while extracting k closest neighbors
        for p in range(int(k_value)):
            curr = neighbors.get()[1]
            neighbor_class = curr[4]
            for specimen in species.keys():
                if str(neighbor_class) == specimen:
                    species[specimen] += 1
                    
        # Find the majority class of the k closest neighbors
        maximum = -1
        majority_class = ""
        for specimen in species.keys():
            if (species[specimen] > maximum):
                maximum = species[specimen]
                majority_class = specimen
            # When the values are equal, return a randomly chosen species
            elif (species[specimen] == maximum):
                random_int = random.randint(0, 1)
                if random_int == 0:
                    maximum = species[specimen]
                    majority_class = specimen
                # Otherwise, the majority class stays the same.


        # Set the instance to be this class
        list_of_classifications.append((sample, majority_class))
        sample_index += 1


    return list_of_classifications

def make_it_pretty(lst):
    for elem in lst:
        print(elem)

def accuracy_of_set(lst):
    correct = 0
    for elem in lst:
        if elem[0][4] == elem[1]:
            correct += 1
    
    return correct / len(lst)

def preprocess(data_set):
    min_length_sepal = (float)(100000000)
    max_length_sepal = (float)(-1)
    min_width_sepal = (float)(100000000)
    max_width_sepal = (float)(-1)
    min_length_petal = (float)(100000000)
    max_length_petal = (float)(-1)
    min_width_petal = (float)(100000000)
    max_width_petal = (float)(-1)

    # Find possible ranges of all features
    for elem in data_set:
        elem[0] = (float)(elem[0])
        elem[1] = (float)(elem[1])
        elem[2] = (float)(elem[2])
        elem[3] = (float)(elem[3])

        if elem[0] < min_length_sepal:
            min_length_sepal = elem[0]
        if elem[0] > max_length_sepal:
            max_length_sepal = elem[0]
        if elem[1] < min_width_sepal:
            min_width_sepal = elem[1]
        if elem[1] > max_width_sepal:
            max_width_sepal = elem[1]
        if elem[2] < min_length_petal:
            min_length_petal = elem[2]
        if elem[2] > max_length_petal:
            max_length_petal = elem[2]
        if elem[3] < min_width_petal:
            min_width_petal = elem[3]
        if elem[3] > max_width_petal:
            max_width_petal = elem[3]

    
    # Normalize features according to possible ranges
    for elem in data_set:
        elem[0] = ((elem[0] - min_length_sepal)/(max_length_sepal - min_length_sepal))
        elem[1] = ((elem[1] - min_width_sepal)/(max_width_sepal - min_width_sepal))
        elem[2] = ((elem[2] - min_length_petal)/(max_length_petal - min_length_petal))
        elem[3] = ((elem[3] - min_width_petal)/(max_width_petal - min_width_petal))


    return data_set

def partition(data_set):
    # (a) Shuffle the dataset to make sure that the order in which examples appear in the dataset file does not affect the learning process.
    random.shuffle(data_set)

    # (b) Randomly partition the dataset into disjoint two subsets: training set, containing 80% of instances selected at random
    #     and testing set, containing the other 20% of the instances.
    stop_index = int(0.8 * len(data_set))
    training_set = data_set[:stop_index]
    testing_set = data_set[stop_index:]    

    # Pre-process features of training set
    training_set = preprocess(training_set) 

    # Pre-process features of testing set
    testing_set = preprocess(testing_set) 

    return (training_set, testing_set)


def show_graph(training_points, testing_points):

    x = np.array([coord[0] for coord in training_points])
    y = np.array([coord[1] for coord in training_points])
    errors = np.array([elem[2] for elem in training_points])

    fig, (ax0, ax1) = plt.subplots(nrows=2)
    plt.xlabel('k')
    plt.ylabel('Avg Accuracy')
    ax0.set_ylim(0.72, 1.0)
    ax0.errorbar(x, y, yerr=errors, fmt='-o')
    ax0.set_title('Accuracy of Training Set based on k')
    
    m = np.array([coord[0] for coord in testing_points])
    n = np.array([coord[1] for coord in testing_points])
    errors = np.array([elem[2] for elem in testing_points])

    plt.xlabel('k')
    plt.ylabel('Avg Accuracy')
    ax1.set_ylim(0.72, 1.0)
    ax1.errorbar(m, n, yerr=errors, fmt='-o')
    ax1.set_title('Accuracy of Testing Set based on k')


    plt.show()


if __name__ == "__main__":
    # Parse command line arguments (python3 knnAlgorithm.py iris.csv)
    csv_name = sys.argv[1]

    # Establish reading through csv:
    # First column: Length of sepal of flower
    # Second column: Width of sepal of flower
    # Third column: Length of petal of flower
    # Fourth column: Width of petal of flower
    data_entry = []
    with open(csv_name, mode = 'r') as csvfile:
        file = csv.reader(csvfile)
        for line in file: 
            if len(line) > 0: 
                data_entry.append(line)
     

    k = 1
    training_points = []
    testing_points = []

    original_dataset = copy.deepcopy(data_entry)
    while k < 53:
        training_arr = []
        testing_arr = []
        # Call k-NN algorithm 20 times for this value of k
        for i in range(20):
            training_set, testing_set = partition(data_entry)
            acc_training, acc_testing = kNN(k, training_set, testing_set)
            training_arr.append(acc_training)
            testing_arr.append(acc_testing)
            data_entry = copy.deepcopy(original_dataset)
        
        training_arr = np.array(training_arr)
        testing_arr = np.array(testing_arr)
        avg_accuracy_training = np.average(training_arr)

        avg_accuracy_testing = np.average(testing_arr)
        std_training = np.sqrt(np.mean((training_arr - np.mean(training_arr)) ** 2)) 
        std_testing = np.sqrt(np.mean((testing_arr - np.mean(testing_arr)) ** 2)) 

        training_points.append((k, avg_accuracy_training, std_training))
        testing_points.append((k, avg_accuracy_testing, std_testing))

        k += 2
 
    show_graph(training_points, testing_points)
        