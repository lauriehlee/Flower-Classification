import sys
import csv
import copy
import numpy as np 
from anytree import Node, RenderTree
import math
import random
import matplotlib.pyplot as plt


def decision_tree(node, lst_of_classifications, attributes_list, original_attributes_list):
    
    # If all instances in data set belong to same class, define node N as leaf node, label w class, and return it
    # IMPORTANT: leaf nodes will be represented by a tuple in decision tree to show classification: (class label, accompanying data, attribute value, index of most recent attribute that led to this node)
    # Otherwise, non-leaf nodes in the tree's name will be identified by tuple (index of most recent attribute that led to this node, accompanying data, attribute value)
    homogenous = check_same_class(node.name[1])
    if homogenous:
        node.name = (node.name[1][0][16], node.name[1], node.name[2], node.name[0])
        lst_of_classifications.append(node.name[:2])
        return node

    # If there are no more attributes that can be tested, define node N as leaf node labeled w majority class in data set and return it
    if len(attributes_list) == 0:
        print("No more attributes can be tested. Size of dataset: ", len(node.name[1]))
        majority_class = find_majority_class(node.name[1])
        node.name = (majority_class, node.name[1], node.name[2], node.name[0])
        lst_of_classifications.append(node.name[:2])
        return node
    
    # Otherwise, select an attribute to add to the tree using the Information Gain criterion
    chosen_attribute = choose_next_attribute(node.name[1], attributes_list, original_attributes_list)
    chosen_attribute_index = original_attributes_list.index(chosen_attribute) # This is so that you can always find corresponding index in data set even if attribute list is continually modified.

    # Add to node N, one branch for each possible value of the selected attribute with non-empty data set
    # Partition the instances / examples -- assign each instance to its corresponding branch, based on the value of that instance's attribute
    partitioned_instances = partition_based_on_attribute(node.name[1], chosen_attribute_index)

    # Remove tested attribute from the attributes list, and recursively call on node N with modified attribute list & original attribute list. 
    # This is so that different child branches may use the same attributes instead of restricting them to attributes their sibling did not use.
    # If it results in an empty partition based on the attribute, define node N as leaf node labeled w majority class in data set and return it
    path_attributes_list = copy.deepcopy(attributes_list) # Modification after HW1: make a deep copy of attributes list so that you can use the same attribute in diff paths that stem from a common ancestor
 
    path_attributes_list.remove(chosen_attribute)
    majority_class = find_majority_class(node.name[1])

    if len(partitioned_instances[0]) > 0:
        child1 = Node((chosen_attribute_index, partitioned_instances[0], 0), parent=node)
        child1 = decision_tree(child1, lst_of_classifications, path_attributes_list, original_attributes_list)
    
    if len(partitioned_instances[0]) == 0:
        child1 = Node((majority_class, [], 0, chosen_attribute_index), parent=node)
        lst_of_classifications.append(child1.name[:2])
    
    if len(partitioned_instances[1]) > 0:
        child2 = Node((chosen_attribute_index, partitioned_instances[1], 1), parent=node)
        child2 = decision_tree(child2, lst_of_classifications, path_attributes_list, original_attributes_list)
    
    if len(partitioned_instances[1]) == 0:
        child2 = Node((majority_class, [], 1, chosen_attribute_index), parent=node)
        lst_of_classifications.append(child2.name[:2])
    
    if len(partitioned_instances[2]) > 0:
        child3 = Node((chosen_attribute_index, partitioned_instances[2], 2), parent=node)
        child3 = decision_tree(child3, lst_of_classifications, path_attributes_list, original_attributes_list)
    
    if len(partitioned_instances[2]) == 0:
        child3 = Node((majority_class, [], 2, chosen_attribute_index), parent=node)
        lst_of_classifications.append(child3.name[:2])


    return node

def return_classification(letter):
    if letter == '0':
        return "Democrat"
    else:
        return "Republican"

def check_same_class(data_set):
    party = data_set[0][16]
    for elem in data_set:
        if elem[16] != party:
            return False
    
    return True

def find_majority_class(data_set):
    democratic = 0
    republican = 0
    for elem in data_set:
        if elem[16] == '0':
            democratic += 1
        else:
            republican += 1
    
    if democratic > republican:
        return '0'
    else:
        return '1'

def choose_next_attribute(data_set, attributes_list, original_attributes_list):
    ig_of_attributes = []
    for attribute in attributes_list:
        # Calculate the Information Gain of each attribute
        ig_of_attributes.append((attribute, information_gain(data_set, original_attributes_list.index(attribute))))

    # Find the tuple that has the maximum IG value among all tuples
    informative_attribute = max(ig_of_attributes, key=lambda x: (x[1], x[0]))  

    return informative_attribute[0]

def information_gain(data_set, index_of_attribute):
    total_instances = (float)(len(data_set))
    # Find entropy of original data set.
    original_entropy = find_entropy(data_set)
    
    # Partition instances based on attribute
    partitioned_instances = partition_based_on_attribute(data_set, index_of_attribute)

    # Find entropy of "yea" of attribute
    yea_entropy = find_entropy(partitioned_instances[0])

    # Find entropy of "nay" of attribute
    nay_entropy = find_entropy(partitioned_instances[1])

    # Find entropy of "no vote" of attribute
    no_vote_entropy = find_entropy(partitioned_instances[2])

    # Find weighted average entropy of resulting partitions
    weighted_average_entropy = ((float)(len(partitioned_instances[0]))/total_instances * yea_entropy) + ((float)(len(partitioned_instances[1]))/total_instances * nay_entropy) + ((float)(len(partitioned_instances[2]))/total_instances * no_vote_entropy)
    
    return original_entropy - weighted_average_entropy

def find_entropy(data_set):
    democratic = 0
    republican = 0
    for elem in data_set:
        if elem[16] == '0':
            democratic += 1
        else:
            republican += 1
    
    total_instances = (float)(len(data_set))

    prob_democrat = 0
    prob_republican = 0
    if total_instances > 0:
        prob_democrat = (float)(democratic) / total_instances
        prob_republican = (float)(republican) / total_instances

    log_democrat = 0
    log_republican = 0
    if prob_democrat > 0:
        log_democrat = math.log2(prob_democrat)
    if prob_republican > 0:
        log_republican = math.log2(prob_republican)

    return ((prob_democrat * -1)*(log_democrat)) - ((prob_republican)*(log_republican))

def partition_based_on_attribute(data_set, chosen_attribute_index):
    yea = []
    nay = []
    no_vote = []
    for elem in data_set:
        if elem[chosen_attribute_index] == '0':
            yea.append(elem)
        elif elem[chosen_attribute_index] == '1':
            nay.append(elem)
        else:
            no_vote.append(elem)
    
    return [yea, nay, no_vote]

def make_it_pretty(lst):

    for elem in lst:
        print(elem)

def partition(data_set):
    # (a) Shuffle the dataset to make sure that the order in which examples appear in the dataset file does not affect the learning process.
    random.shuffle(data_set)

    # (b) Randomly partition the dataset into disjoint two subsets: training set, containing 80% of instances selected at random
    #     and testing set, containing the other 20% of the instances.
    stop_index = int(0.8 * len(data_set))
    training_set = copy.deepcopy(data_set[:stop_index])
    testing_set = copy.deepcopy(data_set[stop_index:])

    return (training_set, testing_set)

def accuracy(classifications):
    # Classifications organized as: (class, instance)
    correct = 0
    total_instances = 0
    for clss, instance in classifications:
        total_instances += 1
        if clss == (instance[16]):
            correct += 1
    
    return (float)(correct) / (float)(total_instances)

def predict_classes(data_set, decision_tree):
    classifications = []
    i = 0
    for instance in data_set:
        clss = explore_decision_tree(instance, decision_tree)
        classifications.append((clss, instance))
        i += 1


    return classifications

def explore_decision_tree(instance, current_node):
    while len(current_node.children) > 0:
        for child in current_node.children:
            if len(child.children) > 0:
                attribute_index = (int)(child.name[0]) # If non-leaf node, child's name fields would be (attribute index, data set, attribute value)
            else:
                attribute_index = (int)(child.name[3]) # If leaf node, child's name fields would be (classification, data set, attribute value, attribute index)
            # If the value of attribute matches the value of attribute in one of the items in dataset, move onto this child node


            if (((int)(instance[attribute_index])) == (child.name[2])):
                current_node = child
                break

    return current_node.name[0] # Returns the class of the leaf node in decision tree

if __name__ == "__main__":
    # Parse command line arguments (python3 decisionTreeAlgorithm.py house_votes_84.csv)
    csv_name = sys.argv[1]

    # Establish reading through csv: [0, 15] = attributes; [16] = political party
    original_dataset = []
    with open(csv_name, mode = 'r') as csvfile:
        file = csv.reader(csvfile)
        i = 0
        for line in file: 
            if i == 0:
                original_attributes_list = line[:16]
                i += 1
                continue
            if len(line) > 0: 
                original_dataset.append(line)
    
    accuracies_training = []
    accuracies_testing = []

    for j in range(100):
        data_entries = copy.deepcopy(original_dataset) 
        attributes_list = copy.deepcopy(original_attributes_list)

        training_set, testing_set = partition(data_entries)

        training_root = Node(("", training_set))
        lst_of_classifications = []
        result_training_tree = decision_tree(training_root, lst_of_classifications, attributes_list, original_attributes_list)

        # Predict for training set
        lst_of_classifications_training = predict_classes(training_set, result_training_tree)

        # Compute accuracy of training set
        accuracy_training = accuracy(lst_of_classifications_training)
        accuracies_training.append(accuracy_training)

        # Predict for testing set
        lst_of_classifications_testing = predict_classes(testing_set, result_training_tree)

        # Compute accuracy of testing set
        accuracy_testing = accuracy(lst_of_classifications_testing)
        accuracies_testing.append(accuracy_testing)


    
    avg_accuracy_training = np.average(accuracies_training)
    std_training = np.sqrt(np.mean((accuracies_training - np.mean(accuracies_training)) ** 2)) 
    print("Average accuracy training: ", avg_accuracy_training)
    print("Stdv training: ", std_training)

    avg_accuracy_testing = np.average(accuracies_testing)
    std_testing = np.sqrt(np.mean((accuracies_testing - np.mean(accuracies_testing)) ** 2)) 
    print("Average accuracy testing: ", avg_accuracy_testing)
    print("Stdv testing: ", std_testing)

    
    fig, axs = plt.subplots(2)

    axs[0].hist(accuracies_training, bins=5, edgecolor='black')
    axs[0].set_xlabel('Accuracy')
    axs[0].set_ylabel('Accuracy Frequency of Training Data')
    axs[0].set_title('Accuracy Histogram for Training Data')

    axs[1].hist(accuracies_testing, bins=5, edgecolor='black')
    axs[1].set_xlabel('Accuracy')
    axs[1].set_ylabel('Accuracy Frequency of Testing Data')
    axs[1].set_title('Accuracy Histogram for Testing Data')
    plt.tight_layout()
 
    plt.show()
