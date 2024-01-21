import math
import pandas as pd

# build a decision tree data structure

# it must have a constructor, a fit function, and a predict function
# constructor must have a max-depth input parameter
# fit function must take in train_x and train_y
# predict function must take in test_x

# it will use entropy criteria
# it will only take in numerical values for train_x
# it will only perform classification

def calculateEntropy(class_amounts):
    total = sum(class_amounts)
    entropy = 0
    for amount in class_amounts:
        p = amount/total
        if (p == 0): continue
        entropy -= p * math.log2(p)
    return entropy

class Attribute:
    def __init__(self, name, entropy, best_split_value, best_split_index):
        self.name = name
        self.entropy = entropy
        self.best_split_value = best_split_value
        self.best_split_index = best_split_index

    def print(self, spacing=""):
        print(spacing,"attribute name:", self.name)
        print(spacing,"attribute entropy:", self.entropy)
        print(spacing,"attribute threshold:", self.best_split_value)

class DecisionTreeNode:
    def __init__(self, attribute):
        self.attribute = attribute
        self.left = None
        self.right = None

    def getThreshold(self): return self.attribute.best_split_value

    def print(self, spacing=""):
        self.attribute.print(spacing)

class DecisionTreeLeafNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def print(self, spacing=""):
        print(spacing,self.value)

class NewDecisionTreeClassifier:

    def __init__(self, max_depth=math.inf):
        self.root = None
        self.max_depth = max_depth

    def fit(self, train_x, train_y):
        target_entropy = calculateEntropy(train_y.value_counts())
        self.root = self.recursivelyBuildTree(target_entropy, train_x, train_y)

    def recursivelyBuildTree(self, parent_entropy, x, y, depth=0):
        # get best attribute with lowest entropy
        best_attribute = self.getBestAttributeForSplit(x, y)

        # if information gain is negative or zero, we don't split any further
        # if entropy is zero, we don't split any further
        # if we reached max_depth, we don't split any further
        information_gain = parent_entropy - best_attribute.entropy
        if (best_attribute.entropy == 0 or information_gain <= 0 or depth == self.max_depth):
            majority = y.value_counts().idxmax()
            return DecisionTreeLeafNode(majority)

        # create decision tree node and give it the best attribute
        node = DecisionTreeNode(best_attribute)

        # divide the remaining data between left and right child nodes
        ltrain_x, ltrain_y, rtrain_x, rtrain_y = self.splitDataByAttribute(x, y, best_attribute)

        # handle case where there was only one attribute left
        if (rtrain_x.empty):
            left_majority = ltrain_y.value_counts().idxmax()
            right_majority = rtrain_y.value_counts().idxmax()
            node.left = DecisionTreeLeafNode(left_majority)
            node.right = DecisionTreeLeafNode(right_majority)
        else: # handle case where there was more than one attribute left
            node.left = self.recursivelyBuildTree(best_attribute.entropy, ltrain_x, ltrain_y, depth+1)
            node.right = self.recursivelyBuildTree(best_attribute.entropy, rtrain_x, rtrain_y, depth+1)

        return node


    def splitDataByAttribute(self, x, y, attribute):
        # join train_x and train_y
        data = pd.concat([x, y], axis=1)

        # sort joined dataframe by best_attribute
        data = data.sort_values(attribute.name)

        # reset indices
        data = data.reset_index()
        del data["index"]

        # remove the chosen attribute from the data
        del data[attribute.name]

        # divide joined dataframe at best_split_index
        data_before_index = data.iloc[:attribute.best_split_index+1]
        data_after_index = data.iloc[attribute.best_split_index+1:]

        # split each division into train_x and train_y
        left_train_x = data_before_index.drop(columns=y.name)
        left_train_y = data_before_index[y.name]

        right_train_x = data_after_index.drop(columns=y.name)
        right_train_y = data_after_index[y.name]

        # return the splits
        return left_train_x,left_train_y,right_train_x,right_train_y
         
             
    def getBestAttributeForSplit(self, x, y):
        attributes_list = []

        if isinstance(x, pd.Series): x.to_frame()

        # add each attributes information (name, entropy and best_split_value) to the attributes list
        for attribute in x:
             current_attribute = self.getEntropyOfAttribute(x, y, attribute)
             attributes_list.append(current_attribute)

        # return the attribute with the lowest entropy
        return min(attributes_list, key=lambda attribute: attribute.entropy)


    def getEntropyOfAttribute(self, train_x, train_y, attribute):
            # step 1: extract the attribute column and join it with the target column
            attribute_target_table = pd.concat([train_x[attribute], train_y], axis=1)

            # step 2: sort the values by attribute column and remove the indices
            attribute_target_table = attribute_target_table.sort_values(attribute).reset_index()
            del attribute_target_table["index"]

            # step 3: For each value
            #         > consider it as a potential split point 
            #         > divide the dataset into two groups: 
            #               1. values less than or equal to the current value 
            #               2. values greater than the current value
            #         > find target class distributions of each group
            #         > find the entropies of both distributions
            #         > calculate the final weighted entropy of the attribute with the target class
            #         > store the split values and final weighted entropies in a list as a tuple

            attribute_column = attribute_target_table.iloc[:,0]
            target_column = attribute_target_table.iloc[:,1]
            number_of_rows = len(attribute_target_table)
            entropy_splitValue_splitIndex_list = []

            for index in range(number_of_rows):
                # get the current split value
                attribute_split_value = attribute_column.iloc[index]

                # get the distribution of target values 
                # corresponding to the attribute values less than or equal to the current split value
                class_amounts_before_index = target_column.iloc[:index+1].value_counts()

                # get the distribution of target values 
                # corresponding to the attribute values greater than the current split value
                class_amounts_after_index = target_column.iloc[index+1:].value_counts()
                
                # calculate the entropies of the two groups
                entropy_before_index = calculateEntropy(class_amounts_before_index)
                entropy_after_index = calculateEntropy(class_amounts_after_index)

                # calculate the final weighted average
                weighted_entropy = sum(class_amounts_before_index) * entropy_before_index
                weighted_entropy += sum(class_amounts_after_index) * entropy_after_index
                weighted_entropy /= number_of_rows

                # add to list
                entropy_splitValue_splitIndex_list.append((weighted_entropy, attribute_split_value, index))

            # find which tuple has the lowest entropy
            lowest_entropy_tuple = min(entropy_splitValue_splitIndex_list, key=lambda x: x[0])
            return Attribute(attribute, lowest_entropy_tuple[0], lowest_entropy_tuple[1], lowest_entropy_tuple[2])
            
    def predict(self, test_x):
        predictions = []
        
        # predict induvidual rows and add result to list
        for i in range(len(test_x)):
            prediction = self.predictSingle(test_x.iloc[i])
            predictions.append(prediction)
        return predictions

    def predictSingle(self, test_row):
        current_node = self.root

        # traverse the tree to reach leaf node
        while (not isinstance(current_node, DecisionTreeLeafNode)):
            if (test_row[current_node.attribute.name] <= current_node.getThreshold()):
                current_node = current_node.left
            else: current_node = current_node.right

        # return value stored in leaf node
        return current_node.value
    
    def print(self):
        if (self.root != None): self.recursivePrint(self.root, 0)
    
    def recursivePrint(self, node, level):
        spacing = ""
        for i in range(level): spacing += "\t"

        node.print(spacing)

        if (node.left != None):
            print("\n",spacing,"left:")
            self.recursivePrint(node.left, level+1)
        if (node.right != None):
            print("\n",spacing,"right:")
            self.recursivePrint(node.right, level+1)