import numpy as np
import json
from sklearn.neighbors import KNeighborsClassifier

sub_num = {"Physics" : 0,"Chemistry" : 1,"English" : 2,"Biology" : 3,"PhysicalEducation" : 4,"Accountancy" : 5,"BusinessStudies" : 6,"ComputerScience" : 7,"Economics" : 8}


# Initialize a model
model_kcls = KNeighborsClassifier(n_neighbors=3)

# Train the above model
def train():
    fd = open("trainingAndTest/training.json", "r") 
    num_data_points = int(fd.readline().strip())
    X = np.zeros((num_data_points, 9))
    y = np.zeros((num_data_points, ))
    
    line_num = 0
    for data_point in fd.readlines():
        data_point = data_point.strip()
        data_point = json.loads(data_point)

        tmp = np.zeros(9)
        for sub in data_point.keys():
            if sub == "serial":
                continue
            elif sub == "Mathematics":
                y[line_num] = data_point[sub]
                continue
            tmp[sub_num[sub]] = data_point[sub]

        
        X[line_num] = tmp
        line_num += 1
    model_kcls.fit(X, y)
    fd.close()

# function to solve a given input
def solve(filename):

    fd = open(filename, "r")
    num_data_points = int(fd.readline().strip())

    X_test = np.zeros((num_data_points, 9))

    line_num = 0
    for data_point in fd.readlines():
        data_point = data_point.strip()
        data_point = json.loads(data_point)

        tmp = np.zeros(9)
        for key, value in data_point.items():
            if key == "serial":
                continue
            tmp[sub_num[key]] = value
        
        X_test[line_num] = tmp
        line_num += 1
    fd.close()
    return model_kcls.predict(X_test)

# file indexes for reading and writing input/output
file_index = ["00","01"]

# Main function
if __name__ == "__main__":
    train()
    
    for f_ind in file_index:
        y_test = solve(f"Testcases/input/input{f_ind}.txt")

        fd = open(f"Testcases/output/output{f_ind}.txt", "w")
        for y in y_test:
            fd.write(f"{int(y)}\n")
        fd.close()
