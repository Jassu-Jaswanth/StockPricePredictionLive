import numpy as np
from sklearn.svm import SVR

file_index = ["00", "01", "02"]

for f_name in file_index:
    
    # Reading data from input files
    data_in = []
    fd = open("Testcases/input/input"+f_name+".txt")
    data_in = fd.readlines()
    fd.close()
    rows = data_in[0]
    data_in = data_in[1:]
    
    # Formating data for better processing
    ref_data_in = []
    for l in data_in:
        ref_data_in.append(l.split('\t')[1].strip('\n'))
        
    # Constructing Data for training and validation
    X_train = []
    y_train = []
    X_test = []
    for i in range(len(ref_data_in)):
        if(ref_data_in[i][0] == 'M'):
            X_test.append(i+1)
        else:
            X_train.append(i+1)
            y_train.append(float(ref_data_in[i]))
    
    # Converting data to numpy array for training & testing
    X_train = np.array(X_train).reshape(-1, 1)
    y_train = np.array(y_train)
    X_test = np.array(X_test).reshape(-1, 1)

    # Defining kernel and training data
    ker_svr = SVR(kernel='poly', degree = 3) 
    ker_svr.fit(X_train, y_train)
    y_pred = ker_svr.predict(X_test)
    
    # Writing outputs to the desired files
    fd2 = open("Testcases/output/output"+f_name+".txt", 'w+')
    for y_hat in y_pred:
        fd2.write(str(y_hat) + "\n")
    fd2.close()