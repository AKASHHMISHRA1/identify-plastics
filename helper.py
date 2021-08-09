from statistics import mode
def compute_ln_norm_distance(vector1, vector2, n):
    vector_len = len(vector1)
    distance = 0
    for i in range(0, vector_len):
      diff = (abs(vector1[i] - vector2[i]))**n
      distance += diff
    distance = distance ** (1.0/n)
    return distance

def find_k_nearest_neighbors(train_X, test_example, k, n):
    indices_dist_pairs = []
    index= 0
    for train_elem_x in train_X:
      distance = compute_ln_norm_distance(train_elem_x, test_example, n)
      indices_dist_pairs.append([index, distance])
      index += 1
    indices_dist_pairs.sort(key = lambda x: x[1])
    top_k_pairs = indices_dist_pairs[:k]
    top_k_indices = [i[0] for i in top_k_pairs]
    return top_k_indices

def classify_points_using_knn(train_X, train_Y, test_X, k, n):
    test_Y = []
    for test_elem_x in test_X:
      top_k_nn_indices = find_k_nearest_neighbors(train_X, test_elem_x, k,n)
      top_knn_labels = []

      for i in top_k_nn_indices:
        top_knn_labels.append(train_Y[i])
      Y_values = list(set(top_knn_labels))

      max_count = 0
      most_frequent_label = -1
      for y in Y_values:
        count = top_knn_labels.count(y)
        if(count > max_count):
          max_count = count
          most_frequent_label = y
          
      test_Y.append(most_frequent_label)
    return test_Y    

def calculate_accuracy(predicted_Y, actual_Y):
    #TODO Complete the function implementation. Read the Question text for details
    count=0
    for x in range(len(predicted_Y)):
        if predicted_Y[x]==actual_Y[x]:
            count+=1
    return count/len(predicted_Y)

def get_best_k_using_validation_set(train_X, train_Y, validation_split_percent,n):
    #TODO Complete the function implementation. Read the Question text for details
    #print(train_X)
    #print(train_Y)
    no_of_train_X=((100-validation_split_percent)*len(train_X))//100
    #print(no_of_train_X)
    no_of_test_X=len(train_X)-no_of_train_X
    new_train_X=train_X[:no_of_train_X]
    test_X=train_X[no_of_train_X:]
    #print(new_train_X)
    #print(test_X)
    max=0
    best_k=0
    pre_max=0
    for k in range(1,len(train_X)+1):
        np=classify_points_using_knn(new_train_X,train_Y[:no_of_train_X],test_X,n,k)
        #print(np)
        max=calculate_accuracy(np,train_Y[no_of_train_X:])
        if max>pre_max:
            pre_max=max
            best_k=k
    return best_k