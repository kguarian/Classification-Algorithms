import numpy as np
from sklearn.datasets import load_breast_cancer

df = load_breast_cancer()

print("cancer keys: ", df.keys())
print("cancer feature names: ", df["feature_names"])
print("cancer data: ", df["data"])

# both lists of indices
neighborhood = None
indices = None


# unoptimized.
def shiftRight(array, start_index, k):
    i = k-1
    while i > start_index:
        array[i] = array[i-1]
        i = i-1


def scalar_dist_1d(v1, v2):
    return abs(v1+v2)


def print_features(dataset):
    names = dataset["feature_names"]
    data = dataset["data"]
    for i in range(0, len(data)):
        for j in range(0, len(names)):
            print(names[j], ": ", data[i][j], end="\t")
        print("\n")

    return


# returns array of indices (k nearest neighbors)
def insert_neighbor(distance_array, index_array, df, index, k_value, feature_index, value):
    global neighborhood
    global indices

    upper_bound = 1
    print(index)

    neighborhood = distance_array.copy()

    # Ω(n),θ(n), O(n)  = (n^2/2+nlog n)
    # NTS: don't pee and moan about python's speed if you're doing search/replace at O(n^2) ove n elements, k times for a dataset of size k, when everything will be sorted after the first run.
    # this would be slow anywhere (to scale, at least).

    # sort distances and indices
    neighborhood_sz = len(df)
    print(neighborhood)

    distval = scalar_dist_1d(df[index][feature_index], value)
    if index == 0:
        index_array[0] = 0
        distance_array[0] = distval
        return distance_array, indices

    else:
        if index < k_value:
            currindex = index
            while (distval < distance_array[currindex] or index_array[currindex]==np.inf)  and currindex >= 0:
                currindex -= 1
            if distval < distance_array[currindex]:
                index_array[currindex] = index
                distance_array[currindex] = distval
            print("index < k case")
            return distance_array, indices
        else:
            currindex = k_value-1
            while distval < distance_array[currindex] and currindex >= 0:
                currindex -= 1
        currindex+=1
        shiftRight(index_array, currindex, k_value)
        shiftRight(distance_array, currindex, k_value)

        print("index is", str(index))
        print("distval is", str(distval))
        if distval < distance_array[currindex] or distance_array[currindex] == np.inf:
            if currindex == k_value:
                print("value >", distance_array[k_value])
                return distance_array, indices
            index_array[currindex] = index
            distance_array[currindex] = distval

        print(distance_array, indices)
        print("done")
        print("exiting insert")

    return distance_array, indices


def k_nn(feature_name, value, dataset, k):
    global neighborhood
    global indices

    yea = 0
    nea = 0
    # edge case. ints are bigger.

    if len(dataset) < k:
        print("dataset too small for k-NN")
        exit(1)

    feature_index = None
    # then using the typical (if < max_dist then ...)
    neighborhood = np.zeros(k)
    indices = np.zeros(k)

    for i in range(k):
        indices[i] = neighborhood[i] = np.inf

    for i in range(0, len(dataset["feature_names"])):
        if dataset["feature_names"][i] == feature_name:
            feature_index = i
            break
        if i == len(dataset["feature_names"])-1:
            print("didn't find feature name")
            exit(1)

    names = dataset["feature_names"]
    data = dataset["data"]
    for i in range(0, len(names)):
        if feature_name == names[i]:
            feature_index = i

    for i in range(len(data)):
        neighborhood, indices = insert_neighbor(
            neighborhood, indices, data, i, k, feature_index, value)
    print(dataset["target"])
    print(neighborhood)
    for i in range(len(indices)):
        if dataset["target"][i] == 1.0:
            yea = yea+1
        else:
            nea = nea+1

    if yea > nea:
        return 1
    return 0

# value is integer


ref = {
    "target": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    "data": [[101], [102], [150], [200], [300], [-2], [5], [-20], [50], [75]],
    "feature_names": ["a"]
}

print(k_nn("mean perimeter", 130, df, 4))

c = k_nn('mean perimeter', 3, df, 4)

print(c)
