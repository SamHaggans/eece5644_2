import csv



# Name of file to process
filename = './mushroom/agaricus-lepiota.data'


# Learn the names of all categories present in the dataset,
# and map them to 0,1,2,...

col_maps = {}


print("Processing",filename,"...",end="")
with open(filename) as csvfile:
    fr = csv.reader(csvfile, delimiter=',') 
    rows = 0
    for row in fr:
        rows += 1
        if rows == 1:
            columns = len(row)
            for c in range(columns):
                col_maps[c] = {}

        for (c,label) in enumerate(row):
            if label not in col_maps[c]:
                index = len(col_maps[c])
                col_maps[c][label] = index
print(" done")
                
print("Read %d rows having %d columns." % (rows,columns))
print("Category maps:")
for c in range(1, columns):
    print(len(col_maps[c]))
print(col_maps)


# Construct matrix X, containing the mapped 
# features, and vector y, containing the mapped
# labels.

X = []
y = []

print("Converting",filename,"...",end="")
with open(filename) as csvfile:
    fr = csv.reader(csvfile, delimiter=',') 
    for row in fr:
        label = row[0]
        y.append(col_maps[0][label])

        features = []
        for (c,label) in enumerate(row[1:]):
            features.append(col_maps[c+1][label])
        
        X.append(features)

print(" done")


# Store them to files.

# with open('X_msrm.csv', 'w') as csvfile:
#     fw = csv.writer(csvfile, delimiter=',')
#     for features in X:
#         fw.writerow(features)

# with open('y_msrm.csv', 'w') as csvfile:
#     fw = csv.writer(csvfile, delimiter=',')
#     for label in y:
#         fw.writerow([label])




                

