
# coding: utf-8


from pyspark import SparkConf, SparkContext
spark = SparkSession.builder.getOrCreate()
from pyspark.ml.feature import StopWordsRemover
sc = SparkContext.getOrCreate()
#Initializing as tokenized data.
sentenceData = spark.createDataFrame([
    (0, ["clifford", "the", "big", "red", "dog","is","a","great","friend"]),
    (1, ["mary", "favorite", "animal", "is", "a","dog"])
], ["id", "raw"])

#since the data is small we would coalesce it in 1 partition (not written in the code here)

remover = StopWordsRemover(inputCol="raw", outputCol="filtered")  #remove stopwords from data
filtered = remover.transform(sentenceData)
filtered = filtered.drop("raw") #drop the raw columns
filtered = filtered.rdd

c = filtered.flatMapValues(lambda x:x)

d = c.map(lambda x:(x[1],x[0]))
"""for merging with future batch process, sort this RDD and store in a file. Sort the new words and take union of the two. The results would appear 
in grouping or combining"""



e = d.groupByKey().mapValues(set)

e = e.zipWithIndex()
#Cartesian would generate 2^n pairs but since we are taking bi-directional equality into consideration we need to find combinations. 
#we eliminate duplicate data by indexing the rdd and filtering by index.
wod_pairs = e.cartesian(e).filter(lambda x:x[0][1] <= x[1][1]).map(lambda x:(x[0][0],x[1][0])).map(lambda x:((x[0][0],x[1][0]),(x[0][1],x[1][1])))

word_pair_similarity = wod_pairs.map(lambda x:(x[0],len(x[1][0] & x[1][1])/(len(x[1][0]|x[1][1])*1.0)))






