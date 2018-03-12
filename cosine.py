

import sys
import numpy as np
import math
from pyspark import SparkContext


#Parsing the Rating File
def RatingForProducts(line):
    list = line.replace("\n", "").split(",") #Combine all the rows of the excel files seperate them by a ,
    try:
        return int(list[0]), int(list[1]), float(list[2]) #Split on the basis of the , and get individual lists of # userIDs, productID and ratings
    except ValueError:    #Exception Handling for if there is any problem while assigning value
        pass


		
		
#Parsing the Movie file
def ProductList(line):
    list = line.replace("\n", "").split(",") #Combine all the rows of the excel files seperate them by a ,   
    try:
        return int(list[0]), list[1]	#Split on the basis of the , and get individual lists of
										#product IDs and products
    except ValueError:	#Exception Handling for if there is any problem while assigning value
        pass


		

#Calculating the average rating of users 
def AverageRatingForUser(usingRating):
    User_ID = usingRating[0]	#collecting the userID of the user
    Rating_Sum = 0.0	#Initialising to zero
    Rating_Count = len(usingRating[1])	#Finding out how many ratings the user has provided
    if Rating_Count == 0:
        return (User_ID, 0.0)	#If not even a single rating given by user, then average rating 0.
    for everyrating in usingRating[1]:
        Rating_Sum += everyrating[1]	#Getting the sum of all the ratings of the particular UserID
    return (User_ID, 1.0 * Rating_Sum / Rating_Count) #Calculating average - Sum of ratings/number of ratings

	


#Getting the user ratings in the form of a list
def UserProduct(usingRating):
    User_ID = usingRating[0]	#Getting userID
    ListOfProducts = [item[0] for item in usingRating[1]]	#Getting the product for that user
    ratingList = [item[1] for item in usingRating[1]]	#Getting the ratings provided
    return (User_ID, (ListOfProducts, ratingList))	#Final output is the user ID along with the list of products and ratings




#Broadcasting the user average rating RDD
def BroadcastUserAndAverageRating(sContext, UTrain_RDD):
    UserRatingAverage_List = UTrain_RDD.map(lambda x: AverageRatingForUser(x)).collect() 
#Storing the output in a list using .collect() to get output from already defined function
    UserRatingAverage_Dict = {}
    for (user, avgscore) in UserRatingAverage_List:
        UserRatingAverage_Dict[user] = avgscore	#Creating a dictionary for user such that the particular average score is corresponding to the userID
	UserAverageRating_Broadcast = sContext.broadcast(UserRatingAverage_Dict) #Broadcasting the dictionary so that it is available to all the functions and thus saves memory
    return UserAverageRating_Broadcast



	
#Broadcasting the User Movie List Dictionary 
def BrodcastUserAndProductList(sContext, UTrain_RDD):
    UserProductHistoryList = UTrain_RDD.map(lambda x: UserProduct(x)).collect()
    UserProductHistoryDict = {}
    for (user, productcollection) in UserProductHistoryList:  #For every collection of products for a particular user, that is a tuple
        UserProductHistoryDict[user] = productcollection #assign the tuple to that userID index in tye dictionary
    uMHistBC = sContext.broadcast(UserProductHistoryDict)	#Broadcasting the dictionary
    return uMHistBC




""""
ConstructRating - This function gets two user IDs and their products and returns the common products between the two users
Input - (userID,[products,ratings]) 
Output - ((user1,user2),[(commonproductrating_user1,commonproductrating_user2)])
"""
def ConstructRating(tuple1, tuple2):	
    user1, user2 = tuple1[0], tuple2[0]	#storing the userIDs
    firstuserproductlist = sorted(tuple1[1])	#Storing the product lists for the two users
    seconduserproductlist = sorted(tuple2[1])
    ratingpair = []
    i, j = 0, 0
    while i < len(firstuserproductlist) and j < len(seconduserproductlist): #iterating between the two product lists
        if firstuserproductlist[i][0] < seconduserproductlist[j][0]:
            i += 1															#keep moving until both i and j are the same index
        elif firstuserproductlist[i][0] == seconduserproductlist[j][0]:
            ratingpair.append((firstuserproductlist[i][1], seconduserproductlist[j][1])) #append the ratings for the common products. Both index contains same product	
	    i += 1
            j += 1
        else:
            j += 1
    return ((user1, user2), ratingpair)



	
""" 
Cosine_Similarity - This function calculates the cosine similarity for two user pairs.
The input for this function is the output of ConstructRating function. 
"""
def Cosine_Similarity(tup):
    dotproduct = 0.0
    square1, square2, c = 0.0, 0.0, 0
    for our_rating_pair in tup[1]:
        dotproduct += our_rating_pair[0] * our_rating_pair[1] #numerator for Cosine Similarity
        square1 += (our_rating_pair[0]) ** 2 #unpacking the tuple and then multiplying with self to get square
        square2 += (our_rating_pair[1]) ** 2	
        c += 1 #the count of common products
    denominator = math.sqrt(square1) * math.sqrt(square2)
    cos_simi = (dotproduct / denominator) if denominator else 0.0
    return (tup[0], (cos_simi, c)) #output the (userIDs, (cosine similarity, count of common products))




"""
In this function the input is from the cosine similarity function. It is taken in and we group the records by userID. 
Now we will get the output(user1,(all the users, corresponding cos_simi, corresponding common product match count)....)
"""
def User_GroupBy(record):
    return [(record[0][0], (record[0][1], record[1][0], record[1][1])), 
            (record[0][1], (record[0][0], record[1][0], record[1][1]))]



			
"""
SimilarUser_pull - this function takes the userID, cosine cos_simi and the number of neighbors we want as input
and returns the corresponding number of neighbors.
"""
def SimilarUser_pull(user, records, numK = 200):
    llist = sorted(records, key=lambda x: x[1], reverse=True) #take in x and return the next value of x (neighbour)
    llist = [x for x in llist if x[2] > 9]	#filter out those whose c is small
    return (user, llist[:numK])



	
"""
UserNeighbourBroadcast - Broadcasting the userNeighborRDD
"""
def UserNeighbourBroadcast(sContext, uNeighborRDD):
    userNeighborList = uNeighborRDD.collect()
    userNeighborDict = {}
    for user, simrecords in userNeighborList:
        userNeighborDict[user] = simrecords			#making a dicionary of user and corresponding neighbourlist
    uNeighborBC = sContext.broadcast(userNeighborDict)	#broadcasting it
    return uNeighborBC



	
"""
Calculating Error. Taking in actual and predicted RDDs as input and calculating RMSE and MSE.
"""

def CalculatingError(predictedRDD, actualRDD):
    #initial transformation and joining the RDD
    predictedReformattedRDD = predictedRDD.map(lambda rec: ((rec[0], rec[1]), rec[2])) #Getting the necessary columns for error calculation
    actualReformattedRDD = actualRDD.map(lambda rec: ((rec[0], rec[1]), rec[2]))
    joinedRDD = predictedReformattedRDD.join(actualReformattedRDD) #Joining the necessary columns for both predictedRDD and actual RDD together
    #Calculating the Errors
    squaredErrorsRDD = joinedRDD.map(lambda x: (x[1][0] - x[1][1])*(x[1][0] - x[1][1]))
    totalSquareError = squaredErrorsRDD.reduce(lambda v1, v2: v1 + v2)
    numRatings = squaredErrorsRDD.count()	#ratings count
    return (math.sqrt(float(totalSquareError) / numRatings))



	
""" 
Neighborhood_size - this function is used to invoke the previous error calculation function and depending on the max number
of neighbors and step size, it iteratess and finds the corresponding error for for all those number of pairs.
"""
def Neighborhood_size(val4PredictRDD, validate_RDD, userNeighborDict, UserProductHistoryDict, UserRatingAverage_Dict, K_Range):
    errors = [0] * len(K_Range)
    err= 0
    for numK in K_Range:
        predictedRatingsRDD = val4PredictRDD.map(
            lambda x: Prediction(x, userNeighborDict, UserProductHistoryDict, UserRatingAverage_Dict, numK)).cache()
        errors[err] = CalculatingError(predictedRatingsRDD, validate_RDD)
        err+= 1
    return errors



	
""" 
Prediction - this function predicts the rating.
it takes the following as input - the validationRDD, the neighbor dict whic has the user cosine similarity
and corresponding count and Ids, average rating of each user and the number of neighbors
"""
def Prediction(tup, neighborDict, userproducthistorydicti, avgDict, topK):
   user, product = tup[0], tup[1] #getting the userID and product
   avgrate = avgDict.get(user, 0.0)
   c = 0
   simsum = 0.0 #Sum of cos_simi
   WeightedRating_Sum = 0.0
   neighbors = neighborDict.get(user, None)
   if neighbors:
       for record in neighbors:
           if c >= topK:	#if count is more than the number of neighbours
               break
           c += 1
           mrlistpair = userproducthistorydicti.get(record[0])
           if mrlistpair is None:
               continue
           index = -1
           try:
               index = mrlistpair[0].index(product)
           except ValueError:# if error, then this neighbor hasn't rated the product yet
               continue
           if index != -1:
               neighborAvg = avgDict.get(record[0], 0.0)
               simsum += abs(record[1])
               WeightedRating_Sum += (mrlistpair[1][index] - neighborAvg) * record[1]
   predRating = (avgrate + WeightedRating_Sum / simsum) if simsum else avgrate
   return (user, product, predRating)
from collections import defaultdict




"""
Final_recommend- this function takes the following inputs
ID of the user who we need recommendation for,
the RDD containg the userid and corresponding cosine cos_simi
the list of users adn every movie they have rated
maintain two dicts, one for cos_simi sum, one for weighted rating sum
for every neighbor of a user, get his rated items which hasn't been rated by current user
then for each movie, sum the weighted rating in the whole neighborhood 
and sum the cos_simi of users who rated the movie
iterate and sort
"""
def Final_recommend(user, neighbors, userproducthistorydicti, topK = 200, nRec = 5):
    simSumDict = defaultdict(float)# cos_simi sum
    weightedSumDict = defaultdict(float)# weighted rating sum
    ProductIDUserRated = userproducthistorydicti.get(user, [])
    for (neighbor, simScore, numCommonRating) in neighbors[:topK]:
        mrlistpair = userproducthistorydicti.get(neighbor)
        if mrlistpair:
            for index in range(0, len(mrlistpair[0])):
                productID = mrlistpair[0][index]
                simSumDict[productID] += simScore
                weightedSumDict[productID] += simScore * mrlistpair[1][index]# sim * rating
    candidates = [(mID, 1.0 * wsum / simSumDict[mID]) for (mID, wsum) in weightedSumDict.items()]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return (user, candidates[:nRec])



	
#BroadcastProductNameDictt- This function broadcasts the movie RDD
def BroadcastProductNameDictt(sContext, movRDD):
    ProductNameList = movRDD.collect()
    ProductNamesDictionary = {}
    for (productID, pname) in ProductNameList:
        ProductNamesDictionary[productID] = pname
    mNameDictBC = sc.broadcast(ProductNamesDictionary)
    return mNameDictBC




def ProductRecorddName(user, records, pnamedict):
    nlist = []
    for record in records:
        nlist.append(pnamedict[record[0]])
    return (user, nlist)



#Spark program execution start
if __name__ == "__main__":
    
     #If input file is not provided, show error
    if len(sys.argv) !=3:
        print >> sys.stderr, "Usage: linreg <datafile> "
        exit(-1)
    #Initiatlize spark context
    sc = SparkContext(appName="Product Recommendation System")
#Reading the Data
ratingrawdata = sc.textFile(sys.argv[1])
productrawdata = sc.textFile(sys.argv[2])




#We remove the header from the rating data
ratingHeader = ratingrawdata.first()
ratingrawdata = ratingrawdata.filter(lambda x: x != ratingHeader)
#We remove the header from the product data
movieHeader = productrawdata.first()
productrawdata = productrawdata.filter(lambda x: x != movieHeader)




# Moving the rating and movies data to ratingRDD and ProductRDD    
ratingRDD = ratingrawdata.map(RatingForProducts).cache()
productRDD = productrawdata.map(ProductList).cache()
#We use cache for faster implementation




#Getting the count
ratings_Count = ratingRDD.count()
ProductCount = productRDD.count()




# Creating train and validate
Train_RDD, test_RDD = ratingRDD.randomSplit([7,3])
PredictionRDD = test_RDD.map(lambda x: (x[0], x[1])) #not including the target variable rating
TrainUserRating_RDD = Train_RDD.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().cache().mapValues(list)
																	




#here we take the cartesian Product so as to get a Matrix so that later we can filter this matrix to get user pairs
GetBroadcastUserRatingAverage = BroadcastUserAndAverageRating(sc, TrainUserRating_RDD)
GetBroadcastUserProductList = BrodcastUserAndProductList(sc, TrainUserRating_RDD)
CartesianRDD = TrainUserRating_RDD.cartesian(TrainUserRating_RDD)






#taking all the values below the diagonal so as to get user pairs
UserPairOne = CartesianRDD.filter(lambda x: x[0] < x[1])




#invoking the cosine function and other RDD transformation functions
UserPairActual = UserPairOne.map(
        lambda x: ConstructRating(x[0], x[1]))

SimiliarUserRDD = UserPairActual.map(lambda x: Cosine_Similarity(x))

SimiliarUserGroupRDD = SimiliarUserRDD.flatMap(lambda x: User_GroupBy(x)).groupByKey()

UserNeighborhood_RDD = SimiliarUserGroupRDD.map(lambda x: SimilarUser_pull(x[0], x[1], 200))




UserNeighborhood_BC = UserNeighbourBroadcast(sc, UserNeighborhood_RDD)
ErrorValue = [0]
#K_range is the starting number of neighbors and the ending number and the step size
K_Range = range(10, 210, 10)
e = 0
ErrorValue[e] = Neighborhood_size(PredictionRDD, test_RDD, UserNeighborhood_BC.value, 
GetBroadcastUserProductList.value, GetBroadcastUserRatingAverage.value, K_Range)
print(ErrorValue)




UserNeighborhood_RDD.map(lambda x: (x[1])).mapValues(list)
RecommendedProductForUserRDD = UserNeighborhood_RDD.map(lambda x: Final_recommend(x[0], x[1], GetBroadcastUserProductList.value))
ProductNameDictionary = BroadcastProductNameDictt(sc, productRDD)
RecProdForUserRDD = RecommendedProductForUserRDD.map(lambda x: ProductRecorddName(x[0], x[1], ProductNameDictionary.value))




#outputting Final Recommendations
print(RecProdForUserRDD.filter(lambda x: x[0]== 3).collect())
