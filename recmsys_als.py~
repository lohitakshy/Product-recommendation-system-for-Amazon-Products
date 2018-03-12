'''
ALS implementation
'''

from pyspark import SparkContext
import numpy as np
import json
from numpy.random import rand
from numpy import matrix
import sys

#For calculating RMSE    
def get_rms_error(rating_mat, product_mat, user_mat):
    productUserTrans = product_mat * user_mat.T
    val_differ = rating_mat - productUserTrans
    val_differ_sq = (np.power(val_differ, 2)) / (product_mat_row * user_mat_row)
    return np.sqrt(np.sum(val_differ_sq))

#For Fixing the user matrix
def fix_user_mat(x):
    uuTrans = broad_user.value.T * broad_user.value
    for a in range(prop):
        uuTrans[a, a] = uuTrans[a,a] + lamda_val * num_col

    userTrans=broad_user.value.T   
    ratingTrans=broad_rating_mat.value[x,:].T    
    upd_user = userTrans * ratingTrans

    return np.linalg.solve(uuTrans, upd_user)
   

#For Fixing the product matrix
def fix_product_mat(x):
    u_rate_Trans = broad_rating_mat.value.T
    mmTrans = broad_product.value.T * broad_product.value
    for i in range(prop):
        mmTrans[i, i] = mmTrans[i,i] + lamda_val * num_row
    productTrans=broad_product.value.T    
    rateTrans=u_rate_Trans[x, :].T
    upd_product = productTrans * rateTrans
    
    return np.linalg.solve(mmTrans, upd_product)


if __name__ == "__main__":
    
    num_itr =  10   
    rms_val = np.zeros(num_itr)
    lamda_val = 0.001
    prop = 15
    sc = SparkContext(appName="recmsys_als")
   
# Getting the product data
    inputData = sc.textFile(sys.argv[1])
    inputLines = inputData.map(lambda line: line.split(","))
    
    #rating_mat = np.matrix(inputLines.collect()).astype('float')
    line_array = np.array(inputLines.collect()).astype('float')
    rating_mat = np.matrix(line_array)

    num_row,num_col = rating_mat.shape
    
    broad_rating_mat = sc.broadcast(rating_mat)
      
# To randomly initialize product matrix and user matrix
    product_mat = matrix(rand(num_row, prop))
    broad_product = sc.broadcast(product_mat)
    
    user_mat = matrix(rand(num_col, prop))
    broad_user = sc.broadcast(user_mat)
   
    product_mat_row,product_mat_col = product_mat.shape
    user_mat_row,user_mat_col = user_mat.shape

# iterating until product matrix and user matrix converges
    for i in range(0,num_itr):
    
        #Fixing the user matrix for finding the product matrix. 
        product_mat = sc.parallelize(range(product_mat_row)).map(fix_user_mat).collect()
        broad_product = sc.broadcast(matrix(np.array(product_mat)[:, :]))

        #Fixing the product matrix for finding the user matrix.
        user_mat = sc.parallelize(range(user_mat_row)).map(fix_product_mat).collect()
        broad_user = sc.broadcast(matrix(np.array(user_mat)[:, :]))

        rms_error_val = get_rms_error(rating_mat, matrix(np.array(product_mat)), matrix(np.array(user_mat)))
        rms_val[i] = rms_error_val
    fin_user_mat = np.array(user_mat).squeeze()
    fin_product_mat = np.array(product_mat).squeeze()
    
    fin_out = np.dot(fin_product_mat,fin_user_mat.T)

# For Initializing the weights matrix 
    weight_mat = np.zeros(shape=(num_row,num_col))
    for r in range(num_row):
        for j in range(num_col):
            if rating_mat[r,j]>= 0.5:
                weight_mat[r,j] = 1.0
            else:
                weight_mat[r,j] = 0.0
    
# subtract the rating that user has rated
    rate_max=5
    product_recom = np.argmax(fin_out - rate_max * weight_mat,axis =1)
    
# To Predict the product for each user
    for u in range(0, product_mat_row):
        r = product_recom.item(u)
        p = fin_out.item(u,r)
        print ('The product predicted for user_id %d: for product_id %d: Predicted rating is %f ' %(u+1,r+1,p) )
        
    print "RMSE value after each iterations: ",rms_val    
    print "Avg rmse---- ",np.mean(rms_val)
    sc.stop()
