'''
ALS implementation


'''

##It takes ratings.csv as input and generate the initital value matrix 

import sys
import csv

users=[]
product=[]
rating_by_user={}

with open(sys.argv[1], "r") as ins:
    arrVal=[]
    for eachline in ins:
       array_row=eachline.split(',')
       product_id= array_row[1]
       user_id=  array_row[0]
       ratingVal=array_row[2]
       if not int(product_id) in product:
          product.append(int(product_id))
       if not int(user_id) in rating_by_user:
          users.append(int(user_id))
          rating_by_user[int(user_id)]={}
          rating_by_user[int(user_id)][int(product_id)]=float(ratingVal)
       else:
          rating_by_user[int(user_id)][int(product_id)]=float(ratingVal)


val_X = open(sys.argv[2], 'w')
val_Y = csv.writer(val_X)

print ('Number of customer: %d' %len(users))
print ('Number of products : %d' %len(product))

for user_id in users:
    users=[0.0]*len(product)
    for product_id in range(0, len(product)):
        if user_id in rating_by_user:
           if product[product_id] in rating_by_user[user_id]:
              try:
                  exist_productid=product[product_id]
                  users[product_id]=float(rating_by_user[user_id][exist_productid])
              except: 
                  continue;
    val_Y.writerows([users])
val_X.close() 
