'''
 ALS implementation

'''

Execution Steps
============================


Step 1: Run the below commnd to get the initial rating matrix.
        

        $ python init_matrix_eval.py ratings1.csv  init_matrix.csv

Step 2: To do Initial Setup
	
	$ sudo su cloudera
	$ hadoop fs -mkdir /user/cloudera/recsys /user/cloudera/recsys/input

Step 3. Put all the input files into the new input directory
	
        $ hadoop fs -put init_matrix.csv /user/cloudera/recsys/input

Step 4. Execute the source code

        $ spark-submit recmsys_als.py /user/cloudera/recsys/input/init_matrix.csv > outputals.txt

Step 5. Delete the input file to place different input file

        $ hadoop fs -rm -r /user/cloudera/recsys
	

OUTPUT:-

The product predicted for user_id 1: for product_id 215: Predicted rating is 0.150100 
The product predicted for user_id 2: for product_id 147: Predicted rating is 3.393534 
The product predicted for user_id 3: for product_id 403: Predicted rating is 1.584582 
The product predicted for user_id 4: for product_id 102: Predicted rating is 2.584855 
The product predicted for user_id 5: for product_id 100: Predicted rating is 2.070448 
The product predicted for user_id 6: for product_id 481: Predicted rating is 0.682331 
The product predicted for user_id 7: for product_id 70: Predicted rating is 2.094582 
The product predicted for user_id 8: for product_id 355: Predicted rating is 2.889686 
The product predicted for user_id 9: for product_id 122: Predicted rating is 1.441237 
The product predicted for user_id 10: for product_id 144: Predicted rating is 1.612323 


RMSE value after each iterations:  [ 0.42346259  0.37553312  0.36927681  0.36704164  0.3661882   0.36581921
  0.36562869  0.36551741  0.36544669  0.36539874]
Avg rmse----  0.372931308576

