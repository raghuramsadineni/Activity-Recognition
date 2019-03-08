# Activity-Recognition
Activity Recognition - Stationary vs Walking

Summary:
1) Install the Accelerometer Analyzer app from play store.
2) Dataset preparation
	i) Keep the mobile on a platform and record the activity in the app and name the file accordingly(1-2 mins).
	ii) Hold the mobile and record the activity while walking and name the file accordingly (1-2 mins).
	iii) Repeat the aboce activities for about 15-20 times.
3) Use the data from the files and use it for training and testing:
	i) KNN
	ii) Decission tree
	iii) Neural networks
	iv) KMeans
	v)Random Forests
4) Record a new activity and use the models built to classify the type of activity.

Procedure:
1)Accelerometer app measures the attributes like:
	i)sensorspeed
	ii)units
	iii)gravity
	iv)Accuracy

How Android Accelerometer Work?
Android accelerometer usually measures the acceleration by force.How?
In simple words, Android Accelerometer senses how much a mass presses on something when a force acts on it.
This is something that almost everyone is familiar with. So, let’s go ahead and see how you can integrate Android accelerometer in your Android app to detect shake in an Android device

2)Download the files I have uploaded or create them in the same procedure as above.This will be processed in this Step.
First, we have a large amount of data we reduce them by taking mean for every 128 points(this doesn't infect bad values to data)
Download and execute cleaning.py file to perform this operation(all files must be placed in same Folder)

3)This creates new files named as "Fdata1.csv" and "Fdata.csv".
Fdata1.csv:-This is the file which is labelled with the class labels(training dataset).I'm going to use this data to train my model.
Fdata.csv:-This is the file which is used to determine the accuracy and precision of the model(testing dataset).
I have even uploaded the files for reference.

4)By now we have completed step-2 of the problem statement.Now we are ready to use that data to train the model.
I have uploaded all the files with different machine learning implementations mentioned above with respective file names.

Conclusion:- On applying different algorithms we get the accuracy compare the result and use the best fit model.We can also used to combine weaker algorithms to produce stronger one(I will be posting on this in future).On applying the algorithms we can Conclude that neural networks are better when compared to rest of the models.

Thats the end of this mini-project.Any doubts or queries feel free to contact me
email:sadineni.raghuram@gmail.com


Thank you.
