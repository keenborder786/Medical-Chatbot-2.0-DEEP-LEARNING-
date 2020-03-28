# disease-detector-machine-learning

<h1> What is this</h1>
This is a Machine Learning Model which will be integrated with chatbot which will detect diease from the symtoms given by the user.More
details coming soon.

# How to use it?
1-Run Data_Creation.py which will create a New-Data-Set.csv which will be given as input to the Machine Learning Model.

2-Run Machine_Learning_Model.py after making sure New-Data-Set.csv in the Root Directory. This will train and test the model. The model which will be
trained will be Bernoulli Naive Bayes Model and will be tested on test data. After test,a BernoulliNB_Performance.png file will be created in the root dir.This
will be a confusion matrix which will reflect how many different dieases the model was able to detect.

3-Finally Run User-Testing.py ,this has to be developed since this is the script which will get user data from chatbot and give the data to the trained Machine-Learning Model.

4-Main Chatbot.py--Coming Soon.


##This Rep is in Progress-Need to Integrate Chatbot and will update Pipeline accordingly.
