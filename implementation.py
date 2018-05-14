from tweet_dumper import get_all_tweets
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

#Lets create some variables that we can easily change to impact the project
account1 = "realDonaldTrump"
account2 = "HillaryClinton"

#Lets begin by fetching our data! This is fairly easy, just pass in the two account names and we get lists in this form:
#[tweet1,twee2,tweet3] where the tweets are the tweets from the acount. We want all the data in one list, so we append to the first list the second time we call get_all_tweets (we actually do this in the next section)
data = get_all_tweets(account1)
data2 = (get_all_tweets(account2))

#Lets also make a list of labels for the tweets. Our labels will be the account names. Though these labels are in a separate list, we want data[i] to be a tweet that was written by labels[i]. Basically keep the labels in the same order as the tweets.
labels = [account1]*len(data)
labels2 = [account2]*len(data2)
labels.extend(labels2)
data.extend(data2)

#So now we have our ordered data with labels. We now need to do 2 things: first jumble the data so its uniformly distributed. And second we now need to create a training set to make our classifier and a test set to determine the accuracy. For simplicity's sake, lets split our data and label lists into 4 new lists: trainingData, trainingLabels, testData, testLabels. For this example, we will use 75% of the data as train, and 25% as test. Thankfully a function exists for exactly this purpose!
trainingData, testData, trainingLabels, testLabels = train_test_split(data, labels, test_size=0.25)
print(trainingData)
#Yay! Now we have our data! But we should probably clean it up. Effectively, this is preparing the data to get features out of it. So since we're dealing with words, we want to do something called tokenizing. This takes our string which is a tweet and makes it a list of all of the words in the tweet. So instead of "This is a test", we have something like ["this","is","a","test"]. This step also does things like removing some punctuation, and is intelligent enough to keep places like New York as one token instead of two. There would typically be code here for what we are calling data preprocessing, but thankfully the function in the next step handles it for us.

#Finally we're at the point where we can start the training process. The first thing we do is extract features from our data. There are a lot of different features we can extract, but one of the simplest is bag of words. As you already know, bag of words is just the frequency of every word in every tweet. So if your dataset was [the quick fox][the red apple], you could represent each of these datapoints by their feature vectors. In our case, [the quick fox] would turn into [1 1 1 0 0] and [the red apple] would become [1 0 0 1 1]. This is a bit unclear so maybe it makes sense to look at it in this way. First we extract all words from our training data and build a vocabulary. In the above example, the vocabulary would be [the,quick,fox,red,apple]. Lets make these the titles of columns in our feature matrix (shown below). Then lets create the feature matrix by taking each datapoint and marking down the number of times we see each word. Our above example would turn into this:
#                   the     quick   fox     red     apple
#[the quick fox]    1       1       1       0       0
#[the red apple]    1       0       0       1       1

#Obviously we would have a lot more rows since we have a lot more datapoints as well as columns, but you get the gyst. The feature matrix is simply the number part of the thing above. Now frequencies work out well for generating features, but words like "the" and "a" will have way more occurences than "democracy" or "facist". But in reality, if we are comparing trump to clinton, words like democracy and facist are probably better at distinguishing the two authors. So thats where tf-idf comes in. It is simply a refinement to bag of words that doesn't use frequencies for the numbers. Instead it creates a weighted value that takes into accound the frequency of the word in the current document compared to the word in all the other documents. You can read bout this online to get an idea of what it is.

#Now, obviously we aren't going to calculate all this on our own. Theres already functions for this. This function below will both preprocess the data and then generate these feature vectors. Something to note, we only build our vocabulary on our training set. This creates a model that we can then use to generate features for the test set. But if the word "cuntmuffin" shows up in test data but not in training data, it would be cheating to have included it in our vocabulary that is used for the model. So since we're in the training stage, we just call this function on our training data.

#This vectorizer is our Feature extraction model. We need to train it with trainingData. The stop_words parameter just removes common words like the, a, an since they have little importance on authorship classification, yet make the data messy.
featureExtractor = TfidfVectorizer(stop_words='english')
#Lets train the model and have it output the feature matrix of the training data
trainingDataFeatures = featureExtractor.fit_transform(trainingData)

#Cool, so now we've turned our training data into a matrix that represents it, and we still have our training labels. Now we build a classifier!!! THeres a lot of different classifiers, one of the typical ones is SVM, another is logistic regression. We're using logistic regression below. It works just like the feature extractor and reducer.

classifier = LogisticRegression()
#Now we train the classifier. This requires both the trainingData and the trainingLabels (how else would it know how to classify things?)
classifier.fit(trainingDataFeatures,trainingLabels)


#FINALLY We've finished training. Now we have a classifier that will output one of two authors given an input of text. Remember that testData we have? We can feed that into our classifier, have it predict labels for it, compare those labels to the labels we already know (our testLabels) and see how well we're doing!

#First we need to generate features from the test data. We use transform, not fit_transform because we don't want to change our model, just get the output.
testDataFeatures = featureExtractor.transform(testData)

#And now lets test our classifier!
predictedResults = classifier.predict(testDataFeatures)

#This will show us how well we did (the closer the numbers are to 1 the better). Look up what precision and recall are to understand this chart.
print(classification_report(testLabels,predictedResults))
