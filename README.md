# Text classification
<br>The idea is understand the difference between Multinomial Naive Bayes(MNB) and Complement Naive Bayes(CNB), and the role of alpha as a parameter.
<br>(Note: Alpha is added as smoothing parameter(0 for no smoothing). The default value of alpha is 1.)
<br>I tested and compared four models: MNB(), MNB(alpha = 1.0e-10), CNB(), CNB(alpha = 1.0e-10), on a self-created small dataset with skewed data.
<br>
<br>Highlights
<br>CNB works with the idea of finding probability of not belonging to a class/label, whereas MNB finds the probability of presence of class/label. 
<br>It was noted that MNB() is biased due to the skewness of the data, same can be observed with CNB(alpha = 1.0e-10).
<br>
<br>Sample of Biased MNB() and CNB(alpha = 1.0e-10):-
<br>It can be noted the text "sun" is not in the dataset, implying the count per each label(object or action) is zero. Therefore it can be presence is either of the class with 50-50 possibility, but for MNB() and CNB(alpha = 1.0e-10) show that it is 90% likely to be present in class "object", which forms 180 out 200 datapoints.
<br>MNB(alpha = 1.0e-10) and CNB() performs most accurately. I found CNB() a more balanced algorithm.
<br>
<br><img src=result/image.jpg>
<br>
# Sentiment Analysis
<br>
<br> On a very small set of text, I trained the model(CNB()) which predicts if the sample text is a positive movie review, negative or neutral. This calucation is done on the basis of number of positive and negative keywords and declares if the text is "positive", "negative" or "neutral".  
