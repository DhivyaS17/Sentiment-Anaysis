import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from prettytable import PrettyTable

# Emotion detection using our model 
def Detect_emotion(sentence):
 # Prepping the given data
 lower_case=sentence.lower()

 punctuation=string.punctuation
 #tokenizing the words
 tokenized_words=word_tokenize(lower_case,"english")
 #remove stopwords(words which doesn't influence the sentiment) and punctuations
 cleaned_words=[]
 for word in tokenized_words:
     if(word not in (stopwords.words('english') and punctuation)):
         cleaned_words.append(word)
 
 # Importing postive and negative word dataset
 positive=[]
 negative=[]
 # getting positive dataset
 with open('pos-dataset.txt','r') as file1:
    for line in file1:
        new_line=line.replace("\n","").strip()
        positive.append(new_line)
 
 # getting negative dataset
 with open('neg-dataset.txt','r') as file2:
    for line in file2:
        new_line=line.replace("\n","").strip()
        negative.append(new_line)
 
 # Calculation of polarity of the sentence  
 pos_count,neg_count,neu_count=0,0,0
 total_words=len(cleaned_words)
 for word in cleaned_words:
     if(word in positive):
         pos_count+=1
     elif(word in negative):
         neg_count+=1
     else:
         neu_count+=1
 
 # polarity
 pos_pol,neg_pol,neu_pol=0.0,0.0,0.0
 if(pos_count):
  pos_pol=(pos_count-neg_count)/(neg_count+1)
 if(neg_count):
  neg_pol=(neg_count-pos_count)/(pos_count+1)
 if(neu_count):
  neu_pol=neu_count/total_words
 
 # deciding the polarity of the sentence
 if(pos_pol>neg_pol and pos_pol>neu_pol):
     return 1
 elif(neg_pol>neu_pol):
     return -1
 else:
     return 0

# Emotion detection using TEXTBLOB
def Emotion_dectector_textblob(sentence):
    senti=TextBlob(sentence)
    sent_polarity=senti.sentiment.polarity
    if(sent_polarity>0):
        return 1
    elif(sent_polarity<0):
        return -1
    else:
        return 0

# Emotion dectection using VADER
def Emotion_dectector_vader(sentence):
    sentiment_analyser=SentimentIntensityAnalyzer()
    sent_polarity=sentiment_analyser.polarity_scores(sentence)
    pos_polarity=sent_polarity['pos']
    neg_polarity=sent_polarity['neg']
    neu_polarity=sent_polarity['neu']
    if(pos_polarity>neg_polarity and pos_polarity>neu_polarity):
        return 1
    elif(neg_polarity>neu_polarity):
        return -1
    else:
        return 0

# Traing the model
def train_model():
    
    file=open("doc1.txt").readlines()
    train_dataset=[]
    train_res=[]
    model_res=[]
    model_res_textblob=[]
    model_res_vader=[]
    
    # reading the file to get dataset
    for lines in file:
        line,polarity=lines.strip().split('#')
        train_dataset.append(line)
        train_res.append(int(polarity))
    
    # classifying the sentence and the polarity from the dataset
    for data in train_dataset:
        res1=Detect_emotion(data)
        res2=Emotion_dectector_textblob(data)
        res3=Emotion_dectector_vader(data)
        model_res.append(res1)
        model_res_textblob.append(res2)
        model_res_vader.append(res3)
    return train_res,model_res,model_res_textblob,model_res_vader

# Calcute accuracy
def Accuracy(train_res,model_res):
    total_outcomes=len(train_res)
    correct_outcome=0
    # checking whether the model outcome matches with the acutal outcome 
    for outcome,outcome1 in zip(train_res,model_res):
        if(outcome==outcome1):
            correct_outcome+=1
    
    accuracy_percentage=(correct_outcome/total_outcomes)*100
    return accuracy_percentage

# Comparing the models 
def comparing_model():
    #traing all the 3 models
    train_res,model_res,model_res_textblob,model_res_vader=train_model()
    
    #calculate the accuracy of each models
    acc_model=Accuracy(train_res,model_res)
    acc_textblob=Accuracy(train_res,model_res_textblob)
    acc_vader=Accuracy(train_res,model_res_vader)
    
    # tabulate the result
    mytable=PrettyTable()
    mytable.field_names=['Models','Accuracy in %']
    mytable.add_row(['Emotion Detector',acc_model])
    mytable.add_row(['Textblob',acc_textblob])
    mytable.add_row(['Vader',acc_vader])
    print(mytable)
    

# MAIN PROGRAM
print("----------------------------------------------------------------")
print("                         EMOTION DETECTOR                       ")
print("----------------------------------------------------------------\n")
ch='y'
while(ch=='y' or ch=='Y'):
    user_data=input("Input:")
    res=Detect_emotion(user_data)
    print("The sentiment of the given sentence: ",end="")
    if(res==1):
      print("Positive")
    elif(res==-1):
      print("Negative")
    else:
      print("Neutral")
    ch=input("Would you like to contiue?(y/n)")
    print("\n")
comparing_model()