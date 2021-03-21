import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
#Spliting the data
from sklearn.utils import shuffle
#For finding out keywords
from sklearn.feature_extraction.text import CountVectorizer

#For checking similarity of our note with the articles available
from sklearn.metrics.pairwise import cosine_similarity




app = Flask(__name__)
model = pickle.load(open('d:/Works/Modeling/mine_attempt/achar101.pkl', 'rb'))
tfidf = pickle.load(open('d:/Works/Modeling/mine_attempt/tf101.pkl', 'rb'))



#tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
#Reading each of the specifically formed dataset 

tech_csv=pd.read_csv('d:/machine learning/articles/categorisation/tech.csv')
politics_csv=pd.read_csv('d:/machine learning/articles/categorisation/politics.csv')
sports_csv=pd.read_csv('d:/machine learning/articles/categorisation/sport.csv')
biz_csv=pd.read_csv('d:/machine learning/articles/categorisation/biz.csv')
others_csv=pd.read_csv('d:/machine learning/articles/categorisation/others.csv')
target_csv='' 


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = request.form.values()
    print("features type",type(features)," and value -",features)
    features = list(features)
    for i ,value in enumerate(features):
        #features[i]=pre_process(value)
        features[i]= (value)
    cv = CountVectorizer(stop_words='english')
    word_count_vector=cv.fit_transform(features)

    '''doc=clean_str(features[0])
    
    cv=CountVectorizer(max_df=0.85, stop_words='english')
    word_count_vector = cv.fit_transform(features[0])'''

    from sklearn.feature_extraction.text import TfidfTransformer

    
    doc=features[0]
    #Generate tf-idf for the given document
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    tfidf_vector =tfidf_transformer.transform(cv.transform([doc]))
    #Sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tfidf_vector.tocoo())
    #Only needed once 
    #msg="Put some more important words and try for recommendation"
    feature_names = cv.get_feature_names()
    if len(feature_names) == 0:
        return render_template('index.html',message="Put some more important words and try for recommendation")
         
    

    #Extract only the top n: n here is 10
    keywords=extract_topn_from_vector(feature_names,sorted_items,3)
    qu=""
    for k in keywords:
        qu+=qu+k+" "
    import requests

    cn='in'
    secret='52c11279b478428daeaf4bfecc1c684a'
    # Define the endpoint
    url = 'https://newsapi.org/v2/everything?'
    # Specify the query and number of returns
    parameters = {
    'q': qu, # query phrase
    'pageSize': 100,
      # maximum is 100
    'apiKey': secret # your own API key
    
    }
    # Make the request
    response = requests.get(url, params=parameters)
    # Convert the response to JSON format and pretty print it
    response_json = response.json()
    print("Length of :",response_json['totalResults'])
    if response_json['totalResults'] == 0:
        return render_template('index.html',message="Content doesn't exist for keywords")
        
    from textblob import Word
    news=response_json
    ###################################################
    final_content=[]
    final_url =[]
    url_col =[]
    img=[]
    title=[]
    description=[]

    for y in news.keys():
	    if (y =='articles'):
		    art=news[y]
		#print(type(art[0]['description']))
		    for i in range(len(art)):
			    if(art[i]['description'] != None):
				    description.append(art[i]['description']+ art[i]['title'])
                    
			    else:
				    description.append(art[i]['description'])
                    


			    url_col.append(art[i]['url'])
			    img.append(art[i]['urlToImage'])
			    title.append(art[i]['title'])
    for index, value in enumerate(description):
        description[index] = ' '.join([Word(word) for word in clean_str(value).split()])



    for index, value in enumerate(title):
        title[index] = ' '.join([Word(word) for word in clean_str(value).split()])	
    dataframe = {'title':title,'url':url_col,'description':description ,'image_url':img}
    df = pd.DataFrame(dataframe)
    #print('writing csv file...')
    df.to_csv('./current.csv', index = False)

    cur_csv=pd.read_csv("./current.csv")
    
    
    text_features = tfidf.transform(features)
    prediction = model.predict(text_features)
    '''for pred in prediction:
        if(pred == 'tech'):
            target_csv = tech_csv
        if(pred == 'sport'):
            target_csv = sports_csv
        if(pred == 'business'):
            target_csv = biz_csv
        if(pred == 'politics'):
            target_csv = politics_csv'''

    # move articles to an array
    articles = cur_csv.description.values

    # move article web_urls to an array
    web_url = cur_csv.url.values
    img_url =cur_csv.image_url.values
    title =cur_csv.title.values

        # shuffle these two arrays 
    articles, web_url,img_url,title = shuffle(articles,web_url, img_url,title,random_state=4)
        
    n=10
    xtrain = articles[:]
    xtrain_urls = web_url[:]
    x_img_url=img_url[:]
    #tfidf.fit(xtrain)
    x_tfidf = tfidf.transform(xtrain)

    top_recommendations, recommended_urls, sorted_sim_scores,rec_img_url = top_articles(x_tfidf, xtrain, text_features,xtrain_urls,x_img_url,title,10)



    print(prediction[0]," classified")
    print("urls:\n",recommended_urls)
    print("top",top_recommendations)
    return render_template('index.html',sentence=request.form.get('content'),c=prediction[0],urls=zip(recommended_urls,rec_img_url,title))
   # return features


@app.route("/show")
def show(pred):
    return render_template('twindex.html',show_text=pred)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

#n = 5 is default values for number of similar articles to be found, if n= 7 is passed then 7 similar articles would be found
def top_articles(xtrain_tfidf, xtrain, texts,xtrain_urls,x_img_url,title,n=10):
    
    similarity_scores = xtrain_tfidf.dot(texts.toarray().T)
    #Calculating similarity between notes and articles and scores are stored
    
    sorted_indices = np.argsort(similarity_scores, axis = 0)[::-1]
    
    sorted_scores =similarity_scores[sorted_indices]
    
    #get n topmost similar articles
    top_rec = xtrain[sorted_indices[:n]]
    
    #get top n corresponding urls
    rec_urls = xtrain_urls[sorted_indices[:n]]
    rec_img_urls = x_img_url[sorted_indices[:n]]
    rec_title = title[sorted_indices[:n]]


    
    return top_rec, rec_urls,sorted_scores,rec_img_urls


def sort_coo(coo_matrix):
    tuples= zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x:(x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_name, sorted_items,topn=10):
    sorted_items =sorted_items[:topn]
    
    score_vals =[]
    feature_vals =[]
    
    #word index and corresponding tf-idf score
    for idx, score in sorted_items:
        #keep track of feature name and its corresponding score
        score_vals.append(round(score,3))
        feature_vals.append(feature_name[idx])
        
    #create a tuples of feature, score
    results={}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] =score_vals[idx]
    
    return results

def clean_str(string):
    '''
    tokenization, cleaning for dataset
    '''
    import re
    string=str(string)
    string = re.sub(r"\'s", "", string)

    string = re.sub(r"\'ve", "", string)

    string = re.sub(r"\'t", "", string) 
    
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'lls", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower() 

import re
def pre_process(text):
    # lowercase
    text = text.lower()
    
    #remove tags
    #text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", text)
    
    #Remove special cahracters and digits
    text=re.sub("(\\d|\\W)+",' ',text)
    
    return text 
    



if __name__ == "__main__":
    app.run(debug=True)
    ECHO is on.
