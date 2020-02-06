# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, send_from_directory
import gensim
import keras
import numpy as np
from keras.applications import vgg16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import Tree
from nltk.draw.tree import TreeView
import os
import random
from imageai.Detection import ObjectDetection
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from pmdarima import auto_arima
import re
from summarization import text_ranking
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pickle
from keras.models import load_model
from gensim.models import KeyedVectors
from flask import jsonify
from collections import Counter
import tensorflow as tf
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm

app = Flask(__name__, static_folder='statics', static_url_path='/statics')
app_root = os.path.dirname(os.path.abspath(__file__))
app.secret_key = 'dev'

global graph
graph = tf.get_default_graph()


##Load the Models
glove_fp = app_root + '/Model/w2v.6B.300d.kv'
wv_model = KeyedVectors.load(glove_fp, mmap='r')
vec_list = wv_model.vectors
vec_vocab = list(wv_model.wv.vocab.keys())
vec_list = np.transpose(vec_list) / np.linalg.norm(vec_list, axis=1)
cat = pickle.load(open(app_root + '/Model/categories_Twitter_Gender.pkl', 'rb'))
model = load_model(app_root + '/Model/trained_model_Twitter_Gender.h5')
#model._make_predict_function()
#entities_model = StanfordNERTagger(
    #'C:/Users/ChenYuan/Flasken/Model/stanford-ner-2018-10-16/classifiers/english.muc.7class.distsim.crf.ser.gz',
    #'C:/Users/ChenYuan/Flasken/Model/stanford-ner-2018-10-16/stanford-ner.jar', encoding='utf-8')
im_classify_model = vgg16.VGG16(weights='imagenet') 
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(app_root + "/Model/resnet50_coco_best_v2.0.1.h5")
detector.loadModel()
entities_model = en_core_web_sm.load()


##dataset for ARIMA Forecasting
cpi_list = [92.30, 97.72, 117.24, 128.91, 107.67, 104.57, 93.45, 93.24, 103.31, 113.31, 113.28, 101.57, 95.32,
            103.58, 112.95, 123.77, 112.98, 106.66, 88.67, 91.91, 98.44, 112.34, 111.48]
#cpi_date = ['2017-10', '2017-11', '2017-12', '2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06', '2018-07', '2018-08', 
            #'2018-09', '2018-10', '2018-11', '2018-12', '2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07', 
            #'2019-08']
ridership_list = [740000, 850000, 911000, 946000, 986000, 1047000, 1071000, 1081000, 1171000, 1270000, 1321000, 1408000,
                  1527000, 1698000, 1782000, 2069000, 2295000, 2525000, 2623000, 2762000, 2879000, 3095000]
#ridership_date = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
                  #2014, 2015, 2016]
house_index = [131.43, 132.68, 134.48, 134.78, 138.3, 146.7, 144.68, 135.5, 122.18, 106.88, 98.90, 85.15, 74.80,
               73.45, 76.70, 74.30, 70.00, 70.35, 76.48, 77.60, 74.18, 90.20, 96.23]
#house_index_date = [1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
                    #2014, 2015, 2016, 2017, 2018]
##Spacy Entities Tagging
spacy_tag = {'PERSON':'People, including fictional', 'NORP':'Nationalities or religious or political groups', 
             'FAC':'Buildings, airports, highways, bridges, etc', 'ORG':'Companies, agencies, institutions, etc', 
             'GPE':'Countries, cities, states', 'LOC':'Non-GPE locations, mountain ranges, bodies of water', 
             'PRODUCT':'Objects, vehicles, foods, etc. (Not services.)', 'EVENT':'Named hurricanes, battles, wars, sports events, etc',
             'WORK_OF_ART':'Titles of books, songs, etc', 'LAW':'Named documents made into laws', 'LANGUAGE':'Any named language', 
             'DATE':'Absolute or relative dates or periods', 'TIME':'Times smaller than a day', 'PERCENT':'Percentage, including ”%“', 
             'MONEY':'Monetary values, including unit', 'QUANTITY':'Measurements, as of weight or distance', 'ORDINAL':'“first”, “second”, etc', 
             'CARDINAL':'Numerals that do not fall under another type', ' ':''}


## Helper Functions
def similar(word, n):
    vec = wv_model[word]
    vec_norm = vec / np.linalg.norm(vec)
    similarity = np.dot(vec_norm, vec_list)
    similarity = Counter({voc: sim for voc,sim in zip(vec_vocab, similarity)}).most_common(n)
    return [w for w,s in similarity]

def image_classify(url):  
    original = load_img(url, target_size=(224, 224))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = vgg16.preprocess_input(image_batch.copy())
    with graph.as_default():
        predictions = im_classify_model.predict(processed_image)
        label = decode_predictions(predictions)
    return label

nltk.download('vader_lexicon')
def sentiment_analysis(sentence):
    sid = SentimentIntensityAnalyzer()
    sentiment_result = sid.polarity_scores(sentence)['compound']
    return sentiment_result

def syntax_analysis(sentence):
    syntax_model = StanfordCoreNLP(
    app_root + '/Model/stanford-corenlp-full-2018-10-05')
    tree = syntax_model.parse(sentence)
    syntax_model.close() # Do not forget to close! The backend server will consume a lot memery.
    t = Tree.fromstring(tree)
    random.randint(0, 999)
    TreeView(t)._cframe.print_to_file('output.ps')
    filename = 'output{}.png'.format(random.randint(0, 999))
    command = 'convert output.ps images/' + filename
    os.system(command)
    return filename

def object_detect(destination):
    execution_path = os.getcwd()
    with graph.as_default():
        processed_filename = 'object_detect{}.jpg'.format(random.randint(0, 999))
        detector.detectObjectsFromImage(input_image=destination,
                                        output_image_path=os.path.join(execution_path, 'images/' + processed_filename))
    return processed_filename

#def entities_analysis(text):
    #tokenized_text = word_tokenize(text)
    #classified_text_ = entities_model.tag(tokenized_text)
    #classified_text = []
    #for i in range(len(classified_text_)):
        #if classified_text_[i][1] != 'O':
            #classified_text.append(classified_text_[i])

    #return classified_text

def entities_analysis(text):
    doc = entities_model(text)
    entities_result = [(x.text, x.label_) for x in doc.ents]
    return entities_result

def arima_fit(x, n_periods=10):
    x_pre = [x[0]] + x[:-1]
    x_post = x[1:] + [x[-1]]

    quantiles = np.quantile(x, [0.05, 0.95])

    x_consider = []
    for pre, cur, pos in zip(x_pre, x, x_post):
        if cur < quantiles[0] or cur > quantiles[1]:
            print(pre, cur, pos)
            x_consider.append((pre+pos)/2)
        else:
            x_consider.append(cur)
            
    arima_model = auto_arima(x_consider, trace=True, error_action='ignore', suppress_warnings=True)
    arima_model.fit(x_consider)
    forecast = arima_model.predict(n_periods=10, return_conf_int=True, alpha=0.1)

    point_pred = forecast[0].tolist()
    lower_pred = forecast[1][:, 0].tolist()
    upper_pred = forecast[1][:, 1].tolist()
    return (point_pred, lower_pred, upper_pred)

def input_x(string):
    words = [w for w in re.findall('[a-z]*',string.lower()) if w != '']
    words_vec = [wv_model[w] for w in words if w in wv_model]
    if len(words_vec) == 0:
        print('Not enough words to process')
        return None
    else:
        return(np.array(words_vec))


## WebApp Routes
@app.route('/')
def home():
    image_name1 = 'Picture5.png'
    image_name2 = 'Picture6.png'
    image_name3 = 'Picture7.png'
    image_name4 = 'Picture8.png'
    return render_template('home.html', image_name1=image_name1, image_name2=image_name2, image_name3=image_name3, image_name4=image_name4)

@app.route('/<filename>')
def send_image_home(filename):
    return send_from_directory("images", filename)

@app.route('/nlp')
def nlp():
    return render_template('nlp.html')

@app.route('/nlp', methods=['POST'])
def nlp_demo():
    text = request.form['text']
    original_text = text
    classified_text = entities_analysis(text)
    null_list = [(' ', ' '), (' ', ' '), (' ', ' '), (' ', ' '), (' ', ' '), (' ', ' '), (' ', ' ')]
    classified_text.extend(null_list)
    en1 = classified_text[0][0]
    en2 = classified_text[1][0]
    en3 = classified_text[2][0]
    en4 = classified_text[3][0]
    en5 = classified_text[4][0]
    c1 = spacy_tag[classified_text[0][1]]
    c2 = spacy_tag[classified_text[1][1]]
    c3 = spacy_tag[classified_text[2][1]]
    c4 = spacy_tag[classified_text[3][1]]
    c5 = spacy_tag[classified_text[4][1]]
    sentiment = round(sentiment_analysis(text), 2)
    if sentiment >= 0:
        sentiment_result = '(' + str(sentiment) + ')' + '  Positive'
    else:
        sentiment_result = '(' + str(sentiment) + ')' + '  Negtive'
    try:
        filename = syntax_analysis(text)
        return render_template('nlp.html', original_text=original_text, image_name=filename, en1=en1, en2=en2, en3=en3,
                               en4=en4, en5=en5, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, sentiment_result=sentiment_result)
    except:
        return render_template('nlp.html', original_text=original_text, en1=en1, en2=en2, en3=en3,
                               en4=en4, en5=en5, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, sentiment_result=sentiment_result)

@app.route('/nlp/<filename>')
def send_syntax_img(filename):
    return send_from_directory("images", filename)

@app.route('/word2vec')
def word2vec():
    return render_template('word2vec.html')

@app.route('/word2vec', methods=["POST"])
def nearest():
    word = request.form['word']
    #top_n = wv_model.most_similar(positive=word, topn=6)
    try:
        top_n = similar(word, 6)
    except:
        top_n = ['None', 'None', 'None', 'None', 'None', 'None']
    return render_template("word2vec.html", 
        word1=top_n[0], word2=top_n[1],
        word3=top_n[2], word4=top_n[3],
        word5=top_n[4], word6=top_n[5], original_word = word)

@app.route('/word2vec-similarity', methods=["POST"])
def similarity():
    w1 = request.form['w1']
    w2 = request.form['w2']
    try:
        similarity = wv_model.similarity(w1, w2)
        similarity = round(similarity, 2)
    except:
        similarity = 'Wrong Input'
    return render_template("word2vec.html", similarity=similarity, original_word1=w1, original_word2=w2)

@app.route('/im_classify')
def im_classify():
    filename = 'background.png'
    return render_template('im_classify.html', image_name=filename)

@app.route("/im_classify", methods=["POST"])
def upload_im():
    target = os.path.join(app_root, 'images/')
    for upload in request.files.getlist("file"):
        filename = upload.filename
        destination = "/".join([target, filename])
        upload.save(destination)
        label = image_classify(destination)
    p1 = int(label[0][0][2] * 100)
    p2 = int(label[0][1][2] * 100)
    p3 = int(label[0][2][2] * 100)
    p4 = int(label[0][3][2] * 100)
    p5 = int(label[0][4][2] * 100)
    l1 = label[0][0][1]
    l2 = label[0][1][1]
    l3 = label[0][2][1]
    l4 = label[0][3][1]
    l5 = label[0][4][1]
    #keras.backend.clear_session()
    return render_template("im_classify.html", image_name=filename, label=label, p1=p1, p2=p2, p3=p3, p4=p4, p5=p5,
                           l1=l1, l2=l2, l3=l3, l4=l4, l5=l5)

@app.route('/im_classify/<filename>')
def send_image_classify(filename):
    return send_from_directory("images", filename)

@app.route('/summarization')
def summarization():
    return render_template('summarization.html')

@app.route('/summarization', methods=['POST'])
def extractive_sum():
    text = request.form['text']
    original_text = text
    try:
        ranked_sentence = text_ranking(text)
        score1 = round(ranked_sentence[0][0] * 100, 2)
        score2 = round(ranked_sentence[1][0] * 100, 2)
        score3 = round(ranked_sentence[2][0] * 100, 2)
        score4 = round(ranked_sentence[3][0] * 100, 2)
        score5 = round(ranked_sentence[4][0] * 100, 2)
        score6 = round(ranked_sentence[5][0] * 100, 2)
        score7 = round(ranked_sentence[6][0] * 100, 2)
        sen1 = ranked_sentence[0][1]
        sen2 = ranked_sentence[1][1]
        sen3 = ranked_sentence[2][1]
        sen4 = ranked_sentence[3][1]
        sen5 = ranked_sentence[4][1]
        sen6 = ranked_sentence[5][1]
        sen7 = ranked_sentence[6][1]
    except: 
        score1 = score2 = score3 = score4 = score5 = score6 = score7 = sen1 = sen2 = sen3 = sen4 = sen5 = sen6 = sen7 = ' '
        original_text = 'Wrong Input'
    return render_template('summarization.html', original_text=original_text, score1=score1, score2=score2, score3=score3,
                           score4=score4, score5=score5, score6=score6, score7=score7, sen1=sen1, sen2=sen2, sen3=sen3,
                           sen4=sen4, sen5=sen5, sen6=sen6, sen7=sen7)

@app.route('/object')
def object():
    original_filename = 'background.png'
    processed_filename = 'background.png'
    return render_template('object.html', original_filename=original_filename, processed_filename=processed_filename)

@app.route("/object", methods=["POST"])
def object_detection():
    target = os.path.join(app_root, 'images/')
    for upload in request.files.getlist("file"):
        original_filename = upload.filename
        destination = "/".join([target, original_filename])
        upload.save(destination)
    processed_filename = object_detect(destination)
    #keras.backend.clear_session()
    return render_template("object.html", original_filename=original_filename, processed_filename=processed_filename)

@app.route('/object/<original_filename>')
def send_image_original(original_filename):
    return send_from_directory("images", original_filename)

@app.route('/object/<processed_filename>')
def send_image_detected(processed_filename):
    return send_from_directory("images", processed_filename)

@app.route('/cpi', methods=['GET', 'POST'])
def cpi():
    provided_data = [{'x': i, 'y': num} for i, num in enumerate(cpi_list)]
    predicted_data = [{'x': 0, 'y': 0}]
    lower = [{'x': 0, 'y': 0}]
    upper = [{'x': 0, 'y': 0}]
    list2 = str(cpi_list)[1:-1]

    if request.method == 'POST':
        string_ = request.form['text_box']
        string = [s.strip() for s in string_.split(',')]

        string = [[i for i in re.findall('[0-9]*[.]?[0-9]*', s) if i != ''] for s in string]
        string = [s for s in string if len(s) != 0]

        if len(string) >= 12:
            numbers = [float(s[0]) for s in string]
            pred = arima_fit(numbers, 7)

            provided_data = [{'x': i, 'y': num} for i, num in enumerate(numbers)]
            base_x = len(provided_data)
            predicted_data = [{'x': i + base_x, 'y': num} for i, num in enumerate(pred[0])]
            lower = [{'x': i + base_x, 'y': num} for i, num in enumerate(pred[1])]
            upper = [{'x': i + base_x, 'y': num} for i, num in enumerate(pred[2])]

            return render_template('cpi.html',
                                   provided_data=provided_data, predicted_data=predicted_data, lower=lower, upper=upper,
                                   number=string_)
        else:
            return render_template('cpi.html',
                                   provided_data=provided_data, predicted_data=predicted_data, lower=lower, upper=upper,
                                   number="you must type more than 12 numbers.")

    return render_template('cpi.html',
                           provided_data=provided_data, predicted_data=predicted_data, lower=lower, upper=upper,
                           number=list2)

@app.route('/ridership', methods=['GET', 'POST'])
def ridership():
    provided_data = [{'x': i, 'y': num} for i, num in enumerate(ridership_list)]
    predicted_data = [{'x': 0, 'y': 0}]
    lower = [{'x': 0, 'y': 0}]
    upper = [{'x': 0, 'y': 0}]
    list2 = str(ridership_list)[1:-1]

    if request.method == 'POST':
        string_ = request.form['text_box']
        string = [s.strip() for s in string_.split(',')]

        string = [[i for i in re.findall('[0-9]*[.]?[0-9]*', s) if i != ''] for s in string]
        string = [s for s in string if len(s) != 0]

        if len(string) >= 12:
            numbers = [float(s[0]) for s in string]
            pred = arima_fit(numbers, 7)

            provided_data = [{'x': i, 'y': num} for i, num in enumerate(numbers)]
            base_x = len(provided_data)
            predicted_data = [{'x': i + base_x, 'y': num} for i, num in enumerate(pred[0])]
            lower = [{'x': i + base_x, 'y': num} for i, num in enumerate(pred[1])]
            upper = [{'x': i + base_x, 'y': num} for i, num in enumerate(pred[2])]

            return render_template('ridership.html',
                                   provided_data=provided_data, predicted_data=predicted_data, lower=lower, upper=upper,
                                   number=string_)
        else:
            return render_template('ridership.html',
                                   provided_data=provided_data, predicted_data=predicted_data, lower=lower, upper=upper,
                                   number=list2)

    return render_template('ridership.html',
                           provided_data=provided_data, predicted_data=predicted_data, lower=lower, upper=upper,
                           number=list2)

@app.route('/house_index', methods=['GET', 'POST'])
def house_index_():
    provided_data = [{'x': i, 'y': num} for i, num in enumerate(house_index)]
    predicted_data = [{'x': 0, 'y': 0}]
    lower = [{'x': 0, 'y': 0}]
    upper = [{'x': 0, 'y': 0}]
    list2 = str(house_index)[1:-1]

    if request.method == 'POST':
        string_ = request.form['text_box']
        string = [s.strip() for s in string_.split(',')]

        string = [[i for i in re.findall('[0-9]*[.]?[0-9]*', s) if i != ''] for s in string]
        string = [s for s in string if len(s) != 0]

        if len(string) >= 12:
            numbers = [float(s[0]) for s in string]
            pred = arima_fit(numbers, 7)

            provided_data = [{'x': i, 'y': num} for i, num in enumerate(numbers)]
            base_x = len(provided_data)
            predicted_data = [{'x': i + base_x, 'y': num} for i, num in enumerate(pred[0])]
            lower = [{'x': i + base_x, 'y': num} for i, num in enumerate(pred[1])]
            upper = [{'x': i + base_x, 'y': num} for i, num in enumerate(pred[2])]

            return render_template('house_index.html',
                                   provided_data=provided_data, predicted_data=predicted_data, lower=lower, upper=upper,
                                   number=string_)
        else:
            return render_template('house_index.html',
                                   provided_data=provided_data, predicted_data=predicted_data, lower=lower, upper=upper,
                                   number=list2)

    return render_template('house_index.html',
                           provided_data=provided_data, predicted_data=predicted_data, lower=lower, upper=upper,
                           number=list2)


@app.route('/text_classify', methods=['GET', 'POST'])
def text_categorization():
    labs = cat
    data = [0] * len(labs)
    color = ["#3e95cd"] * len(labs)

    if request.method == 'POST':
        string = request.form['text']
        transformed = input_x(string)

        if transformed is not None:
            with graph.as_default():
                probs = model.predict(np.array([transformed]))
            data = probs[0].tolist()
            data = [round(d, 3) for d in data]

            return render_template('text_classify.html',
                                   labs=labs, data=data, color=color, original_text=string)
        else:
            result = "No Prediction to be Made"
            return render_template('text_classify.html',
                                   labs=labs, data=data, color=color, original_text=string)

    else:
        return render_template('text_classify.html',
                               labs=labs, data=data, color=color)



if __name__ == '__main__':
    app.run(debug=False)