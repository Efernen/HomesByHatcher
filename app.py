from flask import Flask, render_template, request, jsonify
import json
import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer
nltk.download('all')



# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

# Global variables for the assistant
classifier = None
vectorizer = None
responses = {}

# Training function for the virtual assistant
def train_assistant():
    global classifier, vectorizer, responses  # Access globally

    # Load intents data from the JSON file
    with open('intents.json') as file:
        data = json.load(file)

    lemmatizer = WordNetLemmatizer()
    training_sentences = []
    training_labels = []
    labels = []

    # Process intents data
    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])
        responses[intent['tag']] = intent['responses']
        labels.append(intent['tag'])

    # Preprocess training sentences
    def preprocess(text):
        tokens = nltk.word_tokenize(text)
        return ' '.join([lemmatizer.lemmatize(token.lower()) for token in tokens])

    training_sentences = [preprocess(sentence) for sentence in training_sentences]

    # Vectorize the training data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(training_sentences)
    y = training_labels

    # Train the classifier
    classifier = LogisticRegression()
    classifier.fit(X, y)

# Train the assistant on startup
train_assistant()

# Routes for rendering HTML pages
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/page1')
def page1():
    return render_template('page1.html')

@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/page3')
def page3():
    return render_template('page3.html')

@app.route('/page4')
def page4():
    return render_template('page4.html')

@app.route('/page5')
def page5():
    return render_template('page5.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Utility function to preprocess user input
def preprocess(text):
    tokens = nltk.word_tokenize(text)
    return ' '.join([nltk.WordNetLemmatizer().lemmatize(token.lower()) for token in tokens])

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    
    # Preprocess the user message
    processed_message = preprocess(user_message)
    
    # Vectorize the processed message
    X_test = vectorizer.transform([processed_message])
    
    # Predict the intent using the classifier
    predicted_tag = classifier.predict(X_test)[0]
    
    # Select a random response based on the predicted intent
    response = random.choice(responses[predicted_tag])
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)

