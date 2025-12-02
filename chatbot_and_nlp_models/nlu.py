import re
import tensorflow
import numpy
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

nlp = spacy.load("en_core_web_sm")
maximum_length = 40

model = tensorflow.keras.models.load_model("generated_models/F&N_intent_model.h5")
with open('generated_models/F&N_tokenizer.pickle', 'rb') as k:
    tokenizer_model = pickle.load(k)

labels = ['Hello!', 'Bye!', 'Debunking_Myths']

def build_similarity_index(words):
    words_lower = [word.lower() for word in words]
    vectorizer = TfidfVectorizer().fit(words_lower)
    matrix = vectorizer.transform(words_lower)
    return words_lower, vectorizer, matrix

def best_match(vectorizer, matrix, words_lower, user_input: str):
    vec = vectorizer.transform([user_input.lower()])
    sim = cosine_similarity(vec, matrix)
    n = numpy.argmax(sim)
    if sim[0][n] < 0.5:
        return words_lower[n]
    return None

def intent_classification(user_input: str):
    sequence = tokenizer_model.texts_to_sequences([user_input])
    padded = tensorflow.keras.preprocessing.sequence.pad_sequences(
        sequence, maxlen=maximum_length, padding="post"
    )
    prediction = model.predict(padded)
    return labels[numpy.argmax(prediction)]

def entity_extraction(user_input: str, merged_data):
    doc = nlp(user_input.lower())
    food_list = list(merged_data["food"].dropna().unique())
    nutrient_list = list(merged_data["nutrient_name"].dropna().unique())

    food_words, food_vectorizer, food_matrix = build_similarity_index(food_list)
    nutrient_words, nutrient_vectorizer, nutrient_matrix = build_similarity_index(nutrient_list)

    food_entities = [ent.text for ent in doc.ents if ent.label_ == "FOOD"]
    if food_entities:
        food = food_entities[0]
    else:
        food = best_match(food_vectorizer, food_matrix, food_words, user_input)

    nutrient_entities = [ent.text for ent in doc.ents if ent.label_ == "NUTRIENT"]
    if nutrient_entities:
        nutrient = nutrient_entities[0]
    else:
        nutrient = best_match(nutrient_vectorizer, nutrient_matrix, nutrient_words, user_input)

    quantity = 100
    qty_match = re.search(r"(\d+)\s?(g|gram|grams|mg|ml)", user_input.lower())
    if qty_match:
        quantity = int(qty_match.group(1))

    return {"food": food, "nutrient": nutrient, "quantity": quantity}
