import re
import tensorflow
import numpy
import spacy
from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

nlp = spacy.load("en_core_web_md")
maximum_length = 40

model = tensorflow.keras.models.load_model("generated_models/F&N_intent_model.h5")
with open('generated_models/F&N_tokenizer.pickle', 'rb') as k:
    tokenizer_model = pickle.load(k)

labels = ['Nutritional_Information', 'Healthier_Alternatives', 'Debunking_Myths']

def bsi(word_list):
    list = [word.lower() for word in word_list]
    vectoriser = TfidVectorizer().fit(word_list)
    matrix = vectoriser.transform(list)
    return list, vectoriser, matrix

def best_match(vectoriser, matrix, list, user_input):
    user_vectoriser = vectoriser.transform(user_input.lower())
    sim = cosine_similarity(user_vectoriser, matrix)
    n = numpy.argmax(sim)
    if sim[0][n] > 0.3:
        return list[n]
    return None

def entity_classifciation(user_input: str):
    sequence = tokenizer_model.texts_to_sequences([user_input])
    padded = tensorflow.keras.preprocessing.sequence.pad_sequences(
        sequence, maxlen=maximum_length, padding="post"
    )
    prediction = model.predict(padded)
    return labels[numpy.argmax(prediction)]

def intent_extraction(user_input: str, merged_data):
    s = nlp(user_input.lower())
    food = list(merged_data["food"].dropna().unique())
    nutrient = list(merged_data["nutrient_name"].dropna().unique())

    food_vectoriser, food_matrix, food_list = bsi(food)
    nutrient_vectoriser, nutrient_matrix, nutrient_list = bsi(nutrient)

    s_food = [ent.text for ent in s.ents if ent.label_ == "FOOD"]
    if s_food:
        food = s_food[0]
    else:
        food = best_match(user_input, food_vectoriser, food_matrix, food_list)

    s_nutrient = [ent.text for ent in s.ents if ent.label_ == "NUTRIENT"]
    if s_nutrient:
        nutrient = s_nutrient[0]
    else:
        nutrient = best_match(user_input, nutrient_vectoriser, nutrient_matrix, nutrient_list)

    quantity = 100
    qty_match = re.search(r"(\d+)\s?(g|gram|grams|mg|ml)", user_input.lower())
    if qty_match:
        quantity = int(qty_match.group(1))

    return {"food": food, "nutrient": nutrient, "quantity": quantity}