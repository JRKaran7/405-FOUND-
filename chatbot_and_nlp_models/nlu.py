import tensorflow 
import numpy
import re
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas

natural_language_processing = spacy.load("en_core_web_sm")

max_len = 50

loaded_model = tensorflow.keras.models.load_model("generated_models/F&N_intent_model.h5")
with open('generated_models/F&N_tokenizer.pickle', 'rb') as k:
    t_model = pickle.load(k)
    
with open('generated_models/F&N_labels.pickle', 'rb') as f:
    label = pickle.load(f)

def best_match(vectorizer, matrix, words, user_input):
    if not words or vectorizer is None or matrix is None:
        return None
    
    vec = vectorizer.transform([user_input.lower()])
    sim = cosine_similarity(vec, matrix)
    n = numpy.argmax(sim)
    
    if sim[0][n] >= 0.3:
        return words[n]
    return None

def classifying_intents(text):
    if loaded_model is None or t_model is None:
        return default_intent_fallback(text)
    sequence = t_model.texts_to_sequences([text])
    padded = tensorflow.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_len, padding="post")
    pred = loaded_model.predict(padded, verbose=0)
    intent_idx = numpy.argmax(pred)
    confidence = numpy.max(pred)
    intent = label[intent_idx]
        
    if confidence < 0.3:
        fallback_intent = default_intent_fallback(text)
        if fallback_intent != "nutri_data": 
            return fallback_intent
        
    return intent
    
def default_intent_fallback(text):
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['hello', 'hi', 'hey', 'good evening', 'good morning', 'yo', 'greetings']):
        return "hello"
    
    if any(word in text_lower for word in ['thanks', 'thank', 'cheers', 'appreciate']):
        return "thank you"
    
    if any(word in text_lower for word in ['bye', 'see you', 'goodbye', 'farewell', 'later']):
        return "bye"
    
    if any(word in text_lower for word in ['recipe', 'cook', 'meal idea', 'breakfast idea', 'dinner idea',
                                            'lunch idea', 'what can i make', 'meal prep', 'what to cook']):
        return "recipes"
    
    if any(phrase in text_lower for phrase in ['alternative', 'instead of', 'replace', 'substitute', 
                                                 'healthier option', 'better than', 'foods high in',
                                                 'sources of', 'rich in', 'good for']):
        return "alternative_food"

    if any(phrase in text_lower for phrase in ['myth', 'is it true', 'should i cut', 'does eating', 'detox tea', 'do carbs', 'does lemon', 
                                                'make you fat', 'burn fat', 'is all fat bad', 'after 9pm', 'avoid']):
        return "debunking_myth"
    
    if any(phrase in text_lower for phrase in ['i want to', 'how to', 'build muscle', 'what should i eat', 'help me',
                                                'lose weight', 'gain muscle', 'get lean', 'nutrition plan', 'diet for', 
                                                'lose fat']):
        return "personal_recommendation"
    
    if any(phrase in text_lower for phrase in ['i ate', 'i had', 'just ate', 'just had', 'log meal',
                                                 'add to log', 'track', 'record', 'i consumed']):
        return "diet_log"
    
    if any(phrase in text_lower for phrase in ['what is', 'explain', 'define', 'what does', 'tell me about',
                                                 'how does', 'what are', 'what do']) and \
       any(word in text_lower for word in ['protein', 'fat', 'vitamin', 'cholestrol', 'carb', 'macro',
                                            'calorie', 'nutrient', 'micro', 'fiber', 'mineral']):
        return "clarify_nutrients"
    
    return "nutri_data"

def bsi(words):
    words = [str(w).lower() for w in words if pandas.notna(w)]
    if not words:
        return [], None, None
    
    vec = TfidfVectorizer().fit(words)
    matrix = vec.transform(words)
    return words, vec, matrix


def extracting_entities(text, merged_data):
    if merged_data.empty:
        return {"food": None, "nutrient": None, "quantity": 100}
    
    food_list = merged_data["food"].dropna().unique().tolist()
    nutrient_list = merged_data["nutrient_name"].dropna().unique().tolist()
    
    food_words, food_vec, food_matrix = bsi(food_list)
    nutrient_words, nut_vec, nut_matrix = bsi(nutrient_list)
    
    food_found = best_match(food_vec, food_matrix, food_words, text)
    nutrient_found = best_match(nut_vec, nut_matrix, nutrient_words, text)
    
    if not found_food:
        found_food = keyword_match_food(text, food_list)
    
    if not found_nutrient:
        found_nutrient = match_nutrient(text)
    
    qty = 100
    match = re.search(r"(\d+)\s?(g|grams|gram|ml|mg|ounce|oz|ounces)", text.lower())
    if match:
        qty = int(match.group(1))
    
    if natural_language_processing:
        doc = natural_language_processing(text.lower())
        
    return {
        "food": food_found,
        "quantity": qty,
        "nutrient": nutrient_found
    }

def keyword_match_food(text, food_list):
    text_lower = text.lower()
    
    commonfood = {
        'chicken': ['chicken'],
        'rice': ['rice'],
        'egg': ['egg', 'eggs'],
        'milk': ['milk'],
        'spinach': ['spinach'],
        'pasta': ['pasta'],
        'apple': ['apple', 'apples'],
        'banana': ['banana', 'bananas'],
        'carrot': ['carrot'],
        'potato': ['potato'],
        'yogurt': ['yogurt', 'yoghurt'],
        'oats': ['oats', 'oatmeal'],
        'cheese': ['cheese'],
        'orange': ['orange'],
        'nuts': ['nuts', 'almond', 'walnut'],
        'tomato': ['tomato'],
        'bread': ['bread'],
        'beef': ['beef'],
        'fish': ['fish', 'salmon', 'tuna'],
        'broccoli': ['broccoli'],
        'avocado': ['avocado']
        
    }
    
    for food_key, keywords in commonfood.items():
        for keyword in keywords:
            if keyword in text_lower:
                for food in food_list:
                    if food_key in food.lower() or keyword in food.lower():
                        return food
    
    return None

def match_nutrient(text):
    text_lower = text.lower()
    
    nutrient_keywords = {
        'Energy': ['calorie', 'calories', 'kcal', 'energy'],
        'Protein': ['protein', 'proteins'],
        'Total lipid (fat)': ['fat', 'fats', 'lipid'],
        'Carbohydrate': ['carb', 'carbs', 'carbohydrate', 'carbohydrates'],
        'Fiber': ['fiber', 'fibre', 'dietary fiber'],
        'Sodium': ['sodium', 'salt'],
        'Cholesterol': ['cholesterol'],
        'Sugars': ['sugar', 'sugars'],
        'Calcium': ['calcium'],
        'Vitamin C': ['vitamin c', 'ascorbic'],
        'Vitamin A': ['vitamin a'],
        'Iron': ['iron']
    }
    
    for nutrient, keywords in nutrient_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                return nutrient
    
    return None
