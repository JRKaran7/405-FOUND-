import tensorflow
import numpy
import re
import pickle
import pandas
from difflib import SequenceMatcher
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")
m_l = 50


loaded_model = tensorflow.keras.models.load_model("generated_models/F&N_intent_model.h5")
with open('generated_models/F&N_tokenizer.pickle', 'rb') as k:
    tokenizer_model_for_code = pickle.load(k)
with open('generated_models/F&N_labels.pickle', 'rb') as f:
    label = pickle.load(f)

nutrient_alias = {
    'calorie': 'Energy', 'calories': 'Energy', 'kcal': 'Energy',
    'carb': 'Carbohydrate', 'carbs': 'Carbohydrate',
    'fat': 'Total lipid (fat)', 'fats': 'Total lipid (fat)', 'lipid': 'Total lipid (fat)',
    'protein': 'Protein', 'proteins': 'Protein',
    'fiber': 'Fiber', 'fibre': 'Fiber',
    'sodium': 'Sodium', 'salt': 'Sodium',
    'sugar': 'Sugars', 'sugars': 'Sugars',
    'calcium': 'Calcium', 'iron': 'Iron',
    'vitamin c': 'Vitamin C', 'vitamin a': 'Vitamin A',
    'cholesterol': 'Cholesterol'
}

def intent_classification(text):
    if not text or not text.strip():
        return "hello", 0.0
    
    text_lower = text.lower()
    
    n_keywords = [
        'how much', 'how many', 'what', 'protein in', 'calories in', 'fat in',
        'carbs in', 'fiber in', 'nutrients in', 'nutrition', 'content',
        'amount of', 'quantity of', 'levels of'
    ]
    
    f_keywords = ['chicken', 'rice', 'egg', 'milk', 'spinach', 'pasta', 'apple', 
                     'banana', 'carrot', 'potato', 'yogurt', 'oats', 'cheese', 'orange',
                     'nuts', 'tomato', 'bread', 'beef', 'fish', 'broccoli', 'avocado']
    
    nutrient_query = any(kw in text_lower for kw in n_keywords)
    food_query = any(kw in text_lower for kw in f_keywords)
    
    if nutrient_query and food_query:
        return "nutri_data", 0.95
    if any(word in text_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good evening', 'howdy']):
        return "hello", 0.95
    if any(word in text_lower for word in ['bye', 'goodbye', 'see you', 'farewell']):
        return "bye", 0.95
    if any(word in text_lower for word in ['thanks', 'thank', 'appreciate', 'cheers']):
        return "thank you", 0.95
    if any(phrase in text_lower for phrase in ['recipe', 'cook', 'meal idea', 'meal prep', 'how do i make', 'what to cook', 'prepare']):
        return "recipes", 0.95
    if any(phrase in text_lower for phrase in ['alternative', 'instead of', 'replace', 'substitute', 'healthier', 'alternatives', 'foods high in', 'sources of', 'rich in', 'good for']):
        return "alternative_food", 0.95
    if any(phrase in text_lower for phrase in ['myth', 'make you fat', 'is it true', 'does eating', 'detox', 'after 9pm', 'burn fat', 'lemon water']):
        return "debunking_myth", 0.90
    if any(phrase in text_lower for phrase in ['i want to', 'how to', 'build muscle', 'lose weight', 'gain muscle', 'help me']):
        return "personal_recommendation", 0.90
    if any(phrase in text_lower for phrase in ['i ate', 'i had', 'just ate', 'log meal', 'track']):
        return "diet_log", 0.90
    
    c_triggers = ['what is', 'explain', 'define', 'what does', 'tell me about']
    c_nutrients = ['protein', 'fat', 'vitamin', 'carb', 'fiber', 'calorie', 'nutrient', 'mineral']
    if any(t in text_lower for t in c_triggers) and any(n in text_lower for n in c_nutrients) and not food_query:
        return "clarify_nutrients", 0.90
    
    if loaded_model is not None and tokenizer_model_for_code is not None:
        sequence = tokenizer_model_for_code.texts_to_sequences([text])
        padded = tensorflow.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=m_l, padding="post")
        prediction = loaded_model.predict(padded, verbose=0)
        i_idx = numpy.argmax(prediction)
        confidence = float(numpy.max(prediction))
        intent = label[i_idx]
        return intent, confidence

    return "nutri_data", 0.5

def fuzzy(query, candidates, threshold=0.6):
    b_match = None
    b_score = threshold
    
    query_lower = str(query).lower().strip()
    for c in candidates:
        candidate_lower = str(c).lower().strip()
        score = SequenceMatcher(None, query_lower, candidate_lower).ratio()
        if score > b_score:
            b_score = score
            b_match = c
    
    return b_match, b_score

def bsi(words):
    wrd = [str(w).lower().strip() for w in words if pandas.notna(w)]
    wrd = list(dict.fromkeys(words))
    
    if not wrd:
        return [], None, None
    
    vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 3)).fit(words)
    matrix = vec.transform(words)
    return wrd, vec, matrix

def e_extraction(text, merged_data):
    if merged_data.empty:
        return {"food": None, "nutrient": None, "quantity": 100}
    
    c_foods = {
        'chicken': 'chicken',
        'rice': 'rice',
        'chips': 'chips',
        'egg': 'egg',
        'milk': 'milk',
        'spinach': 'spinach',
        'pasta': 'pasta',
        'apple': 'apple',
        'nuts': 'nuts',
        'tomato': 'tomato',
        'banana': 'banana',
        'carrot': 'carrot',
        'oats': 'oats',
        'potato': 'potato',
        'yogurt': 'yogurt',
        'cheese': 'cheese',
        'orange': 'orange',
        'bread': 'bread',
        'avocado': 'avocado',
        'beef': 'beef',
        'fish': 'fish',
        'broccoli': 'broccoli'
    }
    
    text_lower = text.lower()
    
    f_found = None
    for common, label_name in c_foods.items():
        if common in text_lower:
            df_match = merged_data[merged_data['food'].str.lower().str.contains(common, case=False, na=False)]
            if not df_match.empty:
                f_found = df_match['food'].values[0]
                break
            
    if not f_found:
        f_list = merged_data["food"].dropna().unique().tolist()
        f_found, score = fuzzy(text, f_list, threshold=0.5)
    
    n_found = None
    n_list = merged_data["nutrient_name"].dropna().unique().tolist()

    for al, stan in nutrient_alias.items():
        if al in text_lower:
            for nut in n_list:
                if stan.lower() in nut.lower():
                    n_found = nut
                    break
            if n_found:
                break
    
    if not n_found:
        n_found, score = fuzzy(text, n_list, threshold=0.5)

    qty = 100
    q_match = re.search(r"(\d+)\s?(g|grams?|ml|mg|oz|ounce)", text.lower())
    if q_match:
        qty = int(q_match.group(1))
    
    return {
        "food": f_found,
        "nutrient": n_found,
        "quantity": qty
    }

def bsm(vectorizer, matrix, words, user_input, threshold=0.3):
    if not words or vectorizer is None or matrix is None:
        return None, 0.0
    
    vec = vectorizer.transform([user_input.lower().strip()])
    s = cosine_similarity(vec, matrix)
    best_idx = numpy.argmax(s)
    b_score = float(s[0][best_idx])
        
    if b_score >= threshold:
        return words[best_idx], b_score
    return None, b_score
