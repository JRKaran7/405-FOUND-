from data_cleaning_part.preprocess import loading_datasets
import pandas as pd
import numpy as np

dataset = loading_datasets()

# nutrion numbers from credited sources 
referal_for_nutrition = {
    'chicken': {
        'protein': 26.3,             'fat': 3.6,    'carbs': 0, 
           'calories': 165,      'fiber': 0,               'iron': 0.8, 
                      'calcium': 15,                              'sodium': 74
    },
    'rice': {
        'protein': 2.7,              'fat': 0.3,      'carbs': 28, 
        'calories': 130, 
                        'fiber': 0.4,
                             'iron': 0.2,
                                     'calcium': 10, 
                                                      'sodium': 1
    },
    'egg': {
        'protein': 13.0,                           'fat': 11.0, 
                  'carbs': 1.1,                               'calories': 155, 
                         'fiber': 0,
        'iron': 1.8,           'calcium': 56,   'sodium': 124
    },
    'milk': {
        'protein': 3.2, 
                       'fat': 3.3, 'carbs': 4.8, 
                                          'calories': 61, 'fiber': 0,
                                                          'iron': 0.07,
                                                           
                                         'calcium': 113, 'sodium': 49
    },
    'spinach': {
                         'protein': 2.7, 
                                         'fat': 0.4, 'carbs': 3.6, 
          'calories': 23, 'fiber': 2.2,
                                       'iron': 2.7, 'calcium': 99, 'sodium': 79
    },
    'banana': {
        'protein': 1.1, 'fat': 0.3,                
                   'carbs': 27, 'calories': 105, 
                                   'fiber': 2.6, 'iron': 0.26,
                                                'calcium': 5, 'sodium': 1
    },
    'apple': {
        'protein': 0.3, 'fat': 0.2, 
                      'carbs': 25, 'calories': 95, 'fiber': 4.4,
        'iron': 0.12, 'calcium': 5, 'sodium': 2
    },
    'carrot': {
        'protein': 0.9, 'fat': 0.2, 'carbs': 10,
         
                     'calories': 41, 'fiber': 2.8,
        'iron': 0.3,                         'calcium': 33, 'sodium': 69
    },
    'broccoli': {
             'protein': 2.8, 'fat': 0.4, 
                               'carbs': 7, 'calories': 31, 'fiber': 2.4,
        'iron': 0.73, 'calcium': 89, 'sodium': 65
    },
    'fish': {
        'protein': 25.0, 'fat': 2.0, 
                           'carbs': 0, 'calories': 120, 'fiber': 0,
                                            'iron': 1.0, 'calcium': 15, 'sodium': 75
    }
}

def get_nutrient_info(name_of_the_food, name_of_the_nutrient=None, data=None, quant=100):
    """Lookup nutrients data with credible reference and a fallback of a database"""
    if not name_of_the_food:
        return "Help me by specifying the food item (e.g., 'chicken', 'rice', 'spinach')."
    
    lowerfood = str(name_of_the_food).lower().strip()
    
    # refer the nutrion for the code
    for r_food, nutrients in referal_for_nutrition.items():
        if r_food in lowerfood or lowerfood in r_food:
            return _format_nutrient_response(r_food, name_of_the_nutrient, nutrients, quant)
    
    # backup in case cannot find the food item
    if data is None or data.empty:
        return f"I dont have detailed info of '{name_of_the_food}', but I can reco :\n- Ask more abt foods like 🐔chicken, 🍚rice, 🥚eggs, 🥛milk\n- Specify raw or cooked\n- Include portion size"
    # if the found food is none
    food_found = None
    data_frame_food = data
    [data['food'].str.lower().str.contains
     (lowerfood, 
      case=False, na=False, regex=False)]
    
    # in case data_frame_food doesnt work
    
    
    if data_frame_food.empty:
        words = [w for w in lowerfood.split() if len(w) > 2]

        for word in words:
            
            data_frame_food = data[data['food'].str.lower().str.contains(word, case=False, na=False, regex=False)]
            if not data_frame_food.empty:
                break
    
    if data_frame_food.empty:
       
        return (f"I cannot find '{name_of_the_food}' in our created database. Please Try:\n"
                f"- Being more detailed in the food item (e.g., '🐔chicken breast')\n"
                f"- Using common food names\n"
                f"- Check your spelling")
    actual_name_of_the_food = data_frame_food['food'].values[0]
    
    # Handle specific nutrient query
    if name_of_the_nutrient:
        nutrient_lower = str(name_of_the_nutrient).lower().strip()
        data_frame_nutrient = data_frame_food[data_frame_food['name_of_the_nutrient'].str.lower().str.contains
                                              (nutrient_lower, case=False, na=False, regex=False)]
        
        if data_frame_nutrient.empty:
           
            nutrient_aliases = {
                'calorie': 'energy', 
                                'calories': 'energy',
                                                  'carb': 'carbohydrate', 
                                                                      'carbs': 'carbohydrate',
                'fat': 'lipid'
            }
            
            for alias, actual in nutrient_aliases.items():
                if alias in nutrient_lower:
                    data_frame_nutrient = data_frame_food
                    [data_frame_food['name_of_the_nutrient'].str.lower().str.contains
                     (actual, case=False, na=False, regex=False)]
                    if not data_frame_nutrient.empty:
                        break
        
        if data_frame_nutrient.empty:
            available = data_frame_food['name_of_the_nutrient'].unique()[:5]
            return f"I don't have '{name_of_the_nutrient}' data for '{actual_name_of_the_food}'.\nAvailable: {', '.join(available)}"
        
        data_frame_food = data_frame_nutrient
    
    try:
        if name_of_the_nutrient and not data_frame_food.empty:
            amount_ = data_frame_food['amount'].values[0]
            unit = data_frame_food['unit_name'].values[0]
            actual_nutrient = data_frame_food['name_of_the_nutrient'].values[0]
            final_amount_ = amount_ * (quant / 100)
            interpretation = interpret_nutrient_value(actual_nutrient, final_amount_, unit)
            return f"✅ {quant}g of {actual_name_of_the_food}:\n\n🔹 {actual_nutrient}: {final_amount_:.2f} {unit}\n\n{interpretation}"
        
        response = f"📊 Nutrition Facts for {quant}g of {actual_name_of_the_food}\n\n"
        
        priority_nutrients = ['Energy', 'Protein', 'Total lipid (fat)', 'Carbohydrate', 'Fiber', 'Sugars', 'Calcium', 'Iron', 'Sodium', 'Vitamin C']
        
        shown = 0
        for nutrient in priority_nutrients:
            nutrient_data = data_frame_food[data_frame_food['name_of_the_nutrient'].str.contains(nutrient, case=False, na=False)]
            if not nutrient_data.empty:
                amount_ = nutrient_data['amount of the food item'].values[0]

                unit = nutrient_data['unit name that u are using'].values[0]

                final_amount_ = amount_ * (quant / 100)

                response += f"- {nutrient}: {final_amount_:.2f} {unit}\n"

                shown += 1
        
        if shown == 0:
            response += "\n**The available nutrients:**\n"
            for idx, row in data_frame_food.head(8).iterrows():
                amount_ = row['amount'] * (quant / 100)
                response += f"- {row['name_of_the_nutrient']}: {amount_:.2f} {row['unit_name']}\n"
        
        return response.strip()
    
    except Exception as e:
        return f"There has been an error in retreiving data of the nutrient . Please try again."

def _format_nutrient_response(name_of_the_food, name_of_the_nutrient, nutrients, quant):
    """The formatted data of the nutrition """
    qty_factor = quant / 100
    
    if name_of_the_nutrient:
        nutrient_lower = name_of_the_nutrient.lower()
        
        # Map to hardcoded keys
        nutrient_map = {
            'protein': 'protein', 'fat': 'fat',
          'carbohydrate': 'carbs',
            'carb': 'carbs',    'calories': 'calories',
            'energy': 'calories',     'calorie': 'calories',
            'fiber': 'fiber',                   'iron': 'iron',
            'calcium': 'calcium',
            'sodium': 'sodium'
        }
        
        matched_key = None
        for key, mapped in nutrient_map.items():
            if key in nutrient_lower:
                matched_key = mapped
                break
        
        if matched_key and matched_key in nutrients:
            value = nutrients[matched_key] * qty_factor
            units = {
                'protein': 'g', 'fat': 'g',
                'carbs': 'g','calories': 'kcal','fiber': 'g',
                'iron': 'mg','calcium': 'mg',
                'sodium': 'mg'
            }
            unit = units.get(matched_key, 'g')
            
            interpretation = interpret_nutrient_value(matched_key, value, unit)
            return f"✅ {quant}g of {name_of_the_food.title()}:\n\n🔹 {matched_key.title()}: {value:.2f} {unit}\n\n{interpretation}"
    
    # All the nutritional facts 


    response = f"📊 Nutrition Facts for {quant}g of {name_of_the_food.title()}\n\n"
    response += f"- Protein: {nutrients['protein'] * qty_factor:.2f}g\n"
    response += f"- Fat: {nutrients['fat'] * qty_factor:.2f}g\n"
    response += f"- Carbohydrates: {nutrients['carbs'] * qty_factor:.2f}g\n"
    response += f"- Calories: {nutrients['calories'] * qty_factor:.0f} kcal\n"
    response += f"- Fiber: {nutrients['fiber'] * qty_factor:.2f}g\n"
    response += f"- Calcium: {nutrients['calcium'] * qty_factor:.0f} mg\n"
    response += f"- Iron: {nutrients['iron'] * qty_factor:.2f} mg\n"
    response += f"- Sodium: {nutrients['sodium'] * qty_factor:.0f} mg\n"
    response += f"\n💡 Ask about specific nutrients for more details!"
    
    return response

def interpret_nutrient_value(name_of_the_nutrient, amount, unit):
    """factual nutri data, trust me bro"""
    nutrient_lower = str(name_of_the_nutrient).lower()
    
    if 'protein' in nutrient_lower:
        if amount >= 20:
            return "💪 Excellent protein source - great for muscle building & repair"
        elif amount >= 10:
            return "✅ Good protein content - supports muscle maintenance"
        elif amount >= 5:
            return "📊 Moderate protein - combine with other sources"
        else:
            return "🌾 Low protein - pair with high-protein foods"
    
    elif 'calorie' in nutrient_lower or 'energy' in nutrient_lower:
        if amount < 50:
            return "💚 Very low calorie - excellent for weight management"
        elif amount < 100:
            return "✅ Low-moderate calorie"
        elif amount < 300:
            return "🔹 Moderate calorie density"
        else:
            return "🔸 High calorie - good for energy needs"
    


    elif 'fat' in nutrient_lower or 'lipid' in nutrient_lower:
        
        if amount < 1:
            return "💚 Almost no fat"
        elif amount < 5:
            return "✅ Low fat - healthy choice"
        elif amount < 15:
            return "🔹 Moderate fat - includes healthy fats"
        else:
            return "🔸 Higher fat - good for satiety"
    
    elif 'carb' in nutrient_lower:
        if amount >= 25:
            
            return "🔸 High carbs - good energy source"
        elif amount >= 10:
            return "✅ Moderate carbs - balanced choice"
        else:
            return "💚 Low carbs"
        

    
    elif 'fiber' in nutrient_lower:
        if amount >= 5:
            return "🌾 High fiber - excellent for digestion"
        elif amount >= 2.5:
            

            return "✅ Good fiber content"
        else:
            
            return "📊 Low fiber"
    
    elif 'iron' in nutrient_lower:
        
        if amount >= 2:
            return "⚡ Good iron source - supports energy & oxygen transport"
        else:
            return "📊 Low iron"
    
    elif 'calcium' in nutrient_lower:
        
        if amount >= 100:
            return "💪 Good calcium - supports bone health"
        else:
            
            return "📊 Some calcium content"
    
    elif 'sodium' in nutrient_lower:
        if amount < 100:
            return "💚 Low sodium - heart healthy"
        elif amount < 400:
            return "✅ Moderate sodium"
        else:
            return "🔸 High sodium - limit intake"
    
    return "📊 Nutrient recorded"

def alternative_healthier_suggestions(nutrient, data=None):
    """factual healtheir suggs"""

    # bots personal suggestions

    suggestions_map = {
        'protein': [
            ('Chicken Breast', '165 cal, 31g protein per 100g - lean & versatile'),
            ('Fish (Salmon)', '208 cal, 25g protein + omega-3 fatty acids'),
            ('Eggs', '155 cal, 13g protein - contains choline for brain health'),
            ('Greek Yogurt', '59 cal, 10g protein per 100g - also has probiotics'),
            ('Lentils', '116 cal, 9g protein - high in fiber too')
        ],
        'fiber': [
            ('Oats', '389 cal, 10.6g fiber per 100g - beta-glucan for heart health'),
            ('Spinach', '23 cal, 2.2g fiber per 100g - packed with iron & vitamins'),
            ('Broccoli', '31 cal, 2.4g fiber - cruciferous veggie with sulforaphane'),
            ('Apple', '95 cal, 4.4g fiber - pectin aids digestion'),
            ('Chia Seeds', '486 cal, 9.8g fiber per 100g - soluble & insoluble fiber')
        ],
        'calcium': [
            ('Milk', '61 cal, 113mg calcium per 100g - fortified with vitamin D'),
            ('Greek Yogurt', '59 cal, 100mg calcium - probiotics support gut health'),
            ('Spinach', '23 cal, 99mg calcium per 100g - bioavailable form'),
            ('Broccoli', '31 cal, 89mg calcium - combined with vitamin C for absorption'),
            ('Almonds', '579 cal, 264mg calcium per 100g - healthy fats included')
        ],
        'iron': [
            ('Spinach', '23 cal, 2.7mg iron per 100g - pair with vitamin C for absorption'),
            
            ('Chicken Breast', '165 cal, 0.8mg iron - heme iron is more bioavailable'),
            ('Lentils', '116 cal, 3.3mg iron - also high in fiber'),
            ('Beans', '127 cal, 2.5mg iron - plant-based protein source'),
            ('Dark Chocolate', '598 cal, 12mg iron per 100g - antioxidants too')
        ],
        'carbohydrate': [
            ('Brown Rice', '111 cal, 23g carbs per 100g - retains bran for nutrients'),

            ('Sweet Potato', '86 cal, 20g carbs - high in vitamin A & fiber'),
            ('Oats', '389 cal, 66g carbs - complex carbs with beta-glucan'),
            ('Quinoa', '120 cal, 21g carbs - complete protein with all 9 amino acids'),
            ('Whole Wheat Bread', '265 cal, 49g carbs - sustained energy release')
        ],
        'fat': [
            ('Avocado', '160 cal, 15g fat per 100g - monounsaturated fats for heart'),
            ('Olive Oil', '884 cal, 100g fat - polyphenols reduce inflammation'),

            ('Salmon', '208 cal, 13g fat - omega-3 EPA & DHA for brain health'),
            ('Nuts (Almonds)', '579 cal, 49g fat - vitamin E & magnesium'),

            ('Coconut Oil', '892 cal, 99g fat - medium-chain triglycerides')
        ],

        'energy': [
            ('Banana', '105 cal per 100g - natural sugars + potassium for quick energy'),
            ('Honey', '304 cal per 100g - easily digestible simple sugars'),
            ('Oats', '389 cal per 100g - sustained energy from complex carbs'),

            ('Nuts', '579-680 cal per 100g - healthy fats for lasting energy'),
            ('Dark Chocolate', '598 cal per 100g - caffeine + endorphins')
        ]
    }
    
    nutrient_lower = str(nutrient).lower()
    for key, foods in suggestions_map.items():
        if key in nutrient_lower:
            return foods
    
    # Fallback to database
    if data is None or data.empty:
        return []
    
    df = data[data["name_of_the_nutrient"].str.lower().str.contains(nutrient_lower, case=False, na=False)]
    
    if df.empty:
        return []
    
    df = df.sort_values("amount", ascending=False)
    df = df.drop_duplicates("food", keep='first')
    suggestions = df["food"].head(5).tolist()
    return [str(s) for s in suggestions if pd.notna(s)]

def claim_verification(intent, message=""):
    """final fallback"""


    explanations = {
        "debunking_myth": "Actually many diet mythssss exist! Ask about specific claims and I'll give you the science😉.",
        "clarify_nutrients": "I can explain nutrients and their roles. Ask about protein, carbs, fats, fiber, vitamins, or minerals!",

        "personal_recommendation": "For personalized advice, pls consult a registered dietitian. I can provide general nutrition guidance to you.",
        "recipes": "I can suggest recipe ideas. What ingredients or nutrients interest you?",


        "alternative_food": "I can suggest foods high in specific nutrients. Which nutrient interests you the most?"
    }
    return explanations.get(intent, "I'm here to help with nutrition! Ask about foods, nutrients, recipes, or diet facts!")

def validate_response(response, intent):
    
    """good responses for  when data isnt given"""
    if not response or len(response.strip()) < 5:
        return False
    if "error" in response.lower() and len(response) < 50:
        return False
    if "couldn't" in response.lower() and "food" in response.lower() and len(response) < 100:
        return False
    return True