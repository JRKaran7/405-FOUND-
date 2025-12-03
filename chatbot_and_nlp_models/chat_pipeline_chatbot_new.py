# chat_pipeline

from chatbot_and_nlp_models.nlu import classifying_intents, extracting_entities
from chatbot_and_nlp_models.response_gen import (
    get_nutrient_info,
    alternative_healthier_suggestions,
    claim_verification
)



def handle_user_input (message, data=None):
    # when no messages


    if not message or not message.strip():
        return "Please send me a message! Ask me anything about food nutrients, recipes, or diet myths."
    
    # if theres no response


    if data is None:
        from data_cleaning_part.preprocess import load
        data = load()


        # no data

        if data.empty:
            
            return "Sorry, I'm having trouble accessing the nutrition database. Please try again shortly."  # Give this answer when data isnt given
    
    # NLU run
    intent = classifying_intents(message)
    entities = extracting_entities(message, data)

    # from data
    food = entities.get("food")  #all this is given and under dataset
    
    nutrient = entities.get("nutrient")
    quantity = entities.get("quantity", 100)
    
    # responses begin here
    if intent ==  "hello":
        
        return "Hello! 👋 I'm NutriClear, your nutrition assistant. Ask me about:\n• Nutritional content of foods\n• Healthy alternatives\n• Diet myths\n• Recipe ideas\n\nWhat would you like to know?"
    
    if intent == "bye":

        return "Goodbye! 😊 Stay healthy and remember to eat a balanced diet!"   
    if intent == "thank you":
        
        return"You're welcome! 🌱 Feel free to ask more questions anytime."
    
    # under nutri data
    if intent == "nutri_data":

        if not food and not nutrient:  # both things are missing

            text_lower = message.lower()
            if 'in' in text_lower:

                parts = text_lower.split('in')
                if len(parts) >= 2:
                    
                    potential_food = parts[-1].strip().split()[0]      
                    entities = extracting_entities(potential_food, data)
                    food = entities.get("food")   


            if not food:
                
                return"Please specify a food item. For example:\n• 'How many calories in an apple?'\n• 'What nutrients are in chicken?'\n• 'Protein content in eggs'\n• 'Tell me about rice nutrition'"
        if food:
            

            return get_nutrient_info(food, nutrient, data, quantity)
        else:
            
            return f"I couldn't identify the food you're asking about. Please be more specific!"
        
    # under health alt
    if intent == "alternative_food":   # again under dataste
        if nutrient:
            options = alternative_healthier_suggestions(nutrient, data)

            if not options:
                
             return f"I couldn't find foods high in '{nutrient}'. Try asking about protein, fiber, vitamins, or minerals."
            

            return f"🥗 Foods high in {nutrient}:\n" + "\n".join([f"• {food}" for food in options[:5]])

        return "For healthier alternatives, try:\n• Fruits instead of candy\n• Water instead of soda\n• Nuts instead of chips\n• Whole grain instead of white bread\n\nAsk about specific nutrients for more suggestions!"
    
    # recipes suggestions as well

    if intent == "recipes":    # yummy food suggestios
        
        if food:

            df_food = data[data['food'].str.contains(str(food), case=False, na=False)]
            if not df_food.empty:
                
                actual_food = df_food['food'].values[0]
                return f"🍳 Recipe ideas with {actual_food}:\n• Grilled with vegetables\n• In a salad with leafy greens\n• Mixed into a stir-fry\n• As part of a balanced bowl\n\nWould you like nutritional info for any specific recipe?"
        
        return "🍳 Here are some healthy recipe ideas that you can use:\n• Grilled chicken with roasted vegetables\n• Quinoa salad with beans and avocado\n• Baked salmon with sweet potato\n• Greek yogurt parfait with berries\n\nTell me your preferred ingredients for specific suggestions!"
    
    # further info abt clains

    if intent == "clarify_nutrients":  # again in the data sets
        
        explanations = {
            "protein": "Protein builds and repairs tissues. Found in meat, fish, eggs, beans, and dairy. Adults need ~0.8g per kg of body weight daily.",

            "carbohydrate": "Carbs provide energy. Complex carbs (whole grains, vegetables) are better than simple sugars. They're your body's main fuel source.",
            "fat": "Healthy fats support brain function and hormone production. Sources: avocados, nuts, olive oil, fatty fish.",

            "fiber": "Fiber aids digestion and keeps you full. Found in fruits, vegetables, whole grains, and legumes. Aim for 25-30g daily.",
            "vitamin": "Vitamins are essential nutrients. Each vitamin has specific roles - ask about a specific one!",
            "mineral": "Minerals like calcium, iron, and zinc support various body functions. Found in diverse foods.",

            "calorie": "Calories measure energy in food. Your body needs them to function, but excess leads to weight gain."
        }

        for key, explanation in explanations.items():


            if key in message.lower():
                
                return f"💡 {explanation}"
        
        return"💡 I can explain: protein, carbohydrates, fats, fiber, vitamins, minerals, calories, and more. What would you like to know about?"
    
    # input advice 
    if intent == "personal_recommendation":  
        
        recommendations = {
            "muscle": "💪 For muscle building:\n• High protein: chicken, fish, eggs, Greek yogurt, tofu\n• Complex carbs for energy: brown rice, oats, sweet potato\n• Stay hydrated and eat enough calories!",
            "lose": "⚖️ For healthy weight loss:\n• High-fiber foods: vegetables, fruits, whole grains\n• Lean proteins: chicken breast, fish, legumes\n• Drink plenty of water\n• Avoid crash diets - sustainable habits work best!",
            "energy": "⚡ For more energy:\n• Complex carbs: oats, quinoa, whole grain bread\n• Iron-rich foods: spinach, lean meat, lentils\n• Stay hydrated throughout the day",
            
            "general": "🌱 For general health:\n• Eat a variety of colorful vegetables\n• Include lean proteins\n• Choose whole grains over refined\n• Stay hydrated\n• Practice portion control"
        
        }
        
        for key, rec in recommendations.items():
            
            if key in message.lower():
                return rec + "\n\n⚠️ For personalized advice, consult a registered dietitian!"
        
        return recommendations["general"]
    
    if intent == "debunking_myth":   # lies and truth 
        
        myths = {
            "carb": "❌ MYTH: Carbs make you fat\n✅ TRUTH: Excess calories make you gain weight, not carbs themselves. Whole grains and complex carbs are healthy!",
            
            "detox": "❌ MYTH: Detox teas burn fat\n✅ TRUTH: Your liver and kidneys naturally detox. Stay hydrated and eat balanced meals instead.",
            "9pm": "❌ MYTH: Eating after 9pm causes weight gain\n✅ TRUTH: Total daily calories matter more than timing. Your body doesn't have a clock for fat storage!",
            "lemon": "❌ MYTH: Lemon water burns fat\n✅ TRUTH: Lemon water is hydrating but doesn't burn fat. Weight loss requires calorie deficit and exercise.",
            
            
            "supplement": "❌ MYTH: Supplements are better than food\n✅ TRUTH: Whole foods provide nutrients in better forms with additional beneficial compounds. Supplements can't replace a balanced diet.",
            "breakfast": "❌ MYTH: Skipping breakfast helps lose weight\n✅ TRUTH: It depends on the person. Some do well with intermittent fasting, others need breakfast. Total daily intake matters most."
        }
        
        for key, myth in myths.items():
            if key in message.lower():
                return myth
        
        return"🧐 Many diet myths exist! Ask me about:\n• Carbs and weight gain\n• Detox teas\n• Eating late at night\n• Lemon water\n• Supplements vs food\n• Skipping breakfast"
    
    if intent == "diet_log":
        

        return "📝 Meal logged! (Note: This is a demo - full logging functionality would require a database)\n\nYour meal has been noted. Ask me about its nutritional content if you'd like!"

    return claim_verification(intent)  # if all else fails 
