from chatbot_and_nlp_models.nlu import intent_classification, e_extraction
from chatbot_and_nlp_models.generation_of_responses import (
    get_nutrient_info,
    alternative_healthier_suggestions,
    claim_verification,
    validate_response
)
import json

class BotResponse(str):
    
    """
    Chatbot responses are generated from json dataset
    """
    
    def __new__(cls, message, context=None, debug=False):
        
        
        instance = super().__new__(cls, message)
        instance.message = message
        instance.context = context or {}
        instance.debug = debug
        return instance
    
    def __str__(self):
    
        # reply with msg for the printing
        
        return self.message
    
    def __repr__(self):
        
        # return ..
        
        return self.message
    
    def to_dict(self):
        
        # for API RESPONSES

        return {
            "response": self.message,
            "context": self.context
        }
    
    def to_json(self):
        
        # GO back to json 
        
        return json.dumps(self.to_dict())
    
    def get_message(self):
        
        
        # get a reply back
        
        return self.message
    
    def get_context(self):
        
        
        # background info on the clients msg and question
        return self.context

def handle_user_input(message, data=None, context=None, return_format="text"):
    
    """ 
    main chat func pipeline - take input and give out responses based on question asked
    
    message : users questions 
         data: nutrition data set
             context: memory of prev convo
                  return format: use of "text" "dict" or "json"

    Returns: Object with message and prev context

    """
    
    # intialize the context when it is required 

    if context is None:   # context as in prev history of text
        context = {
            "conversation_history": [],
            "last_food": None,
            "last_nutrient": None
        }
    
    # all abt input
    if not message or not message.strip():
        return BotResponse(
            "Please send me a message! Ask me about:\n- Food nutrients\n- Healthy alternatives\n- Diet myths\n- Recipes",
            context
        )
    
    # when we dont have the data for the questions
    if data is None:
        try:
            from data_cleaning_part.preprocess import load
            data = load()
        except:
            return BotResponse(
                "Sorry, I'm having trouble accessing the nutrition database. Please try again shortly.",
                context
            )
    
    if data.empty:
        return BotResponse(
            "Sorry, the nutrition database is currently unavailable. Please try again shortly.",
            context
        )
    
    # all abt NLU and its processing 
    
    intent, intent_confidence = intent_classification(message)
    entities = e_extraction(message, data)
    
    food = entities.get("food")
    nutrient = entities.get("nutrient")
    quantity = entities.get("quantity", 100)
    
    # Update context from convo history
    context["last_food"] = food
    context["last_nutrient"] = nutrient
    context["conversation_history"].append({

        "message": message,
        "intent": intent,
        "confidence": float(intent_confidence),
        "food": food,
        "nutrient": nutrient

    })
    
    # intent handling - responses to msges
    
    if intent == "hello":
        
        response = ("Hello! 👋 I'm Healthy Makan, your nutrition assistant. Ask me about:\n"
                   "🥦 Nutritional content of foods\n"
                   "🥗 Healthy alternatives\n"
                   "🍽️ Diet myths\n"
                   "🍲 Recipe ideas\n\n"
                   "🪄 What would you like to know?")
        
        return BotResponse(response, context)
    
    if intent == "bye":
        

        response = "Goodbye! 😊 Stay healthy and eat balanced meals!"
        return BotResponse(response, context)
    
    if intent == "thank you":
        response = "You're welcome! 🌱 Feel free to ask anytime."
        
        return BotResponse(response, context)
    
    
    # nutrition data intent 

    if intent == "nutri_data":

        #  extract food if missing

        if not food:
            text_lower = message.lower()
            if 'in' in text_lower:
                parts = text_lower.split('in')
                if len(parts) >= 2:
                    potential_food = parts[-1].strip().split()[0]
                    temp_entities = e_extraction(potential_food, data)
                    food = temp_entities.get("food")
                    context["last_food"] = food
    

        if not food:
            response = ("Please specify a food item. Examples:\n"
                       "- 'How many calories in an apple?'\n"
                       "- 'What nutrients are in chicken?'\n"
                       "- 'Protein content in eggs'\n"
                       "- 'Tell me about rice nutrition'")
            
            return BotResponse(response, context)
        
        response = get_nutrient_info(food, nutrient, data, quantity)
        if not validate_response(response, intent):
            response = f"I couldn't find complete nutrition data for '{food}'. Try being more specific."

        return BotResponse(response, context)
    

    # alternative food intent - opersonalised advice from the bot
    if intent == "alternative_food":
        if not nutrient:
            from chatbot_and_nlp_models.nlu import match_nutrient_keywords
            nutrient = match_nutrient_keywords(message, data['nutrient_name'].unique().tolist())
        
        
        if nutrient:
        
            options = alternative_healthier_suggestions(nutrient, data)
            
            if not options:
                response = (f"I couldn't find foods high in '{nutrient}'.\n"
                           f"Try asking me more about: protein, fiber, vitamins, calcium, or iron.")
                
                return BotResponse(response, context)
            
            # for when the data is present in the dataset 


            if options and isinstance(options[0], tuple):
                foods_list = "\n".join([f"• {food[0]}: {food[1]}" for food in options[:5]])
                
            else:
                foods_list = "\n".join([f"• {food}" for food in options[:5]])
            
            response = f"🥗 Foods high in {nutrient}:\n{foods_list}\n\nAsk for a recipe with any of these!"

            return BotResponse(response, context)
        
        response = ("For healthier alternatives, try:\n"
                   "- Fruits instead of candy\n"
                   "- Water instead of soda\n"
                   "- Nuts instead of chips\n"
                   "- Whole grain bread instead of white bread\n\n"
                   "✨ Ask about specific nutrients for personalized suggestions!")
        

        return BotResponse(response, context)
    
    # intent for recipes 

    if intent == "recipes":
        
        # Recipe but with nutri data set given

        recipes_db = {
            'chicken': {
                'name': 'Grilled Chicken with Vegetables',
                'prep_time': '15 mins',
                'cook_time': '20 mins',
                'servings': 2,
                'nutrition': '~280 cal, 31g protein, 8g carbs, 12g fat per serving',
                'ingredients': ['200g chicken breast', '200g mixed vegetables', '2 tbsp olive oil', 'garlic, herbs'],
                'steps': [
                    '1. Season chicken with salt, pepper, garlic',
                    '2. Heat olive oil in pan',
                    '3. Cook chicken 8-10 mins each side',
                    '4. Add vegetables in last 5 mins',
                    '5. Serve hot with lemon juice'
                ]
            },
            'rice': {
                'name': 'Brown Rice with Veggies',
                'prep_time': '10 mins',
                'cook_time': '25 mins',
                'servings': 3,
                'nutrition': '~220 cal, 5g protein, 45g carbs, 2g fat per serving',
                'ingredients': ['150g brown rice', '200g mixed veggies', '2 cups water', 'salt, oil'],
                'steps': [
                    '1. Rinse rice under water',
                    '2. Boil water with salt',
                    '3. Add rice, reduce heat',
                    '4. Simmer 20-25 mins',
                    '5. Stir-fry with veggies in final 5 mins'
                ]
            },
            'egg': {
                'name': 'Veggie Omelet',
                'prep_time': '5 mins',
                'cook_time': '8 mins',
                'servings': 1,
                'nutrition': '~200 cal, 14g protein, 3g carbs, 15g fat',
                'ingredients': ['2 eggs', '100g spinach', '50g cheese', '1 tsp butter'],
                'steps': [
                    '1. Sauté spinach until tender',
                    '2. Beat eggs with salt & pepper',
                    '3. Heat butter in pan',
                    '4. Pour eggs, add spinach',
                    '5. Fold when set, add cheese'
                ]
            },
            'fish': {
                'name': 'Baked Salmon with Herbs',
                'prep_time': '10 mins',
                'cook_time': '15 mins',
                'servings': 2,
                'nutrition': '~250 cal, 28g protein, 0g carbs, 14g fat per serving',
                'ingredients': ['200g salmon fillet', 'lemon', 'herbs', 'olive oil'],
                'steps': [
                    '1. Preheat oven to 200°C',
                    '2. Place salmon on foil',
                    '3. Season with lemon, herbs, oil',
                    '4. Wrap foil loosely',
                    '5. Bake 12-15 mins until flaky'
                ]
            },
            'spinach': {
                'name': 'Spinach & Garlic Saute',
                'prep_time': '5 mins',
                'cook_time': '5 mins',
                'servings': 2,
                'nutrition': '~35 cal, 3g protein, 3g carbs, 2g fat per serving',
                'ingredients': ['300g spinach', '3 cloves garlic', '1 tsp olive oil', 'salt, pepper'],
                'steps': [
                    '1. Mince garlic finely',
                    '2. Heat olive oil on medium',
                    '3. Add garlic, cook 30 secs',
                    '4. Add spinach, toss well',
                    '5. Cook until wilted (3-4 mins)'
                ]
            },
            'broccoli': {
                'name': 'Roasted Broccoli',
                'prep_time': '5 mins',
                'cook_time': '20 mins',
                'servings': 2,
                'nutrition': '~55 cal, 4g protein, 8g carbs, 2g fat per serving',
                'ingredients': ['300g broccoli', '2 tbsp olive oil', 'garlic, salt, pepper'],
                'steps': [
                    '1. Preheat oven to 220°C',
                    '2. Cut broccoli into florets',
                    '3. Toss with oil, garlic, seasoning',
                    '4. Spread on baking sheet',
                    '5. Roast 18-20 mins until crispy'
                ]
            },
            'banana': {
                'name': 'Banana & Oat Smoothie Bowl',
                'prep_time': '5 mins',
                'cook_time': '0 mins',
                'servings': 1,
                'nutrition': '~280 cal, 8g protein, 52g carbs, 6g fat',
                'ingredients': ['1 banana', '40g oats', '150ml yogurt', 'berries', 'honey'],
                'steps': [
                    '1. Blend banana with yogurt',
                    '2. Pour into bowl',
                    '3. Top with oats',
                    '4. Add fresh berries',
                    '5. Drizzle honey, serve'
                ]
            }
        }
        
       
        # find the keywords to extract more accurate ans
        
        message_lower = message.lower()
        found_recipe = None
        
        # first step and try
        
        if food:
            food_lower = food.lower()
            for key, recipe_data in recipes_db.items():
                if key in food_lower:
                    found_recipe = recipe_data
                    break
        
        # if first try fails do this

        if not found_recipe:
            
            for key, recipe_data in recipes_db.items():
                if key in message_lower:
                    found_recipe = recipe_data
                    break
        
        # when recipe is found, do this
        
        
        if found_recipe:
            response = (f" {found_recipe['name']}\n\n"
                       f"⏲️ Prep: {found_recipe['prep_time']} | Cook: {found_recipe['cook_time']}\n"
                       f"🍽️ Servings: {found_recipe['servings']}\n"
                       f"📊 Nutrition per serving: {found_recipe['nutrition']}\n\n"
                       f"Ingredients:\n")
            response += "\n".join([f"• {ing}" for ing in found_recipe['ingredients']])
            response += f"\n\nSteps:\n" 
            response += "\n".join(found_recipe['steps'])
            response += f"\n\n💡 Tip: All ingredients should be fresh & organic whenever possible!"
            return BotResponse(response, context)
        
        # if recipe is not in data use this
        
        response = ("🥗 Here are some healthy recipe ideas:\n\n"
                   "Try asking about these ingredients:\n"
                   "- Recipe with chicken -> Grilled Chicken with Vegetables\n"
                   "- Recipe with fish -> Baked Salmon with Herbs\n"
                   "- How to cook eggs -> Veggie Omelet\n"
                   "- Recipe with rice -> Brown Rice with Veggies\n"
                   "- How to prepare spinach -> Spinach & Garlic Sauté\n"
                   "- Broccoli recipe -> Roasted Broccoli\n"
                   "- Banana recipe -> Banana & Oat Smoothie Bowl\n\n"
                   "🧂Pick any ingredient of your choice and I'll give you a complete recipe with it's nutrition info!")
        
        
        return BotResponse(response, context)
    
    # clearing the myths or truths abt nutri questions 
    
    if intent == "clarify_nutrients":
        explanations = {
            "protein": ("💡 Protein is an essential macronutrient that:\n"
                       "- Builds and repairs the bodys tissues such as muscles, skin and hair\n"
                       "- Creates enzymes and hormones\n"
                       "- Supports the immune function\n"
                       "🍎 Daily needs: ~0.8g per kg body weight (e.g., 65g for 80kg person)\n"
                       "🍗 Sources: chicken, fish, eggs, Greek yogurt, beans, nuts"),
                       
            "carbohydrate": ("💡 Carbohydrates provide your body's primary fuel:\n"
                           "- Break it down into glucose for energy\n"
                           "- Complex carbs(whole grains) are better than simple sugars\n"
                           "- Essential for brain and muscle functioning\n"
                           "🍎 Daily needs: 225-325g (45-65% of calories)\n"
                           "🌾 Sources: oats, brown rice, sweet potato, quinoa, vegetables"),
                           
            "fat": ("💡 Fats are critical for health:\n"
                   "- Support brain functions and hormone production\n"
                   "- Help absorb fat-soluble vitamins (A, D, E, K)\n"
                   "- Provide concentrated energy (9 cal/g vs 4 for carbs/protein)\n"
                   "🍎 Daily needs: 50-77g\n"
                   "🥑 Sources: avocados, nuts, olive oil, fatty fishes, seeds"),
                   
            "fiber": ("💡 Fiber aids your digestion and health:\n"
                     "- Promotes healthy digestion\n"
                     "- Keeps you full for longer period of time\n"
                     "- Supports heart health and long term maintanace of body\n"
                     "🍎 Daily needs: 25-30g\n"
                     "🥦 Sources: fruits, vegetables, whole grains, legumes, oats"),
                     
            "vitamin": ("💡 Vitamins are essential micronutrients which:\n"
                       "- Regulate body functions\n"
                       "- Support immune system\n"
                       "- Each has specific roles\n"
                       "✨Ask about: Vitamin C (immunity), Vitamin A (vision), Vitamin D (bones), B vitamins (energy)"),

            "mineral": ("💡 Minerals support vital functions:\n"
                       "- Iron: Oxygen transport (2-3 servings/day)\n"
                       "- Calcium: Bone health (1000-1200mg/day)\n"
                       "- Sodium: Fluid balance (limit to <2300mg/day)\n"
                       "- Zinc: Immune function"),

            "calorie": ("💡 Calories measure food energy:\n"
                       "- Your body needs them to function\n"
                       "- Excess calories → weight gain\n"
                       "- Deficit → weight loss\n"
                       "🍎 Daily needs vary: 1600-2400 (women), 2000-3000 (men)")
        }
        
        for key, explanation in explanations.items():
            if key in message.lower():
                return BotResponse(explanation, context)
        
        response = ("💡 I can explain these nutrients in detail for you:\n"
                   "- Protein - builds muscle\n"
                   "- Carbohydrates - provides energy\n"
                   "- Fats - supports brain & hormone\n"
                   "- Fiber - maintain digestive health\n"
                   "- Vitamins - helps in the functioning of immune & cell\n"
                   "- Minerals - functioning of bone & body\n"
                   "- Calories - measures energy\n\n"
                   "✨Ask: 'What is protein?' or 'Tell me more about fiber'")
        return BotResponse(response, context)
    
    # healthy makans own reco
    
    if intent == "personal_recommendation":
        recommendations = {
            "muscle": ("🏋️‍♀️ For muscle building:\n"
                      "- High protein: chicken, fish, eggs, Greek yogurt, tofu\n"
                      "- Complex carbs: brown rice, oats, sweet potato\n"
                      "- Stay hydrated and eat enough calories!"),
            "lose": ("💪 For healthy weight loss:\n"
                    "- High-fiber foods: vegetables, fruits, whole grains\n"
                    "- Lean proteins: chicken breast, fish, legumes\n"
                    "- Drink plenty of water\n"
                    "- Sustainable habits work much better than crash diets!"),
            "energy": ("🔋 For more energy:\n"
                      "- Complex carbs: oats, quinoa, whole grain bread\n"
                      "- Iron-rich foods: spinach, lean meat, lentils\n"
                      "- Stay hydrated throughout the day"),
            "general": ("🍀 For general health:\n"
                       "- Eat a variety of colorful vegetables\n"
                       "- Include lean proteins to your diet\n"
                       "- Choose whole grains\n"
                       "- Stay hydrated at all times\n"
                       "- Practice portion control")
        }
        
        base_rec = recommendations.get("general")
        for key, rec in recommendations.items():
            if key in message.lower():
                base_rec = rec
                break
        
        response = base_rec + "\n\n‼️ For personalized advice, consult a registered dietician!"
        return BotResponse(response, context)
    
    # correct the wrongs abt the questions asked 
    
    
    if intent == "debunking_myth":
        myths = {
            "carb": ("❌ MYTH: Carbs make you fat\n"
                    "✅ TRUTH: Excess calories cause weight gain, not carbs themselves.\n"
                    "Whole grains and complex carbs are healthy!"),

            "detox": ("❌ MYTH: Detox teas burn fat\n"
                     "✅ TRUTH: Your liver and kidneys naturally detox.\n"
                     "Stay hydrated and eat balanced meals."),
                     
            "9pm": ("❌ MYTH: Eating after 9pm causes weight gain\n"
                   "✅ TRUTH: Total daily calories matter more than timing.\n"
                   "Your body doesn't have a clock for fat storage!"),

            "lemon": ("❌ MYTH: Lemon water burns fat\n"
                     "✅ TRUTH: Lemon water is hydrating but doesn't burn fat.\n"
                     "Weight loss requires calorie deficit and exercise."),

            "supplement": ("❌ MYTH: Supplements are better than food\n"
                          "✅ TRUTH: Whole foods provide nutrients in optimal forms.\n"
                          "Supplements can't replace a balanced diet."),
                          
            "breakfast": ("❌ MYTH: Skipping breakfast helps lose weight\n"
                         "✅ TRUTH: It depends on the person.\n"
                         "Total daily intake matters most.")
        }
        
        for key, myth in myths.items():
            if key in message.lower():
                

                return BotResponse(myth, context)
        
        response = ("🥸 I can debunk these diet myths:\n"
                   "- Carbs and weight gain\n"
                   "- Detox teas\n"
                   "- Eating late at night\n"
                   "- Lemon water burning fat\n"
                   "- Supplements vs whole foods\n"
                   "- Skipping breakfast\n\n"
                   "✨Ask me about any of these!")
        return BotResponse(response, context)
    
    # related to users personal diet
    if intent == "diet_log":
        response = ("📠 Meal logged! (Note: Full logging requires a database)\n"
                   "📝 Your meal has been noted. Ask about its nutritional content if interested!")
        return BotResponse(response, context)
    
    # final fallback
    
    response = claim_verification(intent, message)
    
    return BotResponse(response, context)