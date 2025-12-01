# chat_pipeline 

"""
 1. Reads the users message.
 2. Send to NLU.
 3. Connect to Data and generate a response.
 4. Provide final response to the user.
"""


from chatbot_and_nlp_models.nlu import intent_classifciation, entity_extraction
from chatbot_and_nlp_models.response_gen import (
    get_nutrient_info,
    suggest_healthier_alternatives,
    verify_claim
)

def handle_user_input(message, data):
    
    intent = intent_classifciation(message)
    info = entity_extraction(message, data)

    if intent == "nutrition_info":  

        return get_nutrient_info(
            info.get("food"),
            info.get("nutrient"),
            data,
            info.get("quantity", 60)
        )
    elif intent == "healthier_alternative":    

        options = suggest_healthier_alternatives(
            info.get("food"),
            info.get("nutrient"),
            data 
        )
        return f"Here are some healthier options in contrast to {info.get('food')} : {','.join(options)}"

    else:

        return verify_claim(message)
