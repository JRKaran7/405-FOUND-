from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from data_cleaning_part.preprocess import loading_datasets
from chatbot_and_nlp_models.chat_pipeline_chatbot import handle_user_input

m_df = loading_datasets()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # default starting message for the chatbot 
    w_msg = """👋 Hi!
I am your nutritional chatbot. This is HealthyMakan. I can help you with the following:
1. 🥦 Nutritional information about foods
2. 🥗 Healthy food alternatives
3. 🍲 Recipe suggestions
4. 🪄 Explanations of nutrients
5. 🧐 Debunking diet myths

Sample Questions to Ask: - 
- "How many calories in an apple juice?"
- "What nutrients are in chicken?"
- "Suggest healthy alternatives to chips"
- "What is protein?"
- "Do carbs make you fat?"

What do you want to learn about?"""
    
    await update.message.reply_text(w_msg)

async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # show a help menu for whenever someone need help
    h = """HealthyMakan Commands:

/start - Start the bot
/help - Show this help message
/examples - See example queries
You can message me with your nutrition question!
"""
    await update.message.reply_text(h)

async def examples(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # basic questions the user can ask the bot
    eg = """Example Queries:

Healthy Alternatives:
- "Healthier alternatives to soda"
- "Foods high in fiber"
- "What can I eat instead of chips?"

Recipes & Meal Ideas:
- "Healthy dinner ideas"
- "High protein breakfast"
- "What can I make with chicken?"

Understanding Nutrients:
- "What are macronutrients?"
- "Which foods have protein?"
- "What does fiber do?"

Nutritional Information:
- "How many calories are in 150g of rice?"
- "What is the protein content in tofu?"
- "What are the nutrients present in salmon?"

Exposing Myths :
- "Do carbs make you fat?"
- "Does lemon water burn fat?"
- "Is eating after 9pm bad?"

Try any of these or you can ask your own queries!"""
    
    await update.message.reply_text(eg)

async def message_handling(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # taking the chatboths response and then handle the user messages
    u_input = update.message.text
    u_name = update.effective_user.first_name
    
    print(f"Message from {u_name}: {u_input}")
    
    await update.message.chat.send_action("typing")
    
    res = handle_user_input(u_input, m_df)
    
    await update.message.reply_text(res)

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # identify error and let. the user know by a friendly message when something is wrong
    print(f"Error: {context.error}")
    if update and update.message:
        await update.message.reply_text(
            "Sorry! Something went wrong...can you please repeat the question?."
        )

if __name__ == "__main__":
    token_of_bot = "8336978074:AAH0Lncv7fVmzX-pQiRdV4PqzL9O31m9qsA"
    
    telegram_app_bot = ApplicationBuilder().token(token_of_bot).build()
    
    telegram_app_bot.add_handler(CommandHandler("start", start))
    telegram_app_bot.add_handler(CommandHandler("help", help))
    telegram_app_bot.add_handler(CommandHandler("examples", examples))
    telegram_app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handling))
    
    telegram_app_bot.add_error_handler(error)
    print("The Bot is running!")
    
    telegram_app_bot.run_polling(allowed_updates=Update.ALL_TYPES)