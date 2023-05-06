import telebot
from telebot.types  import InlineKeyboardMarkup, InlineKeyboardButton 
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error


df =  pd.read_csv(r"C:\Users\pc\Desktop\telebot_ws\tg_bot.csv")
country_list = df['country']

TOKEN = '5952353728:AAGy_7ASK8hBHimllOAP1Kzj3tcF17OUOGc'  
bot = telebot.TeleBot(TOKEN) 


@bot.message_handler(commands=['start'])
def message_handler(message): 
    check = bot.send_message(message.chat.id, "Введите первую букву страны на латинском")
    bot.register_next_step_handler(check, country)


def country(check):
    if len(check.text) == 1 and check.text.isalpha():
        word = check.text
        res = [idx for idx in country_list if idx[0].lower() == word.lower()]   
        bot.send_message(check.chat.id, str(res).replace('[','').replace(']',''))
        bot.send_message(check.chat.id, "Выберите cтрану из списка")
        bot.register_next_step_handler(check, country_info)
    else:
        bot.send_message(check.chat.id, "Введите только первую букву на латинском")
        bot.register_next_step_handler(check, fix)


def country_info(check):
    word = check.text
    for i in range(213):
        if (df['country'][i]) == word:
            if df['danger level for tourists'][i] == 2:
                bot.send_photo(check.chat.id, open('red.jpg', 'rb')) 
                bot.send_message(check.chat.id, "В cтране сейчас красный уровень заболеваемости")
            elif df['danger level for tourists'][i] == 1:
                bot.send_photo(check.chat.id, open('yellow.jpg', 'rb'))    
                bot.send_message(check.chat.id, "В cтране сейчас жёлтый уровень заболеваемости")         
            elif df['danger level for tourists'][i] == 0:
                bot.send_photo(check.chat.id, open('green.jpg', 'rb'))            
                bot.send_message(check.chat.id, "В cтране сейчас зеленый уровень заболеваемости")    
            bot.send_message(check.chat.id, "Напишите название страны еще раз, чтобы получить график с прогнозом по заболеваемости")     
            bot.register_next_step_handler(check, graph)  


def fix(check):
    word = check.text    
    if len(check.text) == 1 and check.text.isalpha():
        res = [idx for idx in country_list if idx[0].lower() == word.lower()]   
        bot.send_message(check.chat.id, str(res).replace('[','').replace(']',''))
        bot.send_message(check.chat.id, "Выберите cтрану из списка")
        bot.register_next_step_handler(check, country_info)
    else:
        bot.send_message(check.chat.id, "Вы сделали что то не так, попробуйте заново ")
        bot.send_message(check.chat.id, "Введите первую букву страны на латинском")
        bot.register_next_step_handler(country, fix)

def graph(check):
    word = check.text   
    train_data = pd.read_csv(r"C:\Users\pc\Downloads\forML_1.csv")
    ### Делаем тип обьекта datatime в столбце "Date"

    train_data['Date'] = pd.to_datetime(train_data['Date'])
    train_data.set_index('Date', inplace=True)

    country = train_data[train_data["location"].str.contains(word)] # Все строки со значениями Афганистан

    df = country[['total_cases', 'new_cases', 'total_deaths','new_deaths', 'danger level for tourists']]
    forecast_out = int(math.ceil(0.05 * len(df))) # forcasting out 5% of the entire dataset
    df['label'] = df['total_cases'].shift(-forecast_out)
    scaler = StandardScaler()
    X = np.array(df.drop(['label'], 1))
    scaler.fit(X)
    X = scaler.transform(X)
    X_Predictions = X[-forecast_out:] # data to be predicted
    X = X[:-forecast_out] # data to be trained
    df.dropna(inplace=True)
    y = np.array(df['label'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    rf_confidence = rf.score(X_test, y_test)
    last_date = df.index[-1] #getting the lastdate in the dataset
    last_unix = last_date.timestamp() #converting it to time in seconds
    one_day = 86400 #one day equals 86400 seconds
    next_unix = last_unix + one_day # getting the time in seconds for the next day
    forecast_set = rf.predict(X_Predictions) # predicting forecast data
    df['Forecast'] = np.nan
    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

    fig = plt.figure(figsize=(18, 8))
    df['total_cases'].plot()
    df['Forecast'].plot()
    plt.title(country["location"][1])
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('total_deaths')
    fig.savefig(r'C:\Users\pc\Desktop\telebot_ws\saved_figure.png')
    bot.send_photo(check.chat.id, open(r'C:\Users\pc\Desktop\telebot_ws\saved_figure.png', 'rb'))



bot.polling()
