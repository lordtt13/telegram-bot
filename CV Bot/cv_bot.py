# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 00:00:12 2019

@author: tanma
"""
import urllib.request
from telegram import ChatAction
from telegram.ext import Updater, MessageHandler, Filters
import integrator as i
from functools import wraps  
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)


def send_typing_action(func):


    @wraps(func)
    def command_func(update, context, *args, **kwargs):
        context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=ChatAction.TYPING)
        return func(update, context,  *args, **kwargs)

    return command_func


def download_image(url,name):
    fullname = str(name)+".jpeg"
    urllib.request.urlretrieve(url,fullname)

@send_typing_action
def some_func(bot, update):
    pass
    if not update.effective_message.photo:
        update.effective_message.reply_text(text = "This bot is only capable of Computer Vision Tasks!")
    else:
        msg = update.effective_message
        file_id = msg.photo[-1].file_id
        photo = bot.get_file(file_id)
        download_image(photo["file_path"],'wassup')
        update.effective_message.reply_text(text = i.beautify(i.integrator('wassup.jpeg')))
        
def main():
    updater = Updater(token='TOKEN', use_context=True)
    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.all, some_func))
    updater.start_polling()
    updater.idle()
    
if __name__ == '__main__':
    main()