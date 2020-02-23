# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 23:02:17 2020

@author: Tanmay Thakur
""" 
import sys
import logging

from predict import prediction
from functools import wraps
from telegram.ext import Updater, MessageHandler, Filters


sys.setrecursionlimit(1000000)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)


def some_func(bot, update):
    pass
    if not update.effective_message.text:
        update.effective_message.reply_text(text = "This bot is only capable of Natural Language Processing Tasks!")
    else:
        msg = update.effective_message.text
        update.effective_message.reply_text(text = prediction(msg))
        
def main():
    updater = Updater('991501283:AAHKAhIb8LZceoP4GH5FPEhmbVY_MN3HQZA')
    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.all, some_func))
    updater.start_polling()
    updater.idle()
    
if __name__ == '__main__':
    main()