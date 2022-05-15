# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 11:02:06 2020

@author: OHyic

"""
#Import libraries
from GoogleImageScrapper import GoogleImageScraper
import os


if __name__ == "__main__":
    #Define file path
    webdrisover_path = os.path.normpath(os.getcwd()+"\\webdriver\\chromedriver.exe")
    image_path = os.path.normpath(os.getcwd()+"\\photos")

    #Add new search key into array ["cat","t-shirt","apple","orange","pear","fish"]
    # search_keys= ['rickshaw','bike', 'motorbike', 'auto rickshaw', 'rickshaw pakistan']
    # search_keys=['islamabad road traffic']
    # search_keys = ['rawalpindi motorway traffic', 'sindh motorway traffic', 'rickshaw traffic sindh', 'rickshaw traffic hyderabad']
    # search_keys = ['motorway rickshaw motorbike hyderabad','motorway rickshaw motorbike lahore','motorway rickshaw motorbike karachi']
    # search_keys = ['pakistan road rickshaw', 'pakistan road motorbike', 'quetta road traffic', 'hyderabad road traffic', 'foodpanda bykea bike in traffic']
    search_keys = ['motorbike rickshaw in traffic pakistan']

    #Parameters
    number_of_images = 500
    headless = False
    min_resolution=(0,0)
    max_resolution=(9999,9999)

    #Main program
    for search_key in search_keys:
        image_scrapper = GoogleImageScraper(webdrisover_path,image_path,search_key,number_of_images,headless,min_resolution,max_resolution)
        image_urls = image_scrapper.find_image_urls()
        image_scrapper.save_images(image_urls)
    
    #Release resources    
    del image_scrapper