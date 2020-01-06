import re
import os
from fuzzywuzzy import fuzz,process

class EntityExtractor:
    def __init__(self,**kwargs):
        pass

    def getDate(self,sentence):
        dateEntity1 = re.findall(r"\d+[\./-]\d+[\./-]\d+",sentence)
        dateEntity2 = re.findall(r"(\d+\D{1,5}\s(november|december|january|february|march|april|may|june|july|august|september|october))",sentence)
        dateEntity3 = re.findall(r"((november|december|january|february|march|april|may|june|july|august|september|october)\s\d{1,2})",sentence)

        if dateEntity1:
            print("date 1: "+dateEntity1[0])
            # refinedQuery = re.sub(r"\d+[\./-]\d+[\./-]\d{2,4}","",sentence)
            return dateEntity1[0]

        elif dateEntity2:
            print("Date 2: "+dateEntity2[0][0])
            # refinedQuery = re.sub(r"(\d+\D{1,5}\s(november|december|january|february|march|april|may|june|july|august|september|october))","",sentence)
            return dateEntity2[0][0]

        elif dateEntity3:
            print("Date 3: "+dateEntity3[0][0])
            # refinedQuery = re.sub(r"((november|december|january|february|march|april|may|june|july|august|september|october)\s\d{1,2})","",sentence)
            return dateEntity3[0][0]
        else:
            return "",sentence

    def getPhoneNumber(self,sentence):

        phone_number = re.findall(r"\d{10,11}",sentence)
        if phone_number:
            print("Phone Number: "+phone_number[0])
            # refinedQuery = re.sub(r"\d{10,11}","",sentence)
            return phone_number[0]
        else:
            return "",sentence

    def getTime(self,sentence):
        time = re.findall(r"(\d+(((\sam|\spm)|am|pm)|(\.\d+(\sam|\spm|am|pm))))",sentence)
        if time:
            print("Time: "+time[0][0])
            # refinedQuery = re.sub(r"(\d+(((\sam|\spm)|am|pm)|(\.\d+(\sam|\spm|am|pm))))","",sentence)
            return time[0][0]
        else:
            return "",sentence
    def getNumbers(self,sentence):
        numbers = re.findall(r"(\d+(\s(tk|taka)|(taka|tk)))",sentence)
        if numbers:
            print("Numbers: ",numbers[0][0])
            return numbers[0][0],sentence
        else:
            return "",sentence