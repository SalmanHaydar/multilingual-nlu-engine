from entitity_machine import EntityExtractor

if __name__=="__main__":
    sentence = "please send 50 tk to 01834018687 12.11.1995 at 3.00 pm"

    extractor = EntityExtractor()
    extractor.getDate(sentence.lower())
    extractor.getTime(sentence.lower())
    extractor.getPhoneNumber(sentence.lower())
    extractor.getNumbers(sentence)
    # print(date)