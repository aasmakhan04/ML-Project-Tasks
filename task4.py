#Language Tanslation tool
#impoting libraries
from googletrans import Translator

#creating an object
a = Translator()

#taking input from the user
inp = input("Enter your text:")

#specify the language from which we have to translate and to which language
outp = ts.translate(inp, dest='tr',src='en')

#printing the output
print(outp.text)

