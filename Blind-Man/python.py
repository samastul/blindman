from gtts import gTTS
import os
from playsound import playsound  
import pyttsx3  
engine = pyttsx3.init()  

engine.say('stranger')
engine.runAndWait()  