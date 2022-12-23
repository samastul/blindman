import cv2
import os
from flask import Flask,request,render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import warnings
import pyttsx3  
engine = pyttsx3.init()  

warnings.simplefilter(action='ignore', category=FutureWarning)
#### Defining Flask App
app = Flask(__name__)

face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

# #### get a number of total registered users
# def totalreg():
#     return len(os.listdir('static/faces'))

#### extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points

#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    prediction = model.predict(facearray)
    return prediction

#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')


    ################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def home():   
    return render_template('index.html') 

@app.route('/register')
def register():   
    return render_template('register.html') 


#### This function will run when we click on Take Attendance Button
@app.route('/start',methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('index.html',mess='There is no trained model in the static folder. Please add a new face to continue.') 

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret,frame = cap.read()
        if list(extract_faces(frame)) != []:
            for face in extract_faces(frame):
                (x,y,w,h) = face
                cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
                face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
                identified_person = identify_face(face.reshape(1,-1))[0]
                if identified_person:
                    cv2.putText(frame,f'{identified_person.split("_")[0]}',(x,y-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
                else: 
                    cv2.putText(frame,'stranger',(x,y-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
                    engine.say('stranger')  
                    engine.runAndWait() 
        cv2.imshow('scanner', frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return render_template('index.html') 


#### This function will run when we add a new user
@app.route('/add',methods=['GET','POST'])
def add():
    # newusername = request.form.get('name')
    # emailID = request.form.get('emailid')
    # phonenum = request.form.get('phonenum')

    newusername = input('Enter your name: ')
    phonenum = input('Enter your Phone Number: ')
    
    userimagefolder = 'static/faces/'+str(newusername)+'_'+str(phonenum)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = str(newusername)+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==500:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    return render_template('index.html') 


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)