import cv2
import os
from flask import Flask, request, render_template, redirect, url_for, flash
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

app = Flask(__name__)
secret_key = os.environ.get("SECRET_KEY", "default_secret_key")
app.secret_key = secret_key


nimgs = 50

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if not os.path.isdir('Employee_Details'):
    os.makedirs('Employee_Details')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time,Date')

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
def totalreg():
    return len(os.listdir('static/faces'))


def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []


def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


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
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    current_date = date.today().strftime("%m_%d_%y")

    # Update the CSV file path with the current date
    csv_file_path = f'Attendance/Attendance-{current_date}.csv'

    df = pd.DataFrame(columns=['Name', 'Roll', 'Time', 'Date'])

    if os.path.isfile(csv_file_path):
        df = pd.read_csv(csv_file_path)

    if int(userid) not in list(df['Roll']):
        with open(csv_file_path, 'a') as f:
            f.write(f'\n{username},{userid},{current_time},{current_date}')

        # Create a directory for employee details
        employee_directory = 'Employee_Details'
        create_directory(employee_directory)

        # Create a file for each employee
        employee_file_path = f'{employee_directory}/{username}_{userid}.txt'
        try:
            if not os.path.exists(employee_file_path):
                with open(employee_file_path, 'w') as employee_file:
                    employee_file.write(f'Name: {username}\nEmployee ID: {userid}\nTime: {current_time}\n')
                    joining_date = date.today().strftime("%Y-%m-%d")
                    employee_file.write(f'Joining Date: {joining_date}\n')
                print(f"Employee details file created for {username}_{userid} at {employee_file_path}")
            else:
                with open(employee_file_path, 'a') as employee_file:
                    employee_file.write(f'Time: {current_time}\n')
                print(f"Information added to existing file for {username}_{userid} at {employee_file_path}")
        except Exception as e:
            print(f"Error writing to {employee_file_path}: {str(e)}")


def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)


@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/deleteuser/<username>', methods=['GET'])
def deleteuser(username):
    duser = 'static/faces/' + username

    # Delete the employee details file
    employee_file_path = f'Employee_Details/{username}.txt'
    try:
        os.remove(employee_file_path)
        print(f"Employee details file deleted for {username} at {employee_file_path}")
    except FileNotFoundError:
        print(f"Employee details file not found for {username}.")
    except Exception as e:
        print(f"Error deleting employee details file for {username}: {str(e)}")

    # Delete the images folder
    deletefolder(duser)

    if not os.listdir('static/faces/'):
        os.remove('static/face_recognition_model.pkl')

    try:
        train_model()
    except:
        pass

    return redirect(url_for('listusers'))

@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2,
                               mess='There is no employee faces in the folder')

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']

        # Check if the ID is already taken
        existing_ids = set([int(roll.split('_')[-1]) for roll in os.listdir('static/faces')])

        if int(newuserid) in existing_ids:
            flash('This ID is already in use. Please choose a different ID.', 'danger')
            names, rolls, times, l = extract_attendance()
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

        userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        i, j = 0, 0
        cap = cv2.VideoCapture(0)
        while 1:
            _, frame = cap.read()
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 5 == 0:
                    name = newusername+'_'+str(i)+'.jpg'
                    cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                    i += 1
                j += 1
            if j == nimgs*5:
                break
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

        # Update the model after adding a new employee
        try:
            train_model()
        except Exception as e:
            print(f"Error training model: {str(e)}")

    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/viewdetails/<username>/<userid>', methods=['GET'])
def view_details(username, userid):
    employee_directory = 'Employee_Details'
    employee_file_path = f'{employee_directory}/{username}_{userid}.txt'

    try:
        with open(employee_file_path, 'r') as employee_file:
            employee_details = employee_file.read()
            # Extracting name and id from the username and userid
            name, id = username, userid

            # Get the joining date from your data
            joining_date = get_joining_date(name, id)

            attendance_history = get_attendance_history(name, id)

            # Check if the employee was present today
            today_present = is_present_today(name, id)

            return render_template('viewdetails.html',
                                   employee_details=employee_details,
                                   name=name,
                                   id=id,
                                   today_present=today_present,
                                   attendance_history=attendance_history,
                                   joining_date=joining_date)
    except FileNotFoundError:
        error_message = f"Employee details not found for {username}_{userid}."
        return render_template('viewdetails.html', error_message=error_message)
    except Exception as e:
        error_message = f"An error occurred while fetching employee details: {str(e)}"
        return render_template('viewdetails.html', error_message=error_message)
def get_joining_date(name, id):
    employee_details_file = f'Employee_Details/{name}_{id}.txt'

    try:
        with open(employee_details_file, 'r') as employee_file:
            for line in employee_file:
                if line.startswith('Joining Date:'):
                    # Extract the joining date from the line
                    joining_date = line.split(': ')[1].strip()
                    return joining_date
        return "Joining date not available"
    except FileNotFoundError:
        print(f"Employee details file not found for {name}_{id}.")
        return "Joining date not available"
    except Exception as e:
        print(f"Error getting joining date for {name}_{id}: {str(e)}")
        return "Joining date not available"
def get_attendance_history(name, id):
    attendance_file_path = 'Attendance/'  # Assuming your attendance files are in the 'Attendance' folder

    try:
        attendance_history = []
        for file in os.listdir(attendance_file_path):
            if file.startswith('Attendance-') and file.endswith('.csv'):
                df = pd.read_csv(os.path.join(attendance_file_path, file))
                employee_attendance = df[df['Roll'] == int(id)]

                for index, row in employee_attendance.iterrows():
                    attendance_time = row['Time']  # Assuming 'Time' is the name of the column in your CSV file
                    attendance_date = file.split('-')[1].split('.')[0]  # Extract date from the filename

                    attendance_status = 'Present' if int(row['Roll']) == int(id) else 'Absent'

                    attendance_record = {'date': attendance_date, 'time': attendance_time, 'present': attendance_status}
                    attendance_history.append(attendance_record)

        return attendance_history

    except FileNotFoundError:
        print(f"Attendance files not found for {name}_{id}.")
        return []
    except Exception as e:
        print(f"Error getting attendance history for {name}_{id}: {str(e)}")
        return []


def is_present_today(name, id):

    today_date = date.today().strftime("%m_%d_%y")
    attendance_file_path = f'Attendance/Attendance-{today_date}.csv'

    try:
        df = pd.read_csv(attendance_file_path)
        today_present = (df['Name'] == name) & (df['Roll'] == int(id))
        return today_present.any()
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"Error checking attendance for {name}_{id}: {str(e)}")
        return False

if __name__ == '__main__':
    app.run(debug=True)
