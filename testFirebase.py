import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime
timestamp = 1545730073
dt_object = datetime.fromtimestamp(timestamp)
# current date and time


# Fetch the service account key JSON file contents
cred = credentials.Certificate('ebmproject-6eff3-firebase-adminsdk-vh93o-d6f42c2f18.json')
# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://ebmproject-6eff3.firebaseio.com/'
})

print(dt_object)
ref = db.reference('/')
ref.set({
        'device_id : 001 ':
            {
                'status': True
            }
        })
#update
# ref = db.reference('boxes')
# ref.update({
#     'color': 'red'
