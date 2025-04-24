   #!/usr/bin/python

# Import necessary packages
import face_recognition
import cv2
import pickle
import time
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import RPi.GPIO as GPIO
import threading
import os
import json
import sys
from SimpleMFRC522 import SimpleMFRC522


# Fix Qt platform plugin error by setting environment variable
# This needs to be done before any OpenCV import/usage
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Force using X11 instead of Wayland

# GPIO setup
RELAY = 17
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY, GPIO.OUT)
GPIO.output(RELAY, GPIO.HIGH)  # Ensure door is locked at startup

# Configuration
ENCODINGS_FILE = "encodings.pickle"
CASCADE_FILE = "haarcascade_frontalface_default.xml"
RFID_DB_FILE = "rfid_users.json"

FRAME_WIDTH = 400  # Reduced from 500 for faster processing
UNLOCK_DURATION = 5.0  # Seconds to keep door unlocked
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to consider a match

# Load face encodings and detector
print("[INFO] Loading encodings and face detector...")
try:
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.loads(f.read())
except FileNotFoundError:
    print(f"[ERROR] Could not find encodings file: {ENCODINGS_FILE}")
    print("[INFO] Make sure to run the face training script first")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Failed to load encodings: {e}")
    sys.exit(1)
try:
    detector = cv2.CascadeClassifier(CASCADE_FILE)
    if detector.empty():
        print(f"[ERROR] Failed to load cascade classifier: {CASCADE_FILE}")
        sys.exit(1)
except Exception as e:
    print(f"[ERROR] Failed to load face detector: {e}")
    sys.exit(1)

# Initialize RFID reader
reader = SimpleMFRC522()

# Load RFID users database
def load_rfid_users():
    if os.path.exists(RFID_DB_FILE):
        try:
            with open(RFID_DB_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load RFID database: {e}")
            return {}
    else:
        print("[INFO] RFID database not found. Creating empty database.")
        return {}

rfid_users = load_rfid_users()


# Track door state
door_lock_timer = None
door_unlocked = False

def lock_door():
    """Lock the door and update state"""
    global door_unlocked
    GPIO.output(RELAY, GPIO.HIGH)
    door_unlocked = False
    print("[SECURITY] Door locked")

def unlock_door(duration=UNLOCK_DURATION, user="System"):

    """Unlock the door for a specified duration"""
    global door_lock_timer, door_unlocked
   
    # Cancel any existing timer
    if door_lock_timer is not None:
        door_lock_timer.cancel()
   
    # Unlock the door
    GPIO.output(RELAY, GPIO.LOW)
    door_unlocked = True
    print(f"[ACCESS] Door unlocked by {user}")

   
    # Set timer to lock door after duration
    door_lock_timer = threading.Timer(duration, lock_door)
    door_lock_timer.daemon = True
    door_lock_timer.start()

def process_frame(frame, detector, data):
    """Process a video frame and return results"""
    # Resize frame to speed up processing
    frame = imutils.resize(frame, width=FRAME_WIDTH)
   
    # Convert frame to grayscale for detection and RGB for recognition
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   
    # Detect faces
    rects = detector.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)

   
    # Reorder coordinates for face_recognition
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
   
    # Get face encodings
    encodings = face_recognition.face_encodings(rgb, boxes)
   
    # Process each detected face
    names = []
    access_granted = False
    recognized_name = "Unknown"

   
    for encoding in encodings:
        # Compare with known faces
        matches = face_recognition.compare_faces(
            data["encodings"],
            encoding,
            tolerance=CONFIDENCE_THRESHOLD
        )
        name = "Unknown"
       
        # Check for matches
        if True in matches:
            # Count matched faces
            matched_idxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
           
            # Count matches for each person
            for i in matched_idxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
           
            # Get the person with most matches
            name = max(counts, key=counts.get)
            recognized_name = name

            access_granted = True
       
        names.append(name)
   
    # Return processing results
    return frame, boxes, names, access_granted, recognized_name

def check_rfid():
    """Check for RFID cards and authenticate"""
    global rfid_users
   
    try:
        # Non-blocking RFID read (available on some RFID libraries)
        # If your library doesn't support non-blocking reads,
        # this will need to be run in a separate thread
        id = reader.read_id_no_block()
       
        if id:
            id_str = str(id)
            if id_str in rfid_users:
                user = rfid_users[id_str]
                print(f"[RFID] Access granted for {user['name']}")
                unlock_door(user=f"RFID: {user['name']}")
                return True, user['name']
    except Exception as e:
        print(f"[ERROR] RFID read error: {e}")
   
    return False, None

# Thread for RFID scanning
class RFIDThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = True
        self._stop_event = threading.Event()
   

   def stop(self):
        self._stop_event.set()
   
    def stopped(self):
        return self._stop_event.is_set()
   
    def run(self):
        print("[INFO] Starting RFID monitoring thread")
        while not self.stopped():
            try:
                # Read RFID
                id, text = reader.read()
                id_str = str(id)
               
                if id_str in rfid_users:
                    user = rfid_users[id_str]
                    print(f"[RFID] Access granted for {user['name']}")
                    unlock_door(user=f"RFID: {user['name']}")
                else:
                    print(f"[RFID] Unknown card: {id_str}")
               
                # Prevent reading the same card multiple times in quick succession
                time.sleep(2)
               
            except Exception as e:
                print(f"[ERROR] RFID thread error: {e}")
                time.sleep(1)


# Video stream type
use_cv2_directly = False

# Initialize video stream
print("[INFO] Starting video stream...")
try:
    vs = VideoStream(src=0).start()
    # vs = VideoStream(usePiCamera=True).start()  # Uncomment for Pi camera
    time.sleep(1.0)  # Reduced warm-up time
except Exception as e:
    print(f"[ERROR] Failed to start video stream: {e}")
    print("[INFO] Trying with different backend...")
    # Try with direct OpenCV VideoCapture as fallback
    vs = cv2.VideoCapture(0)
    use_cv2_directly = True
    if not vs.isOpened():
        print("[ERROR] Could not open video device")
        sys.exit(1)

# Start FPS counter
fps = FPS().start()

# Start RFID thread
rfid_thread = RFIDThread()
rfid_thread.start()

# Main processing loop
print("[INFO] Facial recognition and RFID security system running...")

try:
    while True:
        # Grab frame from video stream
        if not use_cv2_directly:
            frame = vs.read()
            if frame is None:
                continue
        else:  # OpenCV VideoCapture
            ret, frame = vs.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                continue
       
        # Process the frame
        frame, boxes, names, access_granted, recognized_name = process_frame(frame, detector, data)

       
        # Handle door control based on recognition results
        if access_granted and not door_unlocked:
            # If recognized face, unlock door
            unlock_door(user=f"Face: {recognized_name}")

       
        # Draw results on frame
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # Set color based on access (green for known, red for unknown)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)


            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
           
            # Draw name
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, color, 2)
       
        # Display status information on frame
        status = "UNLOCKED" if door_unlocked else "LOCKED"
        cv2.putText(frame, f"Door: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
       
        # Display authentication method
        cv2.putText(frame, "Auth: Face & RFID", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
       
        # Display the frame
        cv2.imshow("Facial Recognition & RFID Security", frame)

       
        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
       
        # Update FPS counter
        fps.update()

except KeyboardInterrupt:
    print("[INFO] Interrupted by user")
except Exception as e:
    print(f"[ERROR] An unexpected error occurred: {e}")

finally:
    # Clean up
    print("[INFO] Cleaning up...")
   
    # Stop RFID thread
    if rfid_thread.is_alive():
        rfid_thread.stop()
        rfid_thread.join(timeout=1.0)

   
    # Stop FPS counter and display stats
    fps.stop()
    print(f"[INFO] Elapsed time: {fps.elapsed():.2f} seconds")
    print(f"[INFO] Approx. FPS: {fps.fps():.2f}")
   
    # Ensure door is locked before exit
    GPIO.output(RELAY, GPIO.HIGH)
   
    # Release resources
    cv2.destroyAllWindows()
   
    # Properly stop the video stream based on its type
    if use_cv2_directly:
        vs.release()  # OpenCV VideoCapture
    else:
        vs.stop()     # VideoStream
   
    # Cancel any pending timer
    if door_lock_timer is not None:
        door_lock_timer.cancel()
   
    print("[INFO] System shutdown complete")