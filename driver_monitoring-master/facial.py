import cv2
import time
import pygame
import geocoder
import os
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime
from facial_tracking.facialTracking import FacialTracker
import facial_tracking.conf as conf

# Initialize pygame for sound alerts
pygame.mixer.init()

# Load alert sound
ALERT_SOUND = os.path.join(os.path.dirname(__file__), "music.wav")  # Ensure the file exists

# Email Configuration (Use environment variables for security)
EMAIL_SENDER = "fffinalyearproject@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")  # Set this in your environment variables
EMAIL_RECEIVER = "laksh17036@gmail.com"

def play_alert_sound():
    """Plays the alert sound."""
    pygame.mixer.music.load(ALERT_SOUND)
    pygame.mixer.music.play()

def get_location():
    """Gets the current city name using geolocation."""
    try:
        g = geocoder.ip('me')
        return g.city if g.city else "Unknown Location"
    except Exception as e:
        print(f"Error getting location: {e}")
        return "Unknown Location"

def send_email_alert(image_path, alert_type, timestamp, location):
    """Sends an email alert with an image attachment."""
    if not EMAIL_PASSWORD:
        print("❌ Email not sent: Missing App Password in environment variables.")
        return
    
    subject = f"🚨 Driver Alert: {alert_type} Detected!"
    body = f"""
    Alert Type: {alert_type}
    Time: {timestamp}
    Location: {location}

    Immediate attention is required.
    """

    msg = EmailMessage()
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = subject
    msg.set_content(body)

    # Attach image
    with open(image_path, "rb") as img:
        msg.add_attachment(img.read(), maintype="image", subtype="jpeg", filename="alert.jpg")

    # Send email
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print("✅ Email alert sent successfully!")
    except Exception as e:
        print(f"❌ Error sending email: {e}")

def capture_alert_image(frame):
    """Captures and saves an image when an alert is triggered."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image_path = f"alert_{timestamp}.jpg"
    cv2.imwrite(image_path, frame)
    return image_path, timestamp

def main():
    cap = cv2.VideoCapture(conf.CAM_ID)
    cap.set(3, conf.FRAME_W)
    cap.set(4, conf.FRAME_H)
    facial_tracker = FacialTracker()
    ptime = 0
    location = get_location()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        facial_tracker.process_frame(frame_rgb)

        # Get current time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Calculate FPS
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        # Flip frame for correct orientation
        frame = cv2.flip(frame, 1)

        # Display FPS, Location, and Time
        cv2.putText(frame, f'FPS: {int(fps)}', (30, 30), 0, 0.6, conf.TEXT_COLOR, 1, lineType=cv2.LINE_AA)
        cv2.putText(frame, f'Location: {location}', (30, 60), 0, 0.6, conf.TEXT_COLOR, 1, lineType=cv2.LINE_AA)
        cv2.putText(frame, f'Time: {timestamp}', (30, 90), 0, 0.6, conf.TEXT_COLOR, 1, lineType=cv2.LINE_AA)

        alert_triggered = False
        alert_type = ""

        # Check if eyes are closed or yawning
        if facial_tracker.eyes_closed:
            alert_triggered = True
            alert_type = "Drowsiness (Eyes Closed)"
        elif facial_tracker.yawn_detected:
            alert_triggered = True
            alert_type = "Yawning"

        if alert_triggered:
            play_alert_sound()
            cv2.putText(frame, f"ALERT: {alert_type}!", (30, 130), 0, 0.8, conf.WARN_COLOR, 2, lineType=cv2.LINE_AA)

            # Capture image and send email alert
            image_path, timestamp = capture_alert_image(frame)
            send_email_alert(image_path, alert_type, timestamp, location)

        # Debugging
        print(f"Eyes Closed: {facial_tracker.eyes_closed}, Yawning: {facial_tracker.yawn_detected}")

        cv2.imshow('Driver Monitoring System', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
