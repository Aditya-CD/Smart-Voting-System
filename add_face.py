import cv2
import pickle
import numpy as np
import os
import customtkinter as ctk
from tkinter import messagebox

# Ensure data directory exists
if not os.path.exists('data/'):
    os.makedirs('data/')

# Function to capture faces and save data
def start_face_capture():
    def get_aadhar_number():
        # Create a temporary dialog box for Aadhar input
        dialog = ctk.CTkToplevel()
        dialog.title("Enter Aadhar Number")
        dialog.geometry("400x200")
        dialog.resizable(False, False)

        # Submit function
        def submit():
            aadhar_number = entry.get().strip()
            if not aadhar_number.isdigit() or len(aadhar_number) != 12:
                messagebox.showerror("Invalid Input", "Aadhar number must be a 12-digit numeric value.")
            else:
                dialog.aadhar_number = aadhar_number
                dialog.destroy()

        # UI for the dialog box
        label = ctk.CTkLabel(dialog, text="Enter Your Aadhar Number", font=("Arial", 18, "bold"))
        label.pack(pady=20)

        entry = ctk.CTkEntry(dialog, placeholder_text="Aadhar Number (12 digits)", font=("Arial", 14))
        entry.pack(pady=10, padx=20, fill="x")

        submit_button = ctk.CTkButton(dialog, text="Submit", command=submit, font=("Arial", 14), fg_color="#1f6aa5", hover_color="#144d73")
        submit_button.pack(pady=20)

        dialog.aadhar_number = None  # Default if user closes the dialog
        dialog.grab_set()
        dialog.wait_window()
        return dialog.aadhar_number

    # Call the Aadhar dialog box
    aadhar_number = get_aadhar_number()
    if not aadhar_number:
        return  # Exit if no valid input

    # Proceed with face capture
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces_data = []
    frames_total = 51
    capture_after_frame = 2
    i = 0

    while True:
        ret, frame = video.read()
        if not ret:
            messagebox.showerror("Error", "Failed to access the webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(faces_data) <= frames_total and i % capture_after_frame == 0:
                faces_data.append(resized_img)
            i += 1
            cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

        cv2.imshow('Face Capture', frame)
        k = cv2.waitKey(1)
        if k == ord('q') or len(faces_data) >= frames_total:
            break

    video.release()
    cv2.destroyAllWindows()

    # Save captured data
    if len(faces_data) > 0:
        faces_data = np.asarray(faces_data)
        faces_data = faces_data.reshape((frames_total, -1))

        save_data(aadhar_number, faces_data)
        messagebox.showinfo("Success", "Faces captured and saved successfully!")
    else:
        messagebox.showwarning("No Data", "No faces were captured.")

# Function to save data
def save_data(name, faces_data):
    if 'names.pkl' not in os.listdir('data/'):
        names = [name] * faces_data.shape[0]
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)
    else:
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        names = names + [name] * faces_data.shape[0]
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)

    if 'faces_data.pkl' not in os.listdir('data/'):
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open('data/faces_data.pkl', 'rb') as f:
            faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces, f)

# Main GUI setup with customtkinter
def main():
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title("Face Capture Application")
    root.geometry("500x400")

    header_label = ctk.CTkLabel(root, text="Face Capture System", font=("Arial", 24, "bold"))
    header_label.pack(pady=20)

    desc_label = ctk.CTkLabel(root, text="Click below to start capturing your face data.", font=("Arial", 16))
    desc_label.pack(pady=10)

    start_button = ctk.CTkButton(root, text="Start Face Capture", command=start_face_capture, font=("Arial", 14), fg_color="#1f6aa5", hover_color="#144d73")
    start_button.pack(pady=20)

    exit_button = ctk.CTkButton(root, text="Exit", command=root.destroy, font=("Arial", 14), fg_color="red", hover_color="#8b0000")
    exit_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
