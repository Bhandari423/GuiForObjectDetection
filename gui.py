import os
import signal
import tkinter
import PIL.Image, PIL.ImageTk
import time
from tkinter import filedialog, messagebox
import datetime

import cv2
import pandas as pd

class Gui:
    window = tkinter.Tk()
    def __init__(self, window_title):
        self.window.title(window_title)
        self.video_path=0
        self.output_frame = 0
        self.pause_flag = False
        self.i =0
        self.width = 0
        self.height = 0
        self.pid = os.getpid()
        self.data=pd.DataFrame()

        #to upload video
        self.upload_button = tkinter.Button(self.window, text='Upload', command=self.upload_video)
        self.upload_button.pack()
        #to play video
        self.play_button = tkinter.Button(self.window, text='Play', command= self.play_video)
        self.play_button.pack()
        #to pause video
        self.pause_button = tkinter.Button(self.window, text='Pause', command= self.pause_video)
        self.pause_button.pack()
        #to download csv file
        self.download_button = tkinter.Button(self.window, text='Download File', command= self.download_file)
        self.download_button.pack()

        self.canvas = tkinter.Canvas(Gui.window, width = 800, height = 600, bg='black')
        self.canvas.pack()

        self.window.protocol('WM_DELETE_WINDOW', self.on_closing) 

        self.window.geometry("900x800")
        self.window.mainloop()
    
    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            os.kill(int(self.pid), signal.SIGTERM)     #SIGTERM can be used in windows
    
    def detect(self, path):
        cap = cv2.VideoCapture(path)
        open1=True
        if (cap.isOpened()== False):
            open1=False
            messagebox.showerror("Error!", "Error in Opening File")  

        # Object detection
        object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
        fr_no=0
        veh_lis=[]
        bounding_box=[]
        while(cap.isOpened()):
            ret, frame = cap.read()               
            if ret:
                fr_no+=1
                frame = cv2.resize(frame, (500, 500))    
                # Extract Region of interest
                roi = frame[280: 800,100: 750]

                # 1. Object Detection
                mask = object_detector.apply(roi)
                _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    # Calculate area and remove small elements
                    area = cv2.contourArea(cnt)
                    if area > 400:
                        x, y, w, h= cv2.boundingRect(cnt)
                        # detections.append([x, y, w, h])
                        bounding_box.append([x, y, w, h])
                        veh_lis.append([fr_no, bounding_box])
                        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
                cv2.imshow('frame',frame)
                key = cv2.waitKey(30)
                if key == 32:         #press spacebar to pause and press any key to resume
                    cv2.waitKey()
                elif key == ord('q'):   #press Q to exit
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
        col=['frame number','bounding box']
        self.data=pd.DataFrame(veh_lis,columns=col)

    def upload_video(self):
        cwd=os.getcwd()
        result = filedialog.askopenfile(initialdir=cwd, title='select file', filetypes=(('mp4 files', '*.mp4'), ('all files', "*"),))
        if result is None:
            pass
        self.video_path = result.name  
    
    def play_video(self):  
        self.pause_flag=False
        if self.pause_flag==False:           
            if self.i == 0:        
                self.detect(self.video_path)         
                
    def pause_video(self):
        self.pause_flag = True
        u_frame=cv2.cvtColor(self.output_frame, cv2.COLOR_BGR2RGB)
        
        photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(u_frame))
        self.canvas.create_image(200, 50, image = photo, anchor = tkinter.NW)
        
    def download_file(self):
        name = self.video_path.split('/')
        file_name = name[-1].split('.')
        print(file_name)
        if os.path.exists(file_name[0]):
            messagebox.showerror("Error!", "File with same name already exists!")  #message box appears when data get stored in csv file   
        os.makedirs(file_name[0])
        cwd=os.getcwd()
        self.data.to_csv(path_or_buf=os.path.join(file_name[0],'data.csv'),index=False)
        messagebox.showinfo("showinfo", "File Downloaded")  #message box appears when data get stored in csv file
    
# Create a window and pass it to the Application object
Gui("Moving Object Detector")