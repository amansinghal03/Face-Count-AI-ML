import fractions
from operator import le
from pickle import NONE
from cv2 import VideoCapture
from matplotlib.pyplot import text
import pygame
import cv2
import numpy as np

video=VideoCapture(0)
facedetect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
pygame.init()

window = pygame.display.set_mode((1000,600))
pygame.display.set_caption("Face Count App")


img=pygame.image.load("img.jpg").convert()

start=True

while start:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            start=False
            pygame.quit()
    ret,frame= video.read()

    grey= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=facedetect.detectMultiScale(frame,1.3,5)

    for (x,y,w,h) in face:
        x1,y1=x+w, y+h
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.line(frame, (x,y), (x+30, y),(0,0,255), 6) #Top Left
        cv2.line(frame, (x,y), (x, y+30),(0,0,255), 6)

        cv2.line(frame, (x1,y), (x1-30, y),(0,0,255), 6) #Top Right
        cv2.line(frame, (x1,y), (x1, y+30),(0,0,255), 6)

        cv2.line(frame, (x,y1), (x+30, y1),(0,0,255), 6) #Bottom Left
        cv2.line(frame, (x,y1), (x, y1-30),(0,0,255), 6)

        cv2.line(frame, (x1,y1), (x1-30, y1),(0,0,255), 6) #Bottom right
        cv2.line(frame, (x1,y1), (x1, y1-30),(0,0,255), 6)



    imgRGB= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgRGB= np.rot90(imgRGB)
    imgRGB=pygame.surfarray.make_surface(imgRGB).convert()

    
    font= pygame.font.Font(None,50)
    text = font.render("Face Detction : {} Face Detected".format(len(face)),True,(255,255,255))
    
    window.fill((127, 255, 212))
    window.blit(img,(0,0))
    window.blit(imgRGB,(200,50))
    
    pygame.draw.rect(window,(21, 27, 84),(199,45,642,55),border_top_left_radius=10,border_top_right_radius=10)
    pygame.draw.rect(window,(21, 27, 84),(199,500,642,60),border_bottom_left_radius=10,border_bottom_right_radius=10)
    window.blit(text,(240,50))
    pygame.display.update()