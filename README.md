Counting feature added to The AI Guy's code. 
The original repository from The AI Guy link is https://github.com/theAIGuysCode/yolov4-deepsort
and the user's account link is : https://github.com/theAIGuysCode 

Counting feature added to make a people flux control.
Changes we're made to object_control.py file so it generates a csv file with people's entrances and exits after processing a video.

To run this repository, use The AI Guy's colab link [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zmeSTP3J5zu2d5fHgsQC06DyYEYJFXq1?usp=sharing)
and change the object_tracker.py to the code in this repository. Other option is to change the original Google Colab cell number 2 git clone link to this https://github.com/DanielChiyo/People-counter-with-YOLOv4-and-Deep-SORT instead.

Result example: https://drive.google.com/file/d/1zNLDMZcsynwPAOZ2MIrNBxTTYb6v26-D/view?usp=sharing

*Deep SORT's Amax (number of frames a person's features keeps saved before freeing the ID) not being considered, it might cause counting errors processing longer videos.

The original vidoe is AVG-TownCentre from https://motchallenge.net/ 
Adobe Premiere was used to increase contrast and sharpness (both increased to 32) of the video.
The original video cut using Premiere : https://drive.google.com/file/d/1_h1gqlhcF5iat7uqKbWkqppuQmIeiYjo/view?usp=sharing
The increased contrast and sharpness video : https://drive.google.com/file/d/1CRBA46BGLv4mBw3wsDoMyY4h_GM_MFJa/view?usp=sharing

Besides the outputed video shown in "Result example" there will also be a CSV file with the data (Time in seconds).
![image](https://user-images.githubusercontent.com/26650300/126177680-bdfe3ec0-de45-48a1-ba21-b43e2e5e7db4.png)


To try different videos you have to make changes in step 6
![image](https://user-images.githubusercontent.com/26650300/126179394-59f8e610-02ed-48a6-9116-0252e47572aa.png)
Upload the video you want to process and change the parameter of --video to the new video's path.
You can upload your video and check the path in the left side of your google colaboratory:

![image](https://user-images.githubusercontent.com/26650300/126179725-5e139c13-1c26-4671-a60d-569f9b83aed7.png)
