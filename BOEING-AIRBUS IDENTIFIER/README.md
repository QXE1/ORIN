AIRBUS/BOEING COMMERCIAL AIRLINER IDENTIFIER

This Project identifies specific models of Boeing and Airbus commercial airliners and their specific model (IE B737, A320, 747, etc).

## How it works
    This program is based upon NVIDIA ImageNet and is trained on a predetermined set of images of all Boeing and Airbus commercial airliner models. If a photo of a Boeing or Airbus plane is fed into the program, the model will determine what specific plane it is and output the type of aircraft in the terminal.

## Training the Model
1. Run the command "python3 PlaneNet.py" in terminal and wait for it to train
2. Optionally, you can modify the number of epochs in the code, with more epochs leading to a higher accuracy, but it will be more time consuming. 

## How to use it
1. Drag a photo of a Boeing or Airbus plane into AIRBUS/BOEING IDENTIFIER
2. Using your orin, enter the command "python3 PlaneIdent.py [YOUR PHOTO.jpg]"
3. Wait for the program to identify the plane in terminal

Video demonstration cound be found here:
