# AI-Mouse
### Generate human-like mouse movements using LSTMs 


### Requirements:

`pip install -r requirements.txt`

### Instructions:

#### to train a model using your own hand movements (Recommended):

`python main.py`

#### During training:

1. place the mouse cursor at a random position inside the pygame window.
2. press the right mouse button and drag your mouse towards the red circle. 
3. when you reach the target, keep tracking the target for few seconds.
4. release the button.
5. repeat 1-5 for around 20 times.
6. when you feel you are done simply close the pygame window.
7. Wait for the training to finish and a testing pygame window will appear.

#### During evaluation: 

Press the left mouse button and the mouse will automatically move towards the target.

#### to test a pretrained model:

`python aiMove.py targetX targetY maxSteps`

for example:

`python aiMove.py 400 -500 100`

will try to move the cursor +100 pixels in the x axis and -500 in the y axis for a maximum of 100 steps
