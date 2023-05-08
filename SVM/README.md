# CS6140-Final-Project-SVM
## Environment
This model is designed to be run on Jupyter Notebook.

## Modules
<pre>
os
cv2
numpy
matplotlib
moviepy
sklearn
scipy
pickle
glob
</pre>
## Files structure

Create folders so the directoris are look like this:<br>
<pre>
--data
  |---test_images
  |---car
  |---neg
  |---person
  |---etc.
--videos
  |---video1.mp4
  |---video2.mp4
  |---etc.
--svc_pickle_car.p
--svc_pickle_person.p
--svm.ipynb
--svm.py
</pre>

Files in `videos` and `test_images` folder are used to test the model. Dataset is in `data` folder classified with different labels.

## How to operate
Run `import` code cell. <br>
### Train the model
1. Change `object` variable in `parameter` cell to corresponding folder name you want to train. Run `parameters` cell.<br>
2. Run code cells from `load dataset` to `save data` in order.

### Test the model
1. Change `object` variable in `parameter` cell to corresponding folder name you want to test. Run `parameters` cell.<br>
2. Run `features extract functions` cell. <br>
3. Run code cells from `load data` to `test videos` in order. `test images` cell is to test images in `test_images` folder. `test videos` is to test videos in `videos` folder.
