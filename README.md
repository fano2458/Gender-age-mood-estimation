# Gender-age-mood-estimation
Gender, age, mood estimation with PyTorch and OpenCV

# Usage

Pretrained weights can be found in "weights" folder.

To run inference on image: 
```
python inference.py --source image --path "path to the image"
```
In order to save the predictions you can use "--save" argument. Save argument is set False by default. 

To run inference on webcam:
```
python inference.py
```

Sample test image          |  Sample image with predictions
:-------------------------:|:-------------------------:
![alt text](https://github.com/fano2458/Gender-age-mood-estimation/releases/download/data/im.png?raw=true) |  ![alt text](https://github.com/fano2458/Gender-age-mood-estimation/releases/download/data/im_new.png?raw=true)

