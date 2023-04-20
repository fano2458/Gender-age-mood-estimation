# Gender-age-mood-estimation
Gender, age, mood estimation with PyTorch and OpenCV

# Usage
To run inference on image: 
```
python inference.py --source image --path "path to the image"
```
In order to save the predictions you can use "--save" argument. Save argument is set False by default. 

To run inference on webcam:
```
python inference.py --source 0
```

<br>Sample test image:
<br>
![alt text](https://github.com/fano2458/Gender-age-mood-estimation/releases/download/data/im.png?raw=true)
<br>Sample image with detections:
<br>
![alt text](https://github.com/fano2458/Gender-age-mood-estimation/releases/download/data/im_new.png?raw=true)
