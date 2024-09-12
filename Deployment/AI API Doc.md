**this API detects Vitamin Deficiency by detecting the following signs on the Skin**
1. Light diseases and disorders of pigmentation 
1. Acne and Rosacea
1. Dermatitis
1. Hair Loss

**You can send a post request with an image of any suspected part of skin using the end point `/predict`**

Request:
```
[POST] http://localhost:5000/predict
```
Response:
```
{"class_id":#number,"class_name":"name of the condition and related vitamin deficiency"}
```

**how to use**
1. install Python3
2. install PyTorch
1. download the [development folder](https://github.com/VitaQuest-IEEE/AI/tree/main/Deployment)
1. download the [AI model](https://drive.google.com/file/d/1Yevhe-e1wXiHsCAUerink5DScSl0ACW2/view) and move to `/development/`
3. install the required packages
```
pip install -r requirements.txt
```
4. run the `deploy.py` file
5. send a post request using postman with the desired image
6. wait for the response