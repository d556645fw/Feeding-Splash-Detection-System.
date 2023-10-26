import cv2
import csv
import math
from PIL import Image
import multiprocessing
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import tensorflow as tf
import os
import io
import sched
import time
import requests
import signal
import argparse
import mysql.connector
from mysql.connector import Error
from linebot import LineBotApi
from linebot.models import ImageSendMessage,TextSendMessage
from imgurpython import ImgurClient
from imgurpython.helpers.error import ImgurClientError
import sys
global connection,cursor
import json
import build_segformer
#config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True,per_process_gpu_memory_fraction=0.3))
#sess = tf.compat.v1.Session(config=config)
os.environ["CUDA_VISIBLE_DEVICES"]='-1'         
def SegFormerPreprocessing(img):#標準化
    mean   = np.array([0.485, 0.456, 0.406])
    std    = np.array([0.229, 0.224, 0.225])
    img    = (img - mean)/std
    return img
def find_splash(img,flag):#定時測水花面積
    global angel
    def find_ripple(frame,flag):
        frame = cv2.resize(frame,(640,480))
        img = frame.copy()
        img2 = frame.copy()
        img = np.transpose(SegFormerPreprocessing(img[np.newaxis,...]),(0,3,1,2))
        

        x_pred = model.predict(img).logits
        x_pred = tf.transpose(x_pred,(0,2,3,1))
        x_pred = tf.image.resize(x_pred,(480,640))
        x_pred = tf.nn.softmax(x_pred)
        x_pred = np.where(x_pred>0.5,x_pred,0)
        print(np.sum(x_pred[0,:,:,1]))
        if flag == 0:
            maxArea = np.sum(x_pred[0,:,:,1]) 
        elif flag == 1:
            G = (x_pred[0,:,:,1]*255).astype(np.uint8)
            img2[:,:,0] = np.clip(img2[:,:,0]+G,0,255)
            
            contours, hierarchy = cv2.findContours(G,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            maxArea = 0
            for idx,aContour in enumerate(contours):
                a = cv2.contourArea(aContour)
                if a > maxArea:
                    maxArea = a
            contours = sorted(contours,key = lambda c:cv2.contourArea(c), reverse = True)
            lg = np.zeros(G.shape)
            lg = cv2.drawContours(lg, contours, 0, 255, cv2.FILLED)
            
            frame = frame.astype(np.float32)
            frame[:,:,0] = frame[:,:,0]+lg
            frame = np.clip(frame,0,255).astype(np.uint8)
        return maxArea,frame
    bait=[]#投餌機位置
    bound=[]#投餌邊界
    x=[]#投餌點
    y=[]#投餌角度

    i=0
    print('------------------------------------------')

    area,frame=find_ripple(img,flag)
    cv2.imshow('1',frame)
    cv2.waitKey(100)
    return area



def area_warning(area,now_time_sec,start_time,threshold_1,threshold_2,max_feed,now_feed,now_time,field):#threshold_1 時間 threshold_2 投餌量
    global fish_area,sec,count
    sql = "select start_feeding from status WHERE id = %s;"
    val = (int(field),)
    cursor.execute(sql,val)
    stop = cursor.fetchone()
    stop = stop[0]
    print("最大水花面積 : ",fish_area)
    Elapsed_time = now_time_sec - start_time  #目前投餌時間
    print('投餌經過時間 : ',Elapsed_time)
    if Elapsed_time <= threshold_1 :
        if area > fish_area:
            fish_area = area
        elif area < fish_area * 0.15:
            sec = 1
            count +=1
            if count >=5:
                push_img(now_time,3)
                count = 0
                sec = 3
                return 1
            return 0
        count = 0
        sec = 3
    elif Elapsed_time > threshold_1  and Elapsed_time < threshold_2:
        if area < fish_area*0.3 or area>fish_area*3: #判斷是否吃餌  
            sec = 1
            count +=1
            if count >=3:
                push_img(now_time,0)
                count = 0
                sec = 3
                return 1
            return 0
        count = 0
        sec = 3
    elif Elapsed_time > threshold_2 and Elapsed_time < max_feed:
        if area < fish_area*0.6:
            push_img(now_time,1)#傳圖片至line
            return 1
    elif Elapsed_time >= max_feed:
        print('已達指定投餌量')
        return 2
      
    return 0

def stop_program(field,flag,now_time,area):
    if flag ==1:
        print('水花異常')
        sql = "UPDATE status SET start_feeding = %s WHERE id = %s"
        val = (2,int(field))
        cursor.execute(sql,val)
        connection.commit()
    if flag ==2:
        sql = "UPDATE status SET start_feeding = %s WHERE id = %s"
        val = (0,int(field))
        cursor.execute(sql,val)
        connection.commit()
        #push_img(now_time,4)
        print('投餌量達標，停止投餌。')
'''
def img_download(field):
    global cursor
    found_result = False
    while found_result ==  False:
        sql = "select data from frames WHERE id = %s;"
        val = (int(field),)
        cursor.execute(sql,val)                                                                       
        tmp = cursor.fetchone()
        connection.commit()
        if tmp is not None:
            found_result = True
            original_img = tmp[0]
            original_img = Image.open(io.BytesIO(original_img))
        else:
            cursor = connection.cursor(buffered=True)                                                                              
    return original_img
'''
def img_download(file_path):
    original_img = cv2.imread(file_path)                                                                             
    return original_img

def file_exist(field,threshold_1,threshold_2,max_feed,flag):
    global now_feed,start_time,now_time_sec,folder_path,i,sec
    flag_stop=0
    #try:
                                     
    if connection.is_connected():
        
        if i >=874:#1124#1249
            i=874
        filename = f"shelter-{i:08d}.png"
        file_path = os.path.join(folder_path, filename)
        print(i)
        frame = img_download(file_path)
        i+=13*sec
        #frame = cv2.cvtColor(np.asarray(original_img), cv2.COLOR_RGB2BGR)
 
        cv2.imwrite(os.path.join(".",'{}-output.png'.format(csv_name)), frame)

        area=find_splash(frame,flag)
        
        now_time = time.localtime()#時間
        now_time_24hr = time.strftime("%m/%d %H:%M:%S", now_time)
        
        now_time_sec = time.strftime("%I_%M_%S", now_time)#未來抓table
        now_time_sec = int(now_time_sec[0:2])*3600+int(now_time_sec[3:5])*60+int(now_time_sec[6:8])
        now_time_sec = int(now_time_sec)
        
        print('本次搶食水花面積 : ',area)
        #水花異常警告
        flag_stop = area_warning(area,now_time_sec,start_time,threshold_1,threshold_2,max_feed,now_feed,now_time_24hr,field)
        now_feed +=10
        #print('目前投餌量: ',now_feed)  
        
        stop_program(field,flag_stop,now_time_24hr,area)
        
def push_img(time,flag):
    #imgur api
    client_id = 'bd774e291c20e91'
    client_secret = '47647410f933b0a915f5dd53c7d20b4a7283f454'
    access_token = '7db727aec924dc06e2b2a18f4890b4440631bc4a'
    refresh_token = '23b80d97504769d6a4f7a8792cfb96e18310cb37'
    #line api
    line_bot_api = LineBotApi(line_api)
    user_id = str(line_id)
    # Note since access tokens expire after an hour, only the refresh token is required (library handles autorefresh)
    client = ImgurClient(client_id, client_secret, access_token, refresh_token)
    if flag == 0 :
        image_upload = client.upload_from_path(os.path.join(".",'{}-output.png'.format(csv_name)),anon=False)
        # 取得 Imgur 圖片的網址
        image_url = image_upload['link']
        image_message = ImageSendMessage(
            original_content_url=image_url,
            preview_image_url=image_url
        )
        # 傳送圖片訊息
        line_bot_api.push_message(user_id, TextSendMessage(text='於 '+time+' 魚群搶食水花發生異常 , 請決定是否繼續投餌，停止投餌請輸入0。'))
        line_bot_api.push_message(user_id, TextSendMessage(text='若未輸入將繼續投餌。'))
        line_bot_api.push_message(user_id, image_message)
    elif flag == 1:
        image_upload = client.upload_from_path(os.path.join(".",'{}-output.png'.format(csv_name)),anon=False)
        # 取得 Imgur 圖片的網址
        image_url = image_upload['link']
        image_message = ImageSendMessage(
            original_content_url=image_url,
            preview_image_url=image_url
        )
        # 傳送圖片訊息
        line_bot_api.push_message(user_id, TextSendMessage(text='於 '+time+' 魚群搶食水花發生異常 , 請決定是否繼續投餌，停止投餌請輸入0。'))
        line_bot_api.push_message(user_id, TextSendMessage(text='目前餵食量已達可提早停止範圍。'))
        line_bot_api.push_message(user_id, TextSendMessage(text='若未輸入將繼續投餌。'))
        line_bot_api.push_message(user_id, image_message)
    elif flag == 2:
        line_bot_api.push_message(user_id, TextSendMessage(text='於 '+time+' 結束投餌。'))
    elif flag == 3:
        image_upload = client.upload_from_path(os.path.join(".",'{}-output.png'.format(csv_name)),anon=False)
        # 取得 Imgur 圖片的網址
        image_url = image_upload['link']
        image_message = ImageSendMessage(
            original_content_url=image_url,
            preview_image_url=image_url
        )
        # 傳送圖片訊息
        line_bot_api.push_message(user_id, TextSendMessage(text='於 '+time+'魚群搶食水花發生異常，未發現投餌水花, 請決定是否繼續投餌，停止投餌請輸入0。'))
        line_bot_api.push_message(user_id, TextSendMessage(text='若未輸入將繼續投餌。'))
        line_bot_api.push_message(user_id, image_message)
    elif flag ==4:
        line_bot_api.push_message(user_id, TextSendMessage(text='於 '+time+' 結束投餌。'))

            



parser = argparse.ArgumentParser(description='測試')
parser.add_argument("--json_file", required=True, help="Path to the JSON file")
args = parser.parse_args()

with open(args.json_file, "r") as file:
        data = json.load(file)
        

threshold_1 = data["t1"]
threshold_2 = data["t2"]
max_feed = data["max_feed"]#資料庫
weight_name = data["weight"]
field = data["field"]
flag = data["flag"]#0 午 1 鱸 
host_ip = data["host_ip"]
db = data["database"]
db_user = data["user"]
db_password = data["password"]
bst_model=data["bst_model"]
csv_name=data["csv_name"]
line_api = data["line_api"]

angel=90.0
stop=0
lg = np.zeros((640,480))


fish_area = 0
now_feed = 0
count = 0
i = 0
start_time = time.localtime()
start_time = time.strftime("%I_%M_%S", start_time)#未來抓table
start_time = int(start_time[0:2])*3600+int(start_time[3:5])*60+int(start_time[6:8])
now_time_sec = int(start_time)

connection = mysql.connector.connect(host=host_ip,
                                         database=db,
                                         user=db_user,
                                         password=db_password)

cursor = connection.cursor(buffered=True) 
sql = "select fetch_interval from decision WHERE id = %s;"
val = (int(field),)
cursor.execute(sql,val)
#sec = cursor.fetchone()[0]
sec = 3

sql = "select line_id from line WHERE camera_id = %s;"
val = (int(field),)
cursor.execute(sql,val)
line_id=cursor.fetchone()[0]
folder_path = "C:\\Users\\user\\11157011\\seg\\test\\shelter"

models={"SegFormer-b0":lambda : build_segformer.build_segformer('b0','SegFormer-b0'),
        "SegFormer-b1":lambda : build_segformer.build_segformer('b1','SegFormer-b1'),
        "SegFormer-b2":lambda : build_segformer.build_segformer('b2','SegFormer-b2'),
        "SegFormer-b3":lambda : build_segformer.build_segformer('b3','SegFormer-b3'),
        "SegFormer-b4":lambda : build_segformer.build_segformer('b4','SegFormer-b4'),
        "SegFormer-b5":lambda : build_segformer.build_segformer('b5','SegFormer-b5')}
model_name     ='SegFormer-b1'
model=models[model_name]()
model.load_weights(os.path.join('.','seg','weight','{}-{}-{}.h5'.format(weight_name,model.name,bst_model)))
#model.load_weights(os.path.join('.','weight','{}-{}-{}.h5'.format(weight_name,model.name,bst_model)))
while True:
    file_exist(str(field),threshold_1,threshold_2,max_feed,flag)
    time.sleep(int(sec))
        