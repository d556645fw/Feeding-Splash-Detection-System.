import subprocess
import time
import mysql.connector
from mysql.connector import Error
from linebot import LineBotApi
from linebot.models import ImageSendMessage,TextSendMessage
test1 = ['python', 'C:\\Users\\user\\11157011\\seg\\seg_fish_ripple.py','--json_file','C:\\Users\\user\\11157011\\json\\seg_pingtung.json'
]
test2 = ['python', 'C:\\Users\\user\\11157011\\seg\\seg_fish_ripple.py','--json_file','C:\\Users\\user\\11157011\\json\\seg_pingtung.json'
]
test3 = ['python', 'C:\\Users\\user\\11157011\\seg\\seg_fish_ripple.py','--json_file','C:\\Users\\user\\11157011\\json\\seg_pingtung.json'
]
test4 = ['python', 'C:\\Users\\user\\11157011\\seg\\seg_fish_ripple.py','--json_file','C:\\Users\\user\\11157011\\json\\seg_pingtung.json'
]
test5 = ['python', 'C:\\Users\\user\\11157011\\seg\\seg_fish_ripple.py','--json_file','C:\\Users\\user\\11157011\\json\\seg_pingtung.json'
]
test6 = ['python', 'C:\\Users\\user\\11157011\\seg\\seg_fish_ripple.py','--json_file','C:\\Users\\user\\11157011\\json\\test1.json'
]
def start(connection,test,field):
    global processes
    
    if connection.is_connected():
        cursor = connection.cursor(buffered=True)
        connection.commit() 
        query="SELECT start_feeding FROM status WHERE id = %s;"
        val = (1,)
        cursor.execute(query,val)
        tmp = cursor.fetchone()
        status = tmp[0]
        
    if status == 1:
        globals()['pop'+str(field)] = subprocess.Popen(test)
        print('執行場域'+str(field))
        sql = "UPDATE status SET start_feeding = %s WHERE id = %s"
        val = (3, 1)
        cursor.execute(sql,val)
        connection.commit()
def get_line_id(connection):
    cursor = connection.cursor(buffered=True) 
    query= "SELECT line_id,token FROM line_bot WHERE id = %s;"
    val = (1,)
    cursor.execute(query,val)
    tmp = cursor.fetchone()
    line_id = tmp[0]
    api=tmp[1]
    return line_id,api    

def stop_check(connection,pop,line_bot_api,group_id):
    cursor = connection.cursor(buffered=True) 
    query= "SELECT start_feeding FROM status WHERE id = %s;"
    val = (1,)
    cursor.execute(query,val)
    tmp = cursor.fetchone()
    if tmp[0] ==0:
        now_time = time.localtime()#未來抓table
        now_time = time.strftime("%m/%d %H:%M:%S", now_time)
        line_bot_api = LineBotApi(line_bot_api)
        pop.kill()
        line_bot_api.push_message(group_id, TextSendMessage(text='於 '+now_time+' 結束投餌。'))
        sql = "UPDATE status SET start_feeding = %s WHERE id = %s"
        val = (4, 1)
        cursor.execute(sql,val)
        connection.commit()
        #cursor.close()
        #connection.close()
processes = []
host = '34.81.183.159'
connection1 = mysql.connector.connect(host=host,
                                    database='ar0DB',
                                    user='lab403',
                                    password='66386638')
connection2 = mysql.connector.connect(host=host,
                                    database='ar1DB',
                                    user='lab403',
                                    password='66386638')
connection3 = mysql.connector.connect(host=host,
                                    database='ar2DB',
                                    user='lab403',
                                    password='66386638')
connection4 = mysql.connector.connect(host=host,
                                    database='ar3DB',
                                    user='lab403',
                                    password='66386638')
connection5 = mysql.connector.connect(host=host,
                                    database='ar4DB',
                                    user='lab403',
                                    password='66386638')
connection6 = mysql.connector.connect(host=host,
                                    database='fishDB',
                                    user='lab403',
                                    password='66386638')
#api1,gpid1 = get_line_id(connection1)
#api2,gpid2 = get_line_id(connection1)
#api3,gpid3 = get_line_id(connection1)
#api4,gpid4 = get_line_id(connection1)
#api5,gpid5 = get_line_id(connection1)
api6,gpid6 = 'WDcBX1cVYAtLrUGcQJIUO7GFIY/UCGXVvHkiSaiHSkVYy3B5b4o0fKZHbPCRI6BLBGQohFrRnXkFUALt6J/4jy1oOt6KTLwJoRchY6esLlBv85L4aUYjLFhG8vXc9DS9x5Ji40/j7SFqrBwajhYwkgdB04t89/1O/w1cDnyilFU=','Cfd02ef1c35584e58651a368724143a39'
pop1,pop2,pop3,pop4,pop5,pop6 = None,None,None,None,None,None

while True:
    #start(connection1,test1,1)
    #start(connection2,test2,2)
    #start(connection3,test3,3)
    #start(connection4,test4,4)
    #start(connection5,test5,5)
    start(connection6,test6,6)
    time.sleep(2)
    #stop_check(connection1,pop1,api1,gpid1)
    #stop_check(connection2,pop2,api2,gpid2)
    #stop_check(connection3,pop3,api3,gpid3)
    #stop_check(connection4,pop4,api4,gpid4)
    #stop_check(connection5,pop5,api5,gpid5)
    stop_check(connection6,pop6,api6,gpid6)
    

   
for p in processes:
        p.wait()

if connection1.is_connected():
    connection1.close()
    print("MySQL connection 1 is closed")

if connection2.is_connected():
    connection2.close()
    print("MySQL connection 2 is closed")

if connection3.is_connected():
    connection3.close()
    print("MySQL connection 3 is closed")

if connection4.is_connected():
    connection4.close()
    print("MySQL connection 4 is closed")

if connection5.is_connected():
    connection5.close()
    print("MySQL connection 5 is closed")
if connection6.is_connected():
    connection6.close()
    print("MySQL connection 6 is closed")