import cv2
import numpy as np
#K-Nearest Neighbour(en yakın komşu) algoritması kullanılarak resimdeki sayı verilerini eğiten ve verileri çizerek test ettiğimiz program

img = cv2.imread(r"opencv\video_ve_resimler\digits.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Data hazırlanıyor. (Amaç resimdeki 0'dan 9'a kadarki kısımları ayırmak)
"""
vsplit(yatay bölme) ile 50'ye(y değeri) bölünen her değeri hsplit(dikey bölme) ile 100'e(x değeri) böldük.
Elimizde tek bir veri kaldı.Böylece resimdeki her bir veriye ulaşılmış oldu.
"""
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
x = np.array(cells)          #veriler listeye dönüştürüldü.

#Elimizdeki 5000 verinin 4500 tanesini öğretip kalanını test ettik.Amaç işlemin doğruluğunu ölçmek
#reshape ile verileri yukarıdan aşağıya sıraladık.
train = x[:,:90].reshape(-1,400).astype(np.float32)      #400:20x20'lik alan
test = x[:,90:100].reshape(-1,400).astype(np.float32)   

k = np.arange(10)
#response:cevapları(gerçekte ne oldukları), burada cevapları döndürdük.
train_response = np.repeat(k,450).reshape(-1,1)     
test_response = np.repeat(k,50).reshape(-1,1)   

# Data kayıt etme işlemi
np.savez("knn_data.npz", train_data = train,
         train_label = train_response)

# Datayı okuma işlemi
with np.load("knn_data.npz") as data:
    print(data.files)
    train = data["train_data"]
    train_responses = data["train_label"]
    
#Eğitim işlemi gerçekleştirildi.
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_response)
ret, results, neighbours, distance = knn.findNearest(test, 5) 

matches = test_response == results
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / results.size
print("Accuracy : ",accuracy)

#Eğitilen modeli kayıt etmek
knn.save('KNN_Trained_Model.yml')
#Modeli okuma
knn = cv2.ml.KNearest_load('KNN_Trained_Model.yml')

#Programı kendimizin çizerek test ettiğimiz kısım(Sol mouse'a çift tıklayınca çizileni sıfırlar.)

def test(img):
    img = cv2.medianBlur(img,21)
    img = cv2.dilate(img, np.ones((15,15),np.uint8))
    cv2.imshow("img",img)
    img = cv2.resize(img, (20,20)).reshape(-1,400).astype(np.float32) 
    ret, results, neighbours, distance = knn.findNearest(img, 5)
    
    cv2.putText(img2, str(int(ret)), (100,300), font, 10, 255, 4, cv2.LINE_AA)
    return ret

cizim = False
xi,yi = -1,-1
font = cv2.FONT_HERSHEY_SIMPLEX
img = np.ones((400,400),np.uint8)

def draw(event,x,y,flags,param):
    global cizim,xi,yi
    
    if event == cv2.EVENT_LBUTTONDOWN:
        xi,yi = x,y
        cizim = True
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if cizim:
            cv2.circle(img, (x,y), 10, 255, -1)               
        else:
            pass
    elif event == cv2.EVENT_LBUTTONUP:
        cizim == False
    
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        img[:,:] = 0

cv2.namedWindow("paint")
cv2.setMouseCallback("paint",draw)

while(1):
    img2 = np.ones((400,400),np.uint8)
   
    if cv2.waitKey(33) & 0xFF == ord("q"):
        break
    
    test(img)
    cv2.imshow("paint",img)
    cv2.imshow("result",img2)
    
cv2.destroyAllWindows()