# CNN_Keras_cifar-10
classify pictures (use Keras) database : cifar-10

## CIFAR-10資料集

[The CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
[CIFAR-10資料集介紹](https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/505061/)
                
     32x32 RGB影像，10個類別：飛機(airplane)，汽車(automobile)，鳥(bird)，貓(cat)，鹿(deer)，狗(dog)，
     青蛙(frog)，馬(horse)，船(ship)，卡車(truck)。
    共60000張圖片，每個類別6000張。training data：50000張，testing data：10000張。
    train : x_train.shape (50000, 32, 32,  3)  , y_train.shape (50000, 1)
    test : x_test.shape (10000, 32, 32, 3)  ,  y_test.shape(10000, 1)

    有5個測試batch和1個測試batch，每個batch裡面包含10000張圖片。先切出testing data ，測試batch是從分開
    的各個類別內部隨機選擇1000張，因此每個類別都會剩下5000張; 1個訓練batch從10個類別混在一起的50000張
    隨機選擇10000張，因此每個訓練batch內的類別數量分佈可能不均勻。
    
## CNN網路架構
![image](https://github.com/leodflag/CNN_Keras_cifar-10/blob/master/CNN_model_CIFAR10.png)
    
## 結果
![image](https://github.com/leodflag/CNN_Keras_cifar-10/blob/master/CNN_train.png)

    
