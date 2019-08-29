# Recunoaștere emoții faciale

Emoțiile pot fi clasificate in 7 categorii - Furie, Dezgust, Frica, Fericire, Supărare, Surprindere și Neutru.
In acest proiect, am antrenat un model ce diferențiază aceste emoții.

# Data-set
Pentru antrenarea modelului am folosit următoarele seturi de date:

 - facial expression recognition (FER) data-set from Kaggle challenge
 - CK+48
 - jaffe
 
## Prelucrarea setului de date

Funcția processing_data primește calea către setul de date. Setul de date este citit, iar informatia regasita in coloana 'pixels', respenctiv 'emotions' este transpusa intr-un array, respectiv o matrice.
Informația aflată in X, respectiv în y este salvată în 'dataX', respectiv 'dataY'.
Pentru antrenarea/ testarea modelului setul de date este împărțit prin funcția **split_data** în două subseturi, unul de antrenare și unul de testare. Deasemenea informația este salvată în 'dataX_train', 'dataY_train', 'dataX_test' și 'dataY_test'.

# Arhitectura modelului
În funcția **get_model**, folosind Keras am creat o rețea convoluțională secvențială:  `model = Sequential()`. Pentru a constru CNN, am folosit următoarele funcții:

 1. `model.add(Conv2D())` - 2D convolution layer
 2. `model.add(BatchNormalization())` - realizează operația de **batch normalization**
 3. `model.add(MaxPooling2D())` - realizează operația de **pooling**
 4. `model.add(Dropout())` - ignoră la întâmplare câțiva neuroni pentru a preveni **overfitting**
 5. `model.add(Flatten())`
 6. `model.add(Dense())` -

Funcțiile de activare folosite sunt **ReLu** și **Softmax**

**Model summary**
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_33 (Conv2D)           (None, 46, 46, 64)        640       
_________________________________________________________________
batch_normalization_33 (Batc (None, 46, 46, 64)        256       
_________________________________________________________________
conv2d_34 (Conv2D)           (None, 46, 46, 64)        36928     
_________________________________________________________________
batch_normalization_34 (Batc (None, 46, 46, 64)        256       
_________________________________________________________________
max_pooling2d_17 (MaxPooling (None, 23, 23, 64)        0         
_________________________________________________________________
dropout_29 (Dropout)         (None, 23, 23, 64)        0         
_________________________________________________________________
conv2d_35 (Conv2D)           (None, 23, 23, 128)       73856     
_________________________________________________________________
batch_normalization_35 (Batc (None, 23, 23, 128)       512       
_________________________________________________________________
conv2d_36 (Conv2D)           (None, 23, 23, 128)       147584    
_________________________________________________________________
batch_normalization_36 (Batc (None, 23, 23, 128)       512       
_________________________________________________________________
max_pooling2d_18 (MaxPooling (None, 11, 11, 128)       0         
_________________________________________________________________
dropout_30 (Dropout)         (None, 11, 11, 128)       0         
_________________________________________________________________
conv2d_37 (Conv2D)           (None, 11, 11, 256)       295168    
_________________________________________________________________
batch_normalization_37 (Batc (None, 11, 11, 256)       1024      
_________________________________________________________________
conv2d_38 (Conv2D)           (None, 11, 11, 256)       590080    
_________________________________________________________________
batch_normalization_38 (Batc (None, 11, 11, 256)       1024      
_________________________________________________________________
max_pooling2d_19 (MaxPooling (None, 5, 5, 256)         0         
_________________________________________________________________
dropout_31 (Dropout)         (None, 5, 5, 256)         0         
_________________________________________________________________
conv2d_39 (Conv2D)           (None, 5, 5, 512)         1180160   
_________________________________________________________________
batch_normalization_39 (Batc (None, 5, 5, 512)         2048      
_________________________________________________________________
conv2d_40 (Conv2D)           (None, 5, 5, 512)         2359808   
_________________________________________________________________
batch_normalization_40 (Batc (None, 5, 5, 512)         2048      
_________________________________________________________________
max_pooling2d_20 (MaxPooling (None, 2, 2, 512)         0         
_________________________________________________________________
dropout_32 (Dropout)         (None, 2, 2, 512)         0         
_________________________________________________________________
flatten_5 (Flatten)          (None, 2048)              0         
_________________________________________________________________
dense_17 (Dense)             (None, 512)               1049088   
_________________________________________________________________
dropout_33 (Dropout)         (None, 512)               0         
_________________________________________________________________
dense_18 (Dense)             (None, 256)               131328    
_________________________________________________________________
dropout_34 (Dropout)         (None, 256)               0         
_________________________________________________________________
dense_19 (Dense)             (None, 128)               32896     
_________________________________________________________________
dropout_35 (Dropout)         (None, 128)               0         
_________________________________________________________________
dense_20 (Dense)             (None, 7)                 903       
=================================================================
Total params: 5,906,119
Trainable params: 5,902,279
Non-trainable params: 3,840
_________________________________________________________________


Modelul a fost compilat cu *categorical_crossentropy* ca *loss function* și *Adam optimizer*

# Antrenarea și testarea modelului
Pentru a preveni overfitting-ul am antrenat modelul folosind k-fold cross validation.
Funcția **k_fold** realizează antrenarea modelului.
Setul de date de antrenare a fost împărțit in k = 5 subseturi, care la rândul lor au fost  împărțite in seturi de câte 64 de imagini și au fost trecute prin model de 100 de ori. 
![](https://picasaweb.google.com/107503410125380287446/6730451544730563857#6730451547347497458 "KFold")

În urma antrenării s-a obținut o acuratețe de 0.6536 pe setul de date de validare, iar acuratețea pe setul de date de testare este 0.6465.

## Matrice de confuzie
Pentru a afla ce emotii sunt confundate am creat o matrice de conuzie:
![Matricea de confuzie](https://github.com/CristianaLazar/Recunoastere-emotii-faciale/blob/master/images/ConfMat.png "ConfMat")
![Matricea de precizie](https://github.com/CristianaLazar/Recunoastere-emotii-faciale/blob/master/images/PrecMat.png "PredMat")

Se observa astfel ca emotii ca furie si triste sau tristete si suparare sunt usor de confundat de catre model.
