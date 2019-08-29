# Recunoaștere emoții faciale

Emoțiile pot fi clasificate in 7 categorii - Furie, Dezgust, Frica, Fericire, Supărare, Surprindere și Neutru.
In acest proiect, am antrenat un model ce diferențiază aceste emoții.

# Fisiere
In fisierul Fer.py de regaseste intregul proiect

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

![Model summary](https://github.com/CristianaLazar/Recunoastere-emotii-faciale/blob/master/images/modelSummary.png "Model")


Modelul a fost compilat cu *categorical_crossentropy* ca *loss function* și *Adam optimizer*

# Antrenarea și testarea modelului
Pentru a preveni overfitting-ul am antrenat modelul folosind k-fold cross validation.
Funcția **k_fold** realizează antrenarea modelului.

Setul de date de antrenare a fost împărțit in k = 5 subseturi, care la rândul lor au fost  împărțite in seturi de câte 64 de imagini și au fost trecute prin model de 100 de ori. 
![KFold](https://github.com/CristianaLazar/Recunoastere-emotii-faciale/blob/master/images/KFold.png "KFold")

În urma antrenării s-a obținut o acuratețe de 0.6536 pe setul de date de validare, iar acuratețea pe setul de date de testare este 0.6465.

## Matrice de confuzie
Pentru a afla ce emotii sunt confundate am creat o matrice de conuzie:
![Matricea de confuzie](https://github.com/CristianaLazar/Recunoastere-emotii-faciale/blob/master/images/ConfMat.png "ConfMat")

![Matricea de precizie](https://github.com/CristianaLazar/Recunoastere-emotii-faciale/blob/master/images/PrecMat.png "PredMat")

Se observa astfel ca emotii ca furie si triste sau tristete si suparare sunt usor de confundat de catre model.
