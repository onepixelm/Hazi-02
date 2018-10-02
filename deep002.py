#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 10:21:18 2018

@author: kakas
"""
import numpy as np
from sklearn import preprocessing
import copy

def activation(x):
    return 1 / (1 + np.exp(-x))

def dactivation(x):
    return np.exp(-x)/((1+np.exp(-x))**2)

# MLP osztály létrehozása.
class MLP:
    
    # A hálózat inicializálása az argumentumként megadott méretek alapján.
    def __init__(self, *args):
        # random seed megadása
        np.random.seed(123)
        # A hálózat formája (rétegek száma), amely megegyezik a paraméterek számával
        self.shape = args
        n = len(args)
        # Rétegek létrehozása
        self.layers = []
        # Bemeneti réteg létrehozása (+1 egység a BIAS-nak)
        self.layers.append(np.ones(self.shape[0]+1))
        # Rejtett réteg(ek) és a kimeneti réteg létrehozása
        for i in range(1,n):
            self.layers.append(np.ones(self.shape[i]))
        # Súlymátrix létrehozása
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                         self.layers[i+1].size)))
        # dw fogja tartalmazni a súlyok utolsó módosításait (később pl. a momentum módszer számára)
        self.dw = [0,]*len(self.weights)
        # Súlyok újrainicializálása
        self.reset()
    
        # Súlyok újrainicializálási függvényének definiálása
    def reset(self):
        for i in range(len(self.weights)):
            # véletlen számok [0,1) tartományban 
            Z = np.random.random((self.layers[i].size,self.layers[i+1].size))
            # átskálázzuk a súlyokat -1..1 tartományba
            self.weights[i][...] = (2*Z-1)*1

    # A bemenő adatok végigküldése a hálózaton, kimeneti rétegig (forward propagation)
    def propagate_forward(self, data):
        # Bemeneti réteg beállítása (tanító adatok)
        self.layers[0][0:-1] = data
        # Az adatok végigküldése a bemeneti rétegtől az utolsó előtti rétegig (az utolsó ugyanis a kimeneti réteg).
        # A szigmoid aktivációs függvény használatával, mátrixszorzások alkalmazásával.
        # Az előadáson a "layers" változót jelöltük "a"-val.
        for i in range(1,len(self.shape)):
            self.layers[i][...] = activation(np.dot(self.layers[i-1],self.weights[i-1]))
        # Visszatérés a hálózat által becsült eredménnyel
        return self.layers[-1]

   # Hibavisszaterjesztés (backpropagation) definiálása. 
    # A a learning rate (tanulási ráta) paraméter befolyásolja, hogy a hálózat súlyait milyen
    # mértékben módosítsuk a gradiens függvényében. Ha ez az érték túl magas, akkor a háló 
    # "oszcillálhat" egy lokális vagy globális minimum körül. Ha túl kicsi értéket választunk,
    # akkor pedig jelentősen több időbe telik mire elérjük a legjobb megoldást vagy leakad egy lokális 
    # minimumban és sose éri el azt.
    
    # HF2 start Momentum
    def momentum(self, epoch, egyutthato ):   # if egyutthato [nü, alfa, 0] then don't work it, [nü, alfa, 1] then use it
        # Mivel az elso epochnál még nincs a delta W-nek előélete
        if epoch > 1:
            changes = egyutthato[0]*(W(2).T)+egyutthato[1]*changes
        else:
            changes = egyutthato[1]*(W(2).T)
        # A delta W következő generációja
        self.layers += changes
        return self.layers [-i]
        # HF2 end Momentum
    
        # HF2 start L1reg
    def L1reg(self, egyutthato ): # if egyutthato [nü, lambda1, 0] then don't work it, [nü, lambda1, 1] then use it
        self.layers = -egyutthato[0] * (W(2).T)+ changes - egyutthato[0] * egyutthato[1] * numpy.sign (self.layers)
        return self.layers [-i]
        # HF2 end L1reg
    
        # HF2 start L2reg
    def L2reg(self,  egyutthato ): # if egyutthato [nü, lambda2, 0] then don't work it, [nü, lambda2, 1] then use it
        self.layers = -egyutthato[0]*(W(2).T)- (1-egyutthato[0] * egyutthato[1]) * self.layers    
        return self.layers [-i]
        # HF2 end L2reg
    
    def propagate_backward(self, target, lrate=0.1):
        deltas = []
        # Hiba kiszámítása a kimeneti rétegen
        error = -(target-self.layers[-1]) # y-y_kalap
        # error*dactivation(s(3))
        delta = np.multiply(error,dactivation(np.dot(self.layers[-2],self.weights[-1])))
        deltas.append(delta)
        # Gradiens kiszámítása a rejtett réteg(ek)ben
        for i in range(len(self.shape)-2,0,-1):
            # pl. utolsó rejtett réteg: delta(3)*(W(2).T)*dactivation(s(2)) (lásd előadás)
            delta=np.dot(deltas[0],self.weights[i].T)*dactivation(np.dot(self.layers[i-1],self.weights[i-1]))
            deltas.insert(0,delta)            
        # Súlyok módosítása
        for i in range(len(self.weights)):
        # Momentum switch on, if [2] is true, then will change W(2) 
            if (momentum.egyutthato [2])==1 :
                layer = np.atleast_2d(momentum.self.layers[i])
            elif (L1reg.egyutthato [2])==2 :
                layer = np.atleast_2d(L1reg.self.layers[i])        # L1reg switch on, if [2] is true, then will change W(2)        
            elif (L2reg.egyutthato [2])==3 : 
                layer = np.atleast_2d(L2reg.self.layers[i])        # L2reg switch on, if [2] is true, then will change W(2) 
            else :                  
                layer = np.atleast_2d(self.layers[i])
            
            delta = np.atleast_2d(deltas[i])
            # pl. utolsó rétegben: delta(3)*a(2) (lásd előadás)
            dw = -lrate*np.dot(layer.T,delta)
            # súlyok módosítása
            self.weights[i] += dw 
            # a súlymódosítás eltárolása
            self.dw[i] = dw

        # Visszatérés a hibával
        return (error**2).sum()
    
    def learn(network, X, Y, valid_split, test_split, epochs=20, lrate=0.1):

        # train-validation-test minták különválasztása
        X_train = X[0:int(nb_samples*(1-valid_split-test_split))]
        Y_train = Y[0:int(nb_samples*(1-valid_split-test_split))]
        X_valid = X[int(nb_samples*(1-valid_split-test_split)):int(nb_samples*(1-test_split))]
        Y_valid = Y[int(nb_samples*(1-valid_split-test_split)):int(nb_samples*(1-test_split))]
        X_test  = X[int(nb_samples*(1-test_split)):]
        Y_test  = Y[int(nb_samples*(1-test_split)):]
    
        # standardizálás
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test  = scaler.transform(X_test)
    
        # ugyanolyan sorrendben keverjük be a bemeneteket és kimeneteket, a három külön adatbázisra
        randperm = np.random.permutation(len(X_train))
        X_train, Y_train = X_train[randperm], Y_train[randperm]
        randperm = np.random.permutation(len(X_valid))
        X_valid, Y_valid = X_valid[randperm], Y_valid[randperm]
        randperm = np.random.permutation(len(X_test))
        X_test, Y_test = X_test[randperm], Y_test[randperm]
        
        best_valid_err = np.inf
        es_counter = 0 # early stopping counter
        best_model = network
     
        # Tanítási fázis, epoch-szor megyünk át 1-1 véltelenszerűen kiválasztott mintán.
        for i in range(epochs):
            # Jelen megoldás azt a módszert használja, hogy a megadott 
            # tanító adatokon végigmegyünk és minden elemet először végigküldünk
            # a hálózaton, majd terjeszti vissza a kapott eltérést az
            # elvárt eredménytől. Ezt hívjuk SGD-ek (stochastic gradient descent).
            train_err = 0
            for k in range(X_train.shape[0]):
                network.propagate_forward( X_train[k] )
                train_err += network.propagate_backward( Y_train[k], lrate )
            train_err /= X_train.shape[0]

            # validációs fázis
            valid_err = 0
            o_valid = np.zeros(X_valid.shape[0])
            for k in range(X_valid.shape[0]):
                o_valid[k] = network.propagate_forward(X_valid[k])
                valid_err += (o_valid[k]-Y_valid[k])**2
            valid_err /= X_valid.shape[0]

            print("%d epoch, train_err: %.4f, valid_err: %.4f" % (i, train_err, valid_err))

        # Tesztelési fázis
        print("\n--- TESZTELÉS ---\n")
        test_err = 0
        o_test = np.zeros(X_test.shape[0])
        for k in range(X_test.shape[0]):
            o_test[k] = network.propagate_forward(X_test[k])
            test_err += (o_test[k]-Y_test[k])**2
            print(k, X_test[k], '%.2f' % o_test[k], ' (elvart eredmeny: %.2f)' % Y_test[k])
        test_err /= X_test.shape[0]
        
        fig1=plt.figure()
        plt.scatter(X_test[:,0], X_test[:,1], c=np.round(o_test[:]), cmap=plt.cm.cool)       
        
        # Mesterséges neurális hálózat létrehozása, 2 bemenettel, 10 rejtett neuronnal és 1 kimenettel
    network = MLP(2,10,1)
        
     # Tanító, validációs és teszt adatok megadása a rendszernek (zajjal terhelt XOR adatok)
    nb_samples=1000
    X = np.zeros((nb_samples,2))
    Y = np.zeros(nb_samples)
    for i in range(0,nb_samples,4):
        noise = np.random.normal(0,1,8)
        X[i], Y[i] = (-2+noise[0],-2+noise[1]), 0
        X[i+1], Y[i+1] = (2+noise[2],-2+noise[3]), 1
        X[i+2], Y[i+2] = (-2+noise[4],2+noise[5]), 1
        X[i+3], Y[i+3] = (2+noise[6],2+noise[7]), 0

        # ...és az adatok kirajzolása
    fig1=plt.figure()
    plt.scatter(X[:,0],X[:,1],c=Y[:], cmap=plt.cm.cool)      