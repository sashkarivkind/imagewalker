from sklearn.linear_model import LogisticRegression
from nltk.classify import SklearnClassifier


import pickle
import numpy as np
import misc


ff1='mnist_padded_b0p03_act_full1.pkl'
ff2='mnist_padded_b0p03_act_full2.pkl'
print('loading from: ', ff1, 'and', ff2)
with open(ff1,'rb') as f:
    action_records1,labels1 = pickle.load(f)
with open(ff2,'rb') as f:
    action_records2,labels2 = pickle.load(f)
action_records   = action_records1 + action_records2


from mnist import MNIST

mnist = MNIST('/home/bnapp/datasets/mnist/')
_, labels = mnist.load_training()

def ngram_truncate(ngram_records_trunc,th):
    for ii,zz in enumerate(ngram_full):
        if ngram_full[zz]<th:
            for ng in ngram_records_trunc:
                 ng.pop(zz, None)

classifiers=[]
accu=[]
# for lagfac in [1.0,1.2,1.4,1.6,1.8]:
lagfac=1.5
offsets=[np.int32(np.round(uu)) for uu in
         [0*lagfac,1*lagfac,2*lagfac,3*lagfac,4*lagfac,5*lagfac,6*lagfac,7*lagfac,8*lagfac]
        ]
ngram_records=[misc.prep_n_grams(aa[:1000],offsets=offsets) for aa in action_records]
print('done with ngrams', offsets)
ngram_full={}
for rec in ngram_records:
    for zz in rec.keys():
        if not(zz in ngram_full.keys()):
            ngram_full[zz]=rec[zz]
        else:
            ngram_full[zz]+=rec[zz]

ngram_truncate(ngram_records,1000)
print('done truncation')
train_max=55000
train_data4=[(x,y) for x,y in zip(ngram_records[:train_max],labels[:train_max])]

# for C in [1,0.8,0.6,0.4,0.2]:
# for C in [0.1, 0.08, 0.06, 0.04, 0.02]:
for C in [1,0.8,0.6,0.4,0.2, 0.1, 0.08, 0.06, 0.04, 0.02]:
    print('C=',C)
    classifier4 = SklearnClassifier(LogisticRegression(C=C, penalty='l1'), sparse=False).train(train_data4)

    val_labels4=classifier4.classify_many(ngram_records[train_max:])
    aa=[x==y for x,y in zip(val_labels4,labels[train_max:])]
    print(np.mean(aa))
    accu.append(np.mean(aa))

    train_labels4=classifier4.classify_many(ngram_records[:train_max])
    aatr=[x==y for x,y in zip(train_labels4, labels[:train_max])]
    print(np.mean(aatr))
    classifiers.append(classifier4)

    with open('classifiers_temp_b0p03_v5_n9_l1_sweep_th1000_pp.pkl','wb') as f:
        pickle.dump(classifiers,f)