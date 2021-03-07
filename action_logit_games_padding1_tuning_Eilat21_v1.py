from sklearn.linear_model import LogisticRegression
from nltk.classify import SklearnClassifier


import pickle
import numpy as np
import misc


# ff1='mnist_padded_b0p1_v5_X28_Tx0y0_scaling_act_full1.pkl'
# ff2='mnist_padded_b0p1_v5_X28_Tx0y0_scaling_act_full2.pkl'
ff1='mnist_padded_b0p1_v5_X28_Tx0y0_act_full1.pkl'
ff2='mnist_padded_b0p1_v5_X28_Tx0y0_act_full2.pkl'
print('loading from: ', ff1, 'and', ff2)
with open(ff1,'rb') as f:
    action_records1,labels1 = pickle.load(f)
with open(ff2,'rb') as f:
    action_records2,labels2 = pickle.load(f)
action_records = action_records1 + action_records2
labels = labels1+labels2

# rr=6000
# ll=59999
# action_records=[]
# labels=[]
# chunks=(ll-1)//rr+1
# for ii in range(chunks):
#     stri=ii*rr
#     stpi=min([(ii+1)*rr,ll])
#     # ff='mnist_padded_b0p1_v5_X28_Tx0y0_VFB101_act_full'+str(ii+1)+'_of'+str(chunks)+'.pkl'
#     ff='mnist_padded_b0p1_v5_X28_Tx0y0_VFB101_act_full'+str(ii+1)+'_of'+str(chunks)+'.pkl'
#     with open(ff,'rb') as f:
#         ac,la = pickle.load(f)
#     print('load:',ff)
#     action_records += ac
#     labels += la

# from mnist import MNIST
#
# mnist = MNIST('/home/bnapp/datasets/mnist/')
# _, labels = mnist.load_training()

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
# offsets=[np.int32(np.round(uu)) for uu in
#          [0*lagfac,1*lagfac,2*lagfac,3*lagfac,4*lagfac,5*lagfac]
#         ]

# offsets=[np.int32(np.round(uu)) for uu in
#          [0*lagfac,1*lagfac,2*lagfac]
#         ]
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
# for C in [0.2, 0.1, 0.08, 0.06, 0.04, 0.02]:
    print('C=',C)
    classifier4 = SklearnClassifier(LogisticRegression(C=C, penalty='l1'), sparse=False).train(train_data4)

    val_labels4=classifier4.classify_many(ngram_records[train_max:])
    aa=[x==y for x,y in zip(val_labels4,labels[train_max:])]
    print('test accuracy:',np.mean(aa))
    accu.append(np.mean(aa))

    train_labels4=classifier4.classify_many(ngram_records[:train_max])
    aatr=[x==y for x,y in zip(train_labels4, labels[:train_max])]
    print('train accuracy:',np.mean(aatr))
    classifiers.append(classifier4)

    # # with open('classifiers_temp_b0p03_v5_n6_l1_sweep_th1000_scale.pkl','wb') as f:
    # #     pickle.dump(classifiers,f)
    #
    # with open('DEBU_classifiers_temp_b0p1_v5_n6_l1_VFB_sweep_th1000_scale.pkl','wb') as f:
    #     pickle.dump(classifiers,f)