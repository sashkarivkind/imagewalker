��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12v2.4.0-49-g85c8b2a817f8�
z
cnn1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecnn1/kernel
s
cnn1/kernel/Read/ReadVariableOpReadVariableOpcnn1/kernel*&
_output_shapes
: *
dtype0
j
	cnn1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	cnn1/bias
c
cnn1/bias/Read/ReadVariableOpReadVariableOp	cnn1/bias*
_output_shapes
: *
dtype0
|
cnn12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_namecnn12/kernel
u
 cnn12/kernel/Read/ReadVariableOpReadVariableOpcnn12/kernel*&
_output_shapes
:  *
dtype0
l

cnn12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
cnn12/bias
e
cnn12/bias/Read/ReadVariableOpReadVariableOp
cnn12/bias*
_output_shapes
: *
dtype0
z
cnn2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_namecnn2/kernel
s
cnn2/kernel/Read/ReadVariableOpReadVariableOpcnn2/kernel*&
_output_shapes
: @*
dtype0
j
	cnn2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	cnn2/bias
c
cnn2/bias/Read/ReadVariableOpReadVariableOp	cnn2/bias*
_output_shapes
:@*
dtype0
|
cnn22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_namecnn22/kernel
u
 cnn22/kernel/Read/ReadVariableOpReadVariableOpcnn22/kernel*&
_output_shapes
:@@*
dtype0
l

cnn22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
cnn22/bias
e
cnn22/bias/Read/ReadVariableOpReadVariableOp
cnn22/bias*
_output_shapes
:@*
dtype0
{
cnn3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*
shared_namecnn3/kernel
t
cnn3/kernel/Read/ReadVariableOpReadVariableOpcnn3/kernel*'
_output_shapes
:@�*
dtype0
k
	cnn3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name	cnn3/bias
d
cnn3/bias/Read/ReadVariableOpReadVariableOp	cnn3/bias*
_output_shapes	
:�*
dtype0
~
cnn32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*
shared_namecnn32/kernel
w
 cnn32/kernel/Read/ReadVariableOpReadVariableOpcnn32/kernel*(
_output_shapes
:��*
dtype0
m

cnn32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
cnn32/bias
f
cnn32/bias/Read/ReadVariableOpReadVariableOp
cnn32/bias*
_output_shapes	
:�*
dtype0
r

fc1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_name
fc1/kernel
k
fc1/kernel/Read/ReadVariableOpReadVariableOp
fc1/kernel* 
_output_shapes
:
��*
dtype0
i
fc1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
fc1/bias
b
fc1/bias/Read/ReadVariableOpReadVariableOpfc1/bias*
_output_shapes	
:�*
dtype0
u
final/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*
shared_namefinal/kernel
n
 final/kernel/Read/ReadVariableOpReadVariableOpfinal/kernel*
_output_shapes
:	�
*
dtype0
l

final/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
final/bias
e
final/bias/Read/ReadVariableOpReadVariableOp
final/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
Adam/cnn1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/cnn1/kernel/m
�
&Adam/cnn1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/cnn1/kernel/m*&
_output_shapes
: *
dtype0
x
Adam/cnn1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/cnn1/bias/m
q
$Adam/cnn1/bias/m/Read/ReadVariableOpReadVariableOpAdam/cnn1/bias/m*
_output_shapes
: *
dtype0
�
Adam/cnn12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *$
shared_nameAdam/cnn12/kernel/m
�
'Adam/cnn12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/cnn12/kernel/m*&
_output_shapes
:  *
dtype0
z
Adam/cnn12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/cnn12/bias/m
s
%Adam/cnn12/bias/m/Read/ReadVariableOpReadVariableOpAdam/cnn12/bias/m*
_output_shapes
: *
dtype0
�
Adam/cnn2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*#
shared_nameAdam/cnn2/kernel/m
�
&Adam/cnn2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/cnn2/kernel/m*&
_output_shapes
: @*
dtype0
x
Adam/cnn2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameAdam/cnn2/bias/m
q
$Adam/cnn2/bias/m/Read/ReadVariableOpReadVariableOpAdam/cnn2/bias/m*
_output_shapes
:@*
dtype0
�
Adam/cnn22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameAdam/cnn22/kernel/m
�
'Adam/cnn22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/cnn22/kernel/m*&
_output_shapes
:@@*
dtype0
z
Adam/cnn22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/cnn22/bias/m
s
%Adam/cnn22/bias/m/Read/ReadVariableOpReadVariableOpAdam/cnn22/bias/m*
_output_shapes
:@*
dtype0
�
Adam/cnn3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*#
shared_nameAdam/cnn3/kernel/m
�
&Adam/cnn3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/cnn3/kernel/m*'
_output_shapes
:@�*
dtype0
y
Adam/cnn3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*!
shared_nameAdam/cnn3/bias/m
r
$Adam/cnn3/bias/m/Read/ReadVariableOpReadVariableOpAdam/cnn3/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/cnn32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameAdam/cnn32/kernel/m
�
'Adam/cnn32/kernel/m/Read/ReadVariableOpReadVariableOpAdam/cnn32/kernel/m*(
_output_shapes
:��*
dtype0
{
Adam/cnn32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/cnn32/bias/m
t
%Adam/cnn32/bias/m/Read/ReadVariableOpReadVariableOpAdam/cnn32/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/fc1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*"
shared_nameAdam/fc1/kernel/m
y
%Adam/fc1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/fc1/kernel/m* 
_output_shapes
:
��*
dtype0
w
Adam/fc1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_nameAdam/fc1/bias/m
p
#Adam/fc1/bias/m/Read/ReadVariableOpReadVariableOpAdam/fc1/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/final/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*$
shared_nameAdam/final/kernel/m
|
'Adam/final/kernel/m/Read/ReadVariableOpReadVariableOpAdam/final/kernel/m*
_output_shapes
:	�
*
dtype0
z
Adam/final/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/final/bias/m
s
%Adam/final/bias/m/Read/ReadVariableOpReadVariableOpAdam/final/bias/m*
_output_shapes
:
*
dtype0
�
Adam/cnn1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/cnn1/kernel/v
�
&Adam/cnn1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/cnn1/kernel/v*&
_output_shapes
: *
dtype0
x
Adam/cnn1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/cnn1/bias/v
q
$Adam/cnn1/bias/v/Read/ReadVariableOpReadVariableOpAdam/cnn1/bias/v*
_output_shapes
: *
dtype0
�
Adam/cnn12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *$
shared_nameAdam/cnn12/kernel/v
�
'Adam/cnn12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/cnn12/kernel/v*&
_output_shapes
:  *
dtype0
z
Adam/cnn12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/cnn12/bias/v
s
%Adam/cnn12/bias/v/Read/ReadVariableOpReadVariableOpAdam/cnn12/bias/v*
_output_shapes
: *
dtype0
�
Adam/cnn2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*#
shared_nameAdam/cnn2/kernel/v
�
&Adam/cnn2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/cnn2/kernel/v*&
_output_shapes
: @*
dtype0
x
Adam/cnn2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameAdam/cnn2/bias/v
q
$Adam/cnn2/bias/v/Read/ReadVariableOpReadVariableOpAdam/cnn2/bias/v*
_output_shapes
:@*
dtype0
�
Adam/cnn22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameAdam/cnn22/kernel/v
�
'Adam/cnn22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/cnn22/kernel/v*&
_output_shapes
:@@*
dtype0
z
Adam/cnn22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/cnn22/bias/v
s
%Adam/cnn22/bias/v/Read/ReadVariableOpReadVariableOpAdam/cnn22/bias/v*
_output_shapes
:@*
dtype0
�
Adam/cnn3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*#
shared_nameAdam/cnn3/kernel/v
�
&Adam/cnn3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/cnn3/kernel/v*'
_output_shapes
:@�*
dtype0
y
Adam/cnn3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*!
shared_nameAdam/cnn3/bias/v
r
$Adam/cnn3/bias/v/Read/ReadVariableOpReadVariableOpAdam/cnn3/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/cnn32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameAdam/cnn32/kernel/v
�
'Adam/cnn32/kernel/v/Read/ReadVariableOpReadVariableOpAdam/cnn32/kernel/v*(
_output_shapes
:��*
dtype0
{
Adam/cnn32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/cnn32/bias/v
t
%Adam/cnn32/bias/v/Read/ReadVariableOpReadVariableOpAdam/cnn32/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/fc1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*"
shared_nameAdam/fc1/kernel/v
y
%Adam/fc1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/fc1/kernel/v* 
_output_shapes
:
��*
dtype0
w
Adam/fc1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_nameAdam/fc1/bias/v
p
#Adam/fc1/bias/v/Read/ReadVariableOpReadVariableOpAdam/fc1/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/final/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*$
shared_nameAdam/final/kernel/v
|
'Adam/final/kernel/v/Read/ReadVariableOpReadVariableOpAdam/final/kernel/v*
_output_shapes
:	�
*
dtype0
z
Adam/final/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/final/bias/v
s
%Adam/final/bias/v/Read/ReadVariableOpReadVariableOpAdam/final/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
�a
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�`
value�`B�` B�`
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer_with_weights-7
layer-15
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
 regularization_losses
!trainable_variables
"	keras_api
R
#	variables
$regularization_losses
%trainable_variables
&	keras_api
R
'	variables
(regularization_losses
)trainable_variables
*	keras_api
h

+kernel
,bias
-	variables
.regularization_losses
/trainable_variables
0	keras_api
h

1kernel
2bias
3	variables
4regularization_losses
5trainable_variables
6	keras_api
R
7	variables
8regularization_losses
9trainable_variables
:	keras_api
R
;	variables
<regularization_losses
=trainable_variables
>	keras_api
h

?kernel
@bias
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
h

Ekernel
Fbias
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
R
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
R
O	variables
Pregularization_losses
Qtrainable_variables
R	keras_api
R
S	variables
Tregularization_losses
Utrainable_variables
V	keras_api
h

Wkernel
Xbias
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
h

]kernel
^bias
_	variables
`regularization_losses
atrainable_variables
b	keras_api
�
citer

dbeta_1

ebeta_2
	fdecay
glearning_ratem�m�m�m�+m�,m�1m�2m�?m�@m�Em�Fm�Wm�Xm�]m�^m�v�v�v�v�+v�,v�1v�2v�?v�@v�Ev�Fv�Wv�Xv�]v�^v�
v
0
1
2
3
+4
,5
16
27
?8
@9
E10
F11
W12
X13
]14
^15
 
v
0
1
2
3
+4
,5
16
27
?8
@9
E10
F11
W12
X13
]14
^15
�

hlayers
	variables
ilayer_metrics
jlayer_regularization_losses
knon_trainable_variables
lmetrics
regularization_losses
trainable_variables
 
WU
VARIABLE_VALUEcnn1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	cnn1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�

mlayers
	variables
nlayer_metrics
olayer_regularization_losses
pnon_trainable_variables
qmetrics
regularization_losses
trainable_variables
XV
VARIABLE_VALUEcnn12/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
cnn12/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�

rlayers
	variables
slayer_metrics
tlayer_regularization_losses
unon_trainable_variables
vmetrics
 regularization_losses
!trainable_variables
 
 
 
�

wlayers
#	variables
xlayer_metrics
ylayer_regularization_losses
znon_trainable_variables
{metrics
$regularization_losses
%trainable_variables
 
 
 
�

|layers
'	variables
}layer_metrics
~layer_regularization_losses
non_trainable_variables
�metrics
(regularization_losses
)trainable_variables
WU
VARIABLE_VALUEcnn2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	cnn2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1
 

+0
,1
�
�layers
-	variables
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�metrics
.regularization_losses
/trainable_variables
XV
VARIABLE_VALUEcnn22/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
cnn22/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21
 

10
21
�
�layers
3	variables
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�metrics
4regularization_losses
5trainable_variables
 
 
 
�
�layers
7	variables
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�metrics
8regularization_losses
9trainable_variables
 
 
 
�
�layers
;	variables
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�metrics
<regularization_losses
=trainable_variables
WU
VARIABLE_VALUEcnn3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	cnn3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
@1
 

?0
@1
�
�layers
A	variables
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�metrics
Bregularization_losses
Ctrainable_variables
XV
VARIABLE_VALUEcnn32/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
cnn32/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

E0
F1
 

E0
F1
�
�layers
G	variables
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�metrics
Hregularization_losses
Itrainable_variables
 
 
 
�
�layers
K	variables
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�metrics
Lregularization_losses
Mtrainable_variables
 
 
 
�
�layers
O	variables
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�metrics
Pregularization_losses
Qtrainable_variables
 
 
 
�
�layers
S	variables
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�metrics
Tregularization_losses
Utrainable_variables
VT
VARIABLE_VALUE
fc1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEfc1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

W0
X1
 

W0
X1
�
�layers
Y	variables
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�metrics
Zregularization_losses
[trainable_variables
XV
VARIABLE_VALUEfinal/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
final/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

]0
^1
 

]0
^1
�
�layers
_	variables
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�metrics
`regularization_losses
atrainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
v
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
 
 
 

�0
�1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
zx
VARIABLE_VALUEAdam/cnn1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/cnn1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/cnn12/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/cnn12/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/cnn2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/cnn2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/cnn22/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/cnn22/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/cnn3/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/cnn3/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/cnn32/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/cnn32/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/fc1/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/fc1/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/final/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/final/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/cnn1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/cnn1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/cnn12/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/cnn12/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/cnn2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/cnn2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/cnn22/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/cnn22/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/cnn3/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/cnn3/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/cnn32/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/cnn32/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/fc1/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/fc1/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/final/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/final/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_2Placeholder*/
_output_shapes
:���������  *
dtype0*$
shape:���������  
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2cnn1/kernel	cnn1/biascnn12/kernel
cnn12/biascnn2/kernel	cnn2/biascnn22/kernel
cnn22/biascnn3/kernel	cnn3/biascnn32/kernel
cnn32/bias
fc1/kernelfc1/biasfinal/kernel
final/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_12788
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamecnn1/kernel/Read/ReadVariableOpcnn1/bias/Read/ReadVariableOp cnn12/kernel/Read/ReadVariableOpcnn12/bias/Read/ReadVariableOpcnn2/kernel/Read/ReadVariableOpcnn2/bias/Read/ReadVariableOp cnn22/kernel/Read/ReadVariableOpcnn22/bias/Read/ReadVariableOpcnn3/kernel/Read/ReadVariableOpcnn3/bias/Read/ReadVariableOp cnn32/kernel/Read/ReadVariableOpcnn32/bias/Read/ReadVariableOpfc1/kernel/Read/ReadVariableOpfc1/bias/Read/ReadVariableOp final/kernel/Read/ReadVariableOpfinal/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp&Adam/cnn1/kernel/m/Read/ReadVariableOp$Adam/cnn1/bias/m/Read/ReadVariableOp'Adam/cnn12/kernel/m/Read/ReadVariableOp%Adam/cnn12/bias/m/Read/ReadVariableOp&Adam/cnn2/kernel/m/Read/ReadVariableOp$Adam/cnn2/bias/m/Read/ReadVariableOp'Adam/cnn22/kernel/m/Read/ReadVariableOp%Adam/cnn22/bias/m/Read/ReadVariableOp&Adam/cnn3/kernel/m/Read/ReadVariableOp$Adam/cnn3/bias/m/Read/ReadVariableOp'Adam/cnn32/kernel/m/Read/ReadVariableOp%Adam/cnn32/bias/m/Read/ReadVariableOp%Adam/fc1/kernel/m/Read/ReadVariableOp#Adam/fc1/bias/m/Read/ReadVariableOp'Adam/final/kernel/m/Read/ReadVariableOp%Adam/final/bias/m/Read/ReadVariableOp&Adam/cnn1/kernel/v/Read/ReadVariableOp$Adam/cnn1/bias/v/Read/ReadVariableOp'Adam/cnn12/kernel/v/Read/ReadVariableOp%Adam/cnn12/bias/v/Read/ReadVariableOp&Adam/cnn2/kernel/v/Read/ReadVariableOp$Adam/cnn2/bias/v/Read/ReadVariableOp'Adam/cnn22/kernel/v/Read/ReadVariableOp%Adam/cnn22/bias/v/Read/ReadVariableOp&Adam/cnn3/kernel/v/Read/ReadVariableOp$Adam/cnn3/bias/v/Read/ReadVariableOp'Adam/cnn32/kernel/v/Read/ReadVariableOp%Adam/cnn32/bias/v/Read/ReadVariableOp%Adam/fc1/kernel/v/Read/ReadVariableOp#Adam/fc1/bias/v/Read/ReadVariableOp'Adam/final/kernel/v/Read/ReadVariableOp%Adam/final/bias/v/Read/ReadVariableOpConst*F
Tin?
=2;	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_13465
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecnn1/kernel	cnn1/biascnn12/kernel
cnn12/biascnn2/kernel	cnn2/biascnn22/kernel
cnn22/biascnn3/kernel	cnn3/biascnn32/kernel
cnn32/bias
fc1/kernelfc1/biasfinal/kernel
final/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/cnn1/kernel/mAdam/cnn1/bias/mAdam/cnn12/kernel/mAdam/cnn12/bias/mAdam/cnn2/kernel/mAdam/cnn2/bias/mAdam/cnn22/kernel/mAdam/cnn22/bias/mAdam/cnn3/kernel/mAdam/cnn3/bias/mAdam/cnn32/kernel/mAdam/cnn32/bias/mAdam/fc1/kernel/mAdam/fc1/bias/mAdam/final/kernel/mAdam/final/bias/mAdam/cnn1/kernel/vAdam/cnn1/bias/vAdam/cnn12/kernel/vAdam/cnn12/bias/vAdam/cnn2/kernel/vAdam/cnn2/bias/vAdam/cnn22/kernel/vAdam/cnn22/bias/vAdam/cnn3/kernel/vAdam/cnn3/bias/vAdam/cnn32/kernel/vAdam/cnn32/bias/vAdam/fc1/kernel/vAdam/fc1/bias/vAdam/final/kernel/vAdam/final/bias/v*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_13646��

�<
�
E__inference_base_cifar_layer_call_and_return_conditional_losses_12564
input_2

cnn1_12516

cnn1_12518
cnn12_12521
cnn12_12523

cnn2_12528

cnn2_12530
cnn22_12533
cnn22_12535

cnn3_12540

cnn3_12542
cnn32_12545
cnn32_12547
	fc1_12553
	fc1_12555
final_12558
final_12560
identity��cnn1/StatefulPartitionedCall�cnn12/StatefulPartitionedCall�cnn2/StatefulPartitionedCall�cnn22/StatefulPartitionedCall�cnn3/StatefulPartitionedCall�cnn32/StatefulPartitionedCall�fc1/StatefulPartitionedCall�final/StatefulPartitionedCall�
cnn1/StatefulPartitionedCallStatefulPartitionedCallinput_2
cnn1_12516
cnn1_12518*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_cnn1_layer_call_and_return_conditional_losses_122002
cnn1/StatefulPartitionedCall�
cnn12/StatefulPartitionedCallStatefulPartitionedCall%cnn1/StatefulPartitionedCall:output:0cnn12_12521cnn12_12523*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cnn12_layer_call_and_return_conditional_losses_122272
cnn12/StatefulPartitionedCall�
max_pool1/PartitionedCallPartitionedCall&cnn12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_max_pool1_layer_call_and_return_conditional_losses_121552
max_pool1/PartitionedCall�
dropout_3/PartitionedCallPartitionedCall"max_pool1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_122612
dropout_3/PartitionedCall�
cnn2/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0
cnn2_12528
cnn2_12530*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_cnn2_layer_call_and_return_conditional_losses_122852
cnn2/StatefulPartitionedCall�
cnn22/StatefulPartitionedCallStatefulPartitionedCall%cnn2/StatefulPartitionedCall:output:0cnn22_12533cnn22_12535*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cnn22_layer_call_and_return_conditional_losses_123122
cnn22/StatefulPartitionedCall�
max_pool2/PartitionedCallPartitionedCall&cnn22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_max_pool2_layer_call_and_return_conditional_losses_121672
max_pool2/PartitionedCall�
dropout_4/PartitionedCallPartitionedCall"max_pool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_123462
dropout_4/PartitionedCall�
cnn3/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0
cnn3_12540
cnn3_12542*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_cnn3_layer_call_and_return_conditional_losses_123702
cnn3/StatefulPartitionedCall�
cnn32/StatefulPartitionedCallStatefulPartitionedCall%cnn3/StatefulPartitionedCall:output:0cnn32_12545cnn32_12547*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cnn32_layer_call_and_return_conditional_losses_123972
cnn32/StatefulPartitionedCall�
max_pool3/PartitionedCallPartitionedCall&cnn32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_max_pool3_layer_call_and_return_conditional_losses_121792
max_pool3/PartitionedCall�
dropout_5/PartitionedCallPartitionedCall"max_pool3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_124312
dropout_5/PartitionedCall�
flatten_1/PartitionedCallPartitionedCall"dropout_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_124502
flatten_1/PartitionedCall�
fc1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0	fc1_12553	fc1_12555*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_fc1_layer_call_and_return_conditional_losses_124692
fc1/StatefulPartitionedCall�
final/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0final_12558final_12560*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_final_layer_call_and_return_conditional_losses_124962
final/StatefulPartitionedCall�
IdentityIdentity&final/StatefulPartitionedCall:output:0^cnn1/StatefulPartitionedCall^cnn12/StatefulPartitionedCall^cnn2/StatefulPartitionedCall^cnn22/StatefulPartitionedCall^cnn3/StatefulPartitionedCall^cnn32/StatefulPartitionedCall^fc1/StatefulPartitionedCall^final/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������  ::::::::::::::::2<
cnn1/StatefulPartitionedCallcnn1/StatefulPartitionedCall2>
cnn12/StatefulPartitionedCallcnn12/StatefulPartitionedCall2<
cnn2/StatefulPartitionedCallcnn2/StatefulPartitionedCall2>
cnn22/StatefulPartitionedCallcnn22/StatefulPartitionedCall2<
cnn3/StatefulPartitionedCallcnn3/StatefulPartitionedCall2>
cnn32/StatefulPartitionedCallcnn32/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2>
final/StatefulPartitionedCallfinal/StatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_2
�
z
%__inference_cnn32_layer_call_fn_13193

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cnn32_layer_call_and_return_conditional_losses_123972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_12346

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
?__inference_cnn1_layer_call_and_return_conditional_losses_13030

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������   2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�

�
#__inference_signature_wrapper_12788
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_121492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������  ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_2
�

�
*__inference_base_cifar_layer_call_fn_13019

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_base_cifar_layer_call_and_return_conditional_losses_127062
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������  ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�

�
*__inference_base_cifar_layer_call_fn_12982

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_base_cifar_layer_call_and_return_conditional_losses_126182
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������  ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
b
)__inference_dropout_3_layer_call_fn_13081

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_122562
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
x
#__inference_fc1_layer_call_fn_13251

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_fc1_layer_call_and_return_conditional_losses_124692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
E
)__inference_max_pool1_layer_call_fn_12161

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_max_pool1_layer_call_and_return_conditional_losses_121552
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
E
)__inference_max_pool2_layer_call_fn_12173

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_max_pool2_layer_call_and_return_conditional_losses_121672
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_13210

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:����������2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_12431

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:����������2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
?__inference_cnn3_layer_call_and_return_conditional_losses_12370

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
c
D__inference_dropout_5_layer_call_and_return_conditional_losses_13205

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
E
)__inference_flatten_1_layer_call_fn_13231

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_124502
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_13143

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
?__inference_cnn1_layer_call_and_return_conditional_losses_12200

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������   2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
c
D__inference_dropout_5_layer_call_and_return_conditional_losses_12426

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_13226

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�<
�
E__inference_base_cifar_layer_call_and_return_conditional_losses_12706

inputs

cnn1_12658

cnn1_12660
cnn12_12663
cnn12_12665

cnn2_12670

cnn2_12672
cnn22_12675
cnn22_12677

cnn3_12682

cnn3_12684
cnn32_12687
cnn32_12689
	fc1_12695
	fc1_12697
final_12700
final_12702
identity��cnn1/StatefulPartitionedCall�cnn12/StatefulPartitionedCall�cnn2/StatefulPartitionedCall�cnn22/StatefulPartitionedCall�cnn3/StatefulPartitionedCall�cnn32/StatefulPartitionedCall�fc1/StatefulPartitionedCall�final/StatefulPartitionedCall�
cnn1/StatefulPartitionedCallStatefulPartitionedCallinputs
cnn1_12658
cnn1_12660*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_cnn1_layer_call_and_return_conditional_losses_122002
cnn1/StatefulPartitionedCall�
cnn12/StatefulPartitionedCallStatefulPartitionedCall%cnn1/StatefulPartitionedCall:output:0cnn12_12663cnn12_12665*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cnn12_layer_call_and_return_conditional_losses_122272
cnn12/StatefulPartitionedCall�
max_pool1/PartitionedCallPartitionedCall&cnn12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_max_pool1_layer_call_and_return_conditional_losses_121552
max_pool1/PartitionedCall�
dropout_3/PartitionedCallPartitionedCall"max_pool1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_122612
dropout_3/PartitionedCall�
cnn2/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0
cnn2_12670
cnn2_12672*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_cnn2_layer_call_and_return_conditional_losses_122852
cnn2/StatefulPartitionedCall�
cnn22/StatefulPartitionedCallStatefulPartitionedCall%cnn2/StatefulPartitionedCall:output:0cnn22_12675cnn22_12677*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cnn22_layer_call_and_return_conditional_losses_123122
cnn22/StatefulPartitionedCall�
max_pool2/PartitionedCallPartitionedCall&cnn22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_max_pool2_layer_call_and_return_conditional_losses_121672
max_pool2/PartitionedCall�
dropout_4/PartitionedCallPartitionedCall"max_pool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_123462
dropout_4/PartitionedCall�
cnn3/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0
cnn3_12682
cnn3_12684*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_cnn3_layer_call_and_return_conditional_losses_123702
cnn3/StatefulPartitionedCall�
cnn32/StatefulPartitionedCallStatefulPartitionedCall%cnn3/StatefulPartitionedCall:output:0cnn32_12687cnn32_12689*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cnn32_layer_call_and_return_conditional_losses_123972
cnn32/StatefulPartitionedCall�
max_pool3/PartitionedCallPartitionedCall&cnn32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_max_pool3_layer_call_and_return_conditional_losses_121792
max_pool3/PartitionedCall�
dropout_5/PartitionedCallPartitionedCall"max_pool3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_124312
dropout_5/PartitionedCall�
flatten_1/PartitionedCallPartitionedCall"dropout_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_124502
flatten_1/PartitionedCall�
fc1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0	fc1_12695	fc1_12697*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_fc1_layer_call_and_return_conditional_losses_124692
fc1/StatefulPartitionedCall�
final/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0final_12700final_12702*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_final_layer_call_and_return_conditional_losses_124962
final/StatefulPartitionedCall�
IdentityIdentity&final/StatefulPartitionedCall:output:0^cnn1/StatefulPartitionedCall^cnn12/StatefulPartitionedCall^cnn2/StatefulPartitionedCall^cnn22/StatefulPartitionedCall^cnn3/StatefulPartitionedCall^cnn32/StatefulPartitionedCall^fc1/StatefulPartitionedCall^final/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������  ::::::::::::::::2<
cnn1/StatefulPartitionedCallcnn1/StatefulPartitionedCall2>
cnn12/StatefulPartitionedCallcnn12/StatefulPartitionedCall2<
cnn2/StatefulPartitionedCallcnn2/StatefulPartitionedCall2>
cnn22/StatefulPartitionedCallcnn22/StatefulPartitionedCall2<
cnn3/StatefulPartitionedCallcnn3/StatefulPartitionedCall2>
cnn32/StatefulPartitionedCallcnn32/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2>
final/StatefulPartitionedCallfinal/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
z
%__inference_final_layer_call_fn_13271

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_final_layer_call_and_return_conditional_losses_124962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_13138

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
?__inference_cnn3_layer_call_and_return_conditional_losses_13164

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
y
$__inference_cnn2_layer_call_fn_13106

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_cnn2_layer_call_and_return_conditional_losses_122852
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
c
D__inference_dropout_3_layer_call_and_return_conditional_losses_13071

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:��������� 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� 2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:��������� 2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:��������� 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
*__inference_base_cifar_layer_call_fn_12741
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_base_cifar_layer_call_and_return_conditional_losses_127062
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������  ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_2
�p
�
__inference__traced_save_13465
file_prefix*
&savev2_cnn1_kernel_read_readvariableop(
$savev2_cnn1_bias_read_readvariableop+
'savev2_cnn12_kernel_read_readvariableop)
%savev2_cnn12_bias_read_readvariableop*
&savev2_cnn2_kernel_read_readvariableop(
$savev2_cnn2_bias_read_readvariableop+
'savev2_cnn22_kernel_read_readvariableop)
%savev2_cnn22_bias_read_readvariableop*
&savev2_cnn3_kernel_read_readvariableop(
$savev2_cnn3_bias_read_readvariableop+
'savev2_cnn32_kernel_read_readvariableop)
%savev2_cnn32_bias_read_readvariableop)
%savev2_fc1_kernel_read_readvariableop'
#savev2_fc1_bias_read_readvariableop+
'savev2_final_kernel_read_readvariableop)
%savev2_final_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop1
-savev2_adam_cnn1_kernel_m_read_readvariableop/
+savev2_adam_cnn1_bias_m_read_readvariableop2
.savev2_adam_cnn12_kernel_m_read_readvariableop0
,savev2_adam_cnn12_bias_m_read_readvariableop1
-savev2_adam_cnn2_kernel_m_read_readvariableop/
+savev2_adam_cnn2_bias_m_read_readvariableop2
.savev2_adam_cnn22_kernel_m_read_readvariableop0
,savev2_adam_cnn22_bias_m_read_readvariableop1
-savev2_adam_cnn3_kernel_m_read_readvariableop/
+savev2_adam_cnn3_bias_m_read_readvariableop2
.savev2_adam_cnn32_kernel_m_read_readvariableop0
,savev2_adam_cnn32_bias_m_read_readvariableop0
,savev2_adam_fc1_kernel_m_read_readvariableop.
*savev2_adam_fc1_bias_m_read_readvariableop2
.savev2_adam_final_kernel_m_read_readvariableop0
,savev2_adam_final_bias_m_read_readvariableop1
-savev2_adam_cnn1_kernel_v_read_readvariableop/
+savev2_adam_cnn1_bias_v_read_readvariableop2
.savev2_adam_cnn12_kernel_v_read_readvariableop0
,savev2_adam_cnn12_bias_v_read_readvariableop1
-savev2_adam_cnn2_kernel_v_read_readvariableop/
+savev2_adam_cnn2_bias_v_read_readvariableop2
.savev2_adam_cnn22_kernel_v_read_readvariableop0
,savev2_adam_cnn22_bias_v_read_readvariableop1
-savev2_adam_cnn3_kernel_v_read_readvariableop/
+savev2_adam_cnn3_bias_v_read_readvariableop2
.savev2_adam_cnn32_kernel_v_read_readvariableop0
,savev2_adam_cnn32_bias_v_read_readvariableop0
,savev2_adam_fc1_kernel_v_read_readvariableop.
*savev2_adam_fc1_bias_v_read_readvariableop2
.savev2_adam_final_kernel_v_read_readvariableop0
,savev2_adam_final_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename� 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value�B�:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_cnn1_kernel_read_readvariableop$savev2_cnn1_bias_read_readvariableop'savev2_cnn12_kernel_read_readvariableop%savev2_cnn12_bias_read_readvariableop&savev2_cnn2_kernel_read_readvariableop$savev2_cnn2_bias_read_readvariableop'savev2_cnn22_kernel_read_readvariableop%savev2_cnn22_bias_read_readvariableop&savev2_cnn3_kernel_read_readvariableop$savev2_cnn3_bias_read_readvariableop'savev2_cnn32_kernel_read_readvariableop%savev2_cnn32_bias_read_readvariableop%savev2_fc1_kernel_read_readvariableop#savev2_fc1_bias_read_readvariableop'savev2_final_kernel_read_readvariableop%savev2_final_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop-savev2_adam_cnn1_kernel_m_read_readvariableop+savev2_adam_cnn1_bias_m_read_readvariableop.savev2_adam_cnn12_kernel_m_read_readvariableop,savev2_adam_cnn12_bias_m_read_readvariableop-savev2_adam_cnn2_kernel_m_read_readvariableop+savev2_adam_cnn2_bias_m_read_readvariableop.savev2_adam_cnn22_kernel_m_read_readvariableop,savev2_adam_cnn22_bias_m_read_readvariableop-savev2_adam_cnn3_kernel_m_read_readvariableop+savev2_adam_cnn3_bias_m_read_readvariableop.savev2_adam_cnn32_kernel_m_read_readvariableop,savev2_adam_cnn32_bias_m_read_readvariableop,savev2_adam_fc1_kernel_m_read_readvariableop*savev2_adam_fc1_bias_m_read_readvariableop.savev2_adam_final_kernel_m_read_readvariableop,savev2_adam_final_bias_m_read_readvariableop-savev2_adam_cnn1_kernel_v_read_readvariableop+savev2_adam_cnn1_bias_v_read_readvariableop.savev2_adam_cnn12_kernel_v_read_readvariableop,savev2_adam_cnn12_bias_v_read_readvariableop-savev2_adam_cnn2_kernel_v_read_readvariableop+savev2_adam_cnn2_bias_v_read_readvariableop.savev2_adam_cnn22_kernel_v_read_readvariableop,savev2_adam_cnn22_bias_v_read_readvariableop-savev2_adam_cnn3_kernel_v_read_readvariableop+savev2_adam_cnn3_bias_v_read_readvariableop.savev2_adam_cnn32_kernel_v_read_readvariableop,savev2_adam_cnn32_bias_v_read_readvariableop,savev2_adam_fc1_kernel_v_read_readvariableop*savev2_adam_fc1_bias_v_read_readvariableop.savev2_adam_final_kernel_v_read_readvariableop,savev2_adam_final_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *H
dtypes>
<2:	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : :  : : @:@:@@:@:@�:�:��:�:
��:�:	�
:
: : : : : : : : : : : :  : : @:@:@@:@:@�:�:��:�:
��:�:	�
:
: : :  : : @:@:@@:@:@�:�:��:�:
��:�:	�
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-	)
'
_output_shapes
:@�:!


_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:, (
&
_output_shapes
:@@: !

_output_shapes
:@:-")
'
_output_shapes
:@�:!#

_output_shapes	
:�:.$*
(
_output_shapes
:��:!%

_output_shapes	
:�:&&"
 
_output_shapes
:
��:!'

_output_shapes	
:�:%(!

_output_shapes
:	�
: )

_output_shapes
:
:,*(
&
_output_shapes
: : +

_output_shapes
: :,,(
&
_output_shapes
:  : -

_output_shapes
: :,.(
&
_output_shapes
: @: /

_output_shapes
:@:,0(
&
_output_shapes
:@@: 1

_output_shapes
:@:-2)
'
_output_shapes
:@�:!3

_output_shapes	
:�:.4*
(
_output_shapes
:��:!5

_output_shapes	
:�:&6"
 
_output_shapes
:
��:!7

_output_shapes	
:�:%8!

_output_shapes
:	�
: 9

_output_shapes
:
::

_output_shapes
: 
�
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_13076

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:��������� 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:��������� 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
@__inference_cnn12_layer_call_and_return_conditional_losses_12227

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������   2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_12341

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
?__inference_cnn2_layer_call_and_return_conditional_losses_13097

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�o
�	
E__inference_base_cifar_layer_call_and_return_conditional_losses_12877

inputs'
#cnn1_conv2d_readvariableop_resource(
$cnn1_biasadd_readvariableop_resource(
$cnn12_conv2d_readvariableop_resource)
%cnn12_biasadd_readvariableop_resource'
#cnn2_conv2d_readvariableop_resource(
$cnn2_biasadd_readvariableop_resource(
$cnn22_conv2d_readvariableop_resource)
%cnn22_biasadd_readvariableop_resource'
#cnn3_conv2d_readvariableop_resource(
$cnn3_biasadd_readvariableop_resource(
$cnn32_conv2d_readvariableop_resource)
%cnn32_biasadd_readvariableop_resource&
"fc1_matmul_readvariableop_resource'
#fc1_biasadd_readvariableop_resource(
$final_matmul_readvariableop_resource)
%final_biasadd_readvariableop_resource
identity��cnn1/BiasAdd/ReadVariableOp�cnn1/Conv2D/ReadVariableOp�cnn12/BiasAdd/ReadVariableOp�cnn12/Conv2D/ReadVariableOp�cnn2/BiasAdd/ReadVariableOp�cnn2/Conv2D/ReadVariableOp�cnn22/BiasAdd/ReadVariableOp�cnn22/Conv2D/ReadVariableOp�cnn3/BiasAdd/ReadVariableOp�cnn3/Conv2D/ReadVariableOp�cnn32/BiasAdd/ReadVariableOp�cnn32/Conv2D/ReadVariableOp�fc1/BiasAdd/ReadVariableOp�fc1/MatMul/ReadVariableOp�final/BiasAdd/ReadVariableOp�final/MatMul/ReadVariableOp�
cnn1/Conv2D/ReadVariableOpReadVariableOp#cnn1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
cnn1/Conv2D/ReadVariableOp�
cnn1/Conv2DConv2Dinputs"cnn1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
cnn1/Conv2D�
cnn1/BiasAdd/ReadVariableOpReadVariableOp$cnn1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
cnn1/BiasAdd/ReadVariableOp�
cnn1/BiasAddBiasAddcnn1/Conv2D:output:0#cnn1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
cnn1/BiasAddo
	cnn1/ReluRelucnn1/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
	cnn1/Relu�
cnn12/Conv2D/ReadVariableOpReadVariableOp$cnn12_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
cnn12/Conv2D/ReadVariableOp�
cnn12/Conv2DConv2Dcnn1/Relu:activations:0#cnn12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
cnn12/Conv2D�
cnn12/BiasAdd/ReadVariableOpReadVariableOp%cnn12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
cnn12/BiasAdd/ReadVariableOp�
cnn12/BiasAddBiasAddcnn12/Conv2D:output:0$cnn12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
cnn12/BiasAddr

cnn12/ReluRelucnn12/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2

cnn12/Relu�
max_pool1/MaxPoolMaxPoolcnn12/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pool1/MaxPoolw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout_3/dropout/Const�
dropout_3/dropout/MulMulmax_pool1/MaxPool:output:0 dropout_3/dropout/Const:output:0*
T0*/
_output_shapes
:��������� 2
dropout_3/dropout/Mul|
dropout_3/dropout/ShapeShapemax_pool1/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape�
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype020
.dropout_3/dropout/random_uniform/RandomUniform�
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2"
 dropout_3/dropout/GreaterEqual/y�
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� 2 
dropout_3/dropout/GreaterEqual�
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:��������� 2
dropout_3/dropout/Cast�
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*/
_output_shapes
:��������� 2
dropout_3/dropout/Mul_1�
cnn2/Conv2D/ReadVariableOpReadVariableOp#cnn2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
cnn2/Conv2D/ReadVariableOp�
cnn2/Conv2DConv2Ddropout_3/dropout/Mul_1:z:0"cnn2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
cnn2/Conv2D�
cnn2/BiasAdd/ReadVariableOpReadVariableOp$cnn2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
cnn2/BiasAdd/ReadVariableOp�
cnn2/BiasAddBiasAddcnn2/Conv2D:output:0#cnn2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
cnn2/BiasAddo
	cnn2/ReluRelucnn2/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
	cnn2/Relu�
cnn22/Conv2D/ReadVariableOpReadVariableOp$cnn22_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
cnn22/Conv2D/ReadVariableOp�
cnn22/Conv2DConv2Dcnn2/Relu:activations:0#cnn22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
cnn22/Conv2D�
cnn22/BiasAdd/ReadVariableOpReadVariableOp%cnn22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
cnn22/BiasAdd/ReadVariableOp�
cnn22/BiasAddBiasAddcnn22/Conv2D:output:0$cnn22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
cnn22/BiasAddr

cnn22/ReluRelucnn22/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2

cnn22/Relu�
max_pool2/MaxPoolMaxPoolcnn22/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pool2/MaxPoolw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout_4/dropout/Const�
dropout_4/dropout/MulMulmax_pool2/MaxPool:output:0 dropout_4/dropout/Const:output:0*
T0*/
_output_shapes
:���������@2
dropout_4/dropout/Mul|
dropout_4/dropout/ShapeShapemax_pool2/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shape�
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype020
.dropout_4/dropout/random_uniform/RandomUniform�
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2"
 dropout_4/dropout/GreaterEqual/y�
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@2 
dropout_4/dropout/GreaterEqual�
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@2
dropout_4/dropout/Cast�
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*/
_output_shapes
:���������@2
dropout_4/dropout/Mul_1�
cnn3/Conv2D/ReadVariableOpReadVariableOp#cnn3_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
cnn3/Conv2D/ReadVariableOp�
cnn3/Conv2DConv2Ddropout_4/dropout/Mul_1:z:0"cnn3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
cnn3/Conv2D�
cnn3/BiasAdd/ReadVariableOpReadVariableOp$cnn3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
cnn3/BiasAdd/ReadVariableOp�
cnn3/BiasAddBiasAddcnn3/Conv2D:output:0#cnn3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
cnn3/BiasAddp
	cnn3/ReluRelucnn3/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
	cnn3/Relu�
cnn32/Conv2D/ReadVariableOpReadVariableOp$cnn32_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
cnn32/Conv2D/ReadVariableOp�
cnn32/Conv2DConv2Dcnn3/Relu:activations:0#cnn32/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
cnn32/Conv2D�
cnn32/BiasAdd/ReadVariableOpReadVariableOp%cnn32_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
cnn32/BiasAdd/ReadVariableOp�
cnn32/BiasAddBiasAddcnn32/Conv2D:output:0$cnn32/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
cnn32/BiasAdds

cnn32/ReluRelucnn32/BiasAdd:output:0*
T0*0
_output_shapes
:����������2

cnn32/Relu�
max_pool3/MaxPoolMaxPoolcnn32/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pool3/MaxPoolw
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout_5/dropout/Const�
dropout_5/dropout/MulMulmax_pool3/MaxPool:output:0 dropout_5/dropout/Const:output:0*
T0*0
_output_shapes
:����������2
dropout_5/dropout/Mul|
dropout_5/dropout/ShapeShapemax_pool3/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_5/dropout/Shape�
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype020
.dropout_5/dropout/random_uniform/RandomUniform�
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2"
 dropout_5/dropout/GreaterEqual/y�
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������2 
dropout_5/dropout/GreaterEqual�
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������2
dropout_5/dropout/Cast�
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*0
_output_shapes
:����������2
dropout_5/dropout/Mul_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_1/Const�
flatten_1/ReshapeReshapedropout_5/dropout/Mul_1:z:0flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_1/Reshape�
fc1/MatMul/ReadVariableOpReadVariableOp"fc1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
fc1/MatMul/ReadVariableOp�

fc1/MatMulMatMulflatten_1/Reshape:output:0!fc1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

fc1/MatMul�
fc1/BiasAdd/ReadVariableOpReadVariableOp#fc1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
fc1/BiasAdd/ReadVariableOp�
fc1/BiasAddBiasAddfc1/MatMul:product:0"fc1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
fc1/BiasAdde
fc1/ReluRelufc1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2

fc1/Relu�
final/MatMul/ReadVariableOpReadVariableOp$final_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02
final/MatMul/ReadVariableOp�
final/MatMulMatMulfc1/Relu:activations:0#final/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
final/MatMul�
final/BiasAdd/ReadVariableOpReadVariableOp%final_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
final/BiasAdd/ReadVariableOp�
final/BiasAddBiasAddfinal/MatMul:product:0$final/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
final/BiasAdds
final/SoftmaxSoftmaxfinal/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
final/Softmax�
IdentityIdentityfinal/Softmax:softmax:0^cnn1/BiasAdd/ReadVariableOp^cnn1/Conv2D/ReadVariableOp^cnn12/BiasAdd/ReadVariableOp^cnn12/Conv2D/ReadVariableOp^cnn2/BiasAdd/ReadVariableOp^cnn2/Conv2D/ReadVariableOp^cnn22/BiasAdd/ReadVariableOp^cnn22/Conv2D/ReadVariableOp^cnn3/BiasAdd/ReadVariableOp^cnn3/Conv2D/ReadVariableOp^cnn32/BiasAdd/ReadVariableOp^cnn32/Conv2D/ReadVariableOp^fc1/BiasAdd/ReadVariableOp^fc1/MatMul/ReadVariableOp^final/BiasAdd/ReadVariableOp^final/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������  ::::::::::::::::2:
cnn1/BiasAdd/ReadVariableOpcnn1/BiasAdd/ReadVariableOp28
cnn1/Conv2D/ReadVariableOpcnn1/Conv2D/ReadVariableOp2<
cnn12/BiasAdd/ReadVariableOpcnn12/BiasAdd/ReadVariableOp2:
cnn12/Conv2D/ReadVariableOpcnn12/Conv2D/ReadVariableOp2:
cnn2/BiasAdd/ReadVariableOpcnn2/BiasAdd/ReadVariableOp28
cnn2/Conv2D/ReadVariableOpcnn2/Conv2D/ReadVariableOp2<
cnn22/BiasAdd/ReadVariableOpcnn22/BiasAdd/ReadVariableOp2:
cnn22/Conv2D/ReadVariableOpcnn22/Conv2D/ReadVariableOp2:
cnn3/BiasAdd/ReadVariableOpcnn3/BiasAdd/ReadVariableOp28
cnn3/Conv2D/ReadVariableOpcnn3/Conv2D/ReadVariableOp2<
cnn32/BiasAdd/ReadVariableOpcnn32/BiasAdd/ReadVariableOp2:
cnn32/Conv2D/ReadVariableOpcnn32/Conv2D/ReadVariableOp28
fc1/BiasAdd/ReadVariableOpfc1/BiasAdd/ReadVariableOp26
fc1/MatMul/ReadVariableOpfc1/MatMul/ReadVariableOp2<
final/BiasAdd/ReadVariableOpfinal/BiasAdd/ReadVariableOp2:
final/MatMul/ReadVariableOpfinal/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
E
)__inference_dropout_5_layer_call_fn_13220

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_124312
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
D__inference_max_pool3_layer_call_and_return_conditional_losses_12179

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�	
�
>__inference_fc1_layer_call_and_return_conditional_losses_12469

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
?__inference_cnn2_layer_call_and_return_conditional_losses_12285

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
@__inference_cnn22_layer_call_and_return_conditional_losses_12312

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
>__inference_fc1_layer_call_and_return_conditional_losses_13242

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
@__inference_final_layer_call_and_return_conditional_losses_12496

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������
2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
D__inference_max_pool1_layer_call_and_return_conditional_losses_12155

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�	
�
@__inference_final_layer_call_and_return_conditional_losses_13262

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������
2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
z
%__inference_cnn12_layer_call_fn_13059

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cnn12_layer_call_and_return_conditional_losses_122272
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������   ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_12450

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
E
)__inference_max_pool3_layer_call_fn_12185

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_max_pool3_layer_call_and_return_conditional_losses_121792
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
z
%__inference_cnn22_layer_call_fn_13126

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cnn22_layer_call_and_return_conditional_losses_123122
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
E
)__inference_dropout_4_layer_call_fn_13153

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_123462
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
y
$__inference_cnn3_layer_call_fn_13173

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_cnn3_layer_call_and_return_conditional_losses_123702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�
!__inference__traced_restore_13646
file_prefix 
assignvariableop_cnn1_kernel 
assignvariableop_1_cnn1_bias#
assignvariableop_2_cnn12_kernel!
assignvariableop_3_cnn12_bias"
assignvariableop_4_cnn2_kernel 
assignvariableop_5_cnn2_bias#
assignvariableop_6_cnn22_kernel!
assignvariableop_7_cnn22_bias"
assignvariableop_8_cnn3_kernel 
assignvariableop_9_cnn3_bias$
 assignvariableop_10_cnn32_kernel"
assignvariableop_11_cnn32_bias"
assignvariableop_12_fc1_kernel 
assignvariableop_13_fc1_bias$
 assignvariableop_14_final_kernel"
assignvariableop_15_final_bias!
assignvariableop_16_adam_iter#
assignvariableop_17_adam_beta_1#
assignvariableop_18_adam_beta_2"
assignvariableop_19_adam_decay*
&assignvariableop_20_adam_learning_rate
assignvariableop_21_total
assignvariableop_22_count
assignvariableop_23_total_1
assignvariableop_24_count_1*
&assignvariableop_25_adam_cnn1_kernel_m(
$assignvariableop_26_adam_cnn1_bias_m+
'assignvariableop_27_adam_cnn12_kernel_m)
%assignvariableop_28_adam_cnn12_bias_m*
&assignvariableop_29_adam_cnn2_kernel_m(
$assignvariableop_30_adam_cnn2_bias_m+
'assignvariableop_31_adam_cnn22_kernel_m)
%assignvariableop_32_adam_cnn22_bias_m*
&assignvariableop_33_adam_cnn3_kernel_m(
$assignvariableop_34_adam_cnn3_bias_m+
'assignvariableop_35_adam_cnn32_kernel_m)
%assignvariableop_36_adam_cnn32_bias_m)
%assignvariableop_37_adam_fc1_kernel_m'
#assignvariableop_38_adam_fc1_bias_m+
'assignvariableop_39_adam_final_kernel_m)
%assignvariableop_40_adam_final_bias_m*
&assignvariableop_41_adam_cnn1_kernel_v(
$assignvariableop_42_adam_cnn1_bias_v+
'assignvariableop_43_adam_cnn12_kernel_v)
%assignvariableop_44_adam_cnn12_bias_v*
&assignvariableop_45_adam_cnn2_kernel_v(
$assignvariableop_46_adam_cnn2_bias_v+
'assignvariableop_47_adam_cnn22_kernel_v)
%assignvariableop_48_adam_cnn22_bias_v*
&assignvariableop_49_adam_cnn3_kernel_v(
$assignvariableop_50_adam_cnn3_bias_v+
'assignvariableop_51_adam_cnn32_kernel_v)
%assignvariableop_52_adam_cnn32_bias_v)
%assignvariableop_53_adam_fc1_kernel_v'
#assignvariableop_54_adam_fc1_bias_v+
'assignvariableop_55_adam_final_kernel_v)
%assignvariableop_56_adam_final_bias_v
identity_58��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9� 
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value�B�:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_cnn1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_cnn1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_cnn12_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_cnn12_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_cnn2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_cnn2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_cnn22_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_cnn22_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_cnn3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_cnn3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp assignvariableop_10_cnn32_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_cnn32_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_fc1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_fc1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp assignvariableop_14_final_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_final_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp&assignvariableop_25_adam_cnn1_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp$assignvariableop_26_adam_cnn1_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_cnn12_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_cnn12_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp&assignvariableop_29_adam_cnn2_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp$assignvariableop_30_adam_cnn2_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_cnn22_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_cnn22_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp&assignvariableop_33_adam_cnn3_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp$assignvariableop_34_adam_cnn3_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_cnn32_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp%assignvariableop_36_adam_cnn32_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp%assignvariableop_37_adam_fc1_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp#assignvariableop_38_adam_fc1_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp'assignvariableop_39_adam_final_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp%assignvariableop_40_adam_final_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp&assignvariableop_41_adam_cnn1_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp$assignvariableop_42_adam_cnn1_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp'assignvariableop_43_adam_cnn12_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp%assignvariableop_44_adam_cnn12_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp&assignvariableop_45_adam_cnn2_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp$assignvariableop_46_adam_cnn2_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp'assignvariableop_47_adam_cnn22_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp%assignvariableop_48_adam_cnn22_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp&assignvariableop_49_adam_cnn3_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp$assignvariableop_50_adam_cnn3_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp'assignvariableop_51_adam_cnn32_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp%assignvariableop_52_adam_cnn32_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp%assignvariableop_53_adam_fc1_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOp#assignvariableop_54_adam_fc1_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOp'assignvariableop_55_adam_final_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOp%assignvariableop_56_adam_final_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_569
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_57�

Identity_58IdentityIdentity_57:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_58"#
identity_58Identity_58:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
b
)__inference_dropout_4_layer_call_fn_13148

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_123412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�A
�
E__inference_base_cifar_layer_call_and_return_conditional_losses_12513
input_2

cnn1_12211

cnn1_12213
cnn12_12238
cnn12_12240

cnn2_12296

cnn2_12298
cnn22_12323
cnn22_12325

cnn3_12381

cnn3_12383
cnn32_12408
cnn32_12410
	fc1_12480
	fc1_12482
final_12507
final_12509
identity��cnn1/StatefulPartitionedCall�cnn12/StatefulPartitionedCall�cnn2/StatefulPartitionedCall�cnn22/StatefulPartitionedCall�cnn3/StatefulPartitionedCall�cnn32/StatefulPartitionedCall�!dropout_3/StatefulPartitionedCall�!dropout_4/StatefulPartitionedCall�!dropout_5/StatefulPartitionedCall�fc1/StatefulPartitionedCall�final/StatefulPartitionedCall�
cnn1/StatefulPartitionedCallStatefulPartitionedCallinput_2
cnn1_12211
cnn1_12213*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_cnn1_layer_call_and_return_conditional_losses_122002
cnn1/StatefulPartitionedCall�
cnn12/StatefulPartitionedCallStatefulPartitionedCall%cnn1/StatefulPartitionedCall:output:0cnn12_12238cnn12_12240*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cnn12_layer_call_and_return_conditional_losses_122272
cnn12/StatefulPartitionedCall�
max_pool1/PartitionedCallPartitionedCall&cnn12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_max_pool1_layer_call_and_return_conditional_losses_121552
max_pool1/PartitionedCall�
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall"max_pool1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_122562#
!dropout_3/StatefulPartitionedCall�
cnn2/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0
cnn2_12296
cnn2_12298*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_cnn2_layer_call_and_return_conditional_losses_122852
cnn2/StatefulPartitionedCall�
cnn22/StatefulPartitionedCallStatefulPartitionedCall%cnn2/StatefulPartitionedCall:output:0cnn22_12323cnn22_12325*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cnn22_layer_call_and_return_conditional_losses_123122
cnn22/StatefulPartitionedCall�
max_pool2/PartitionedCallPartitionedCall&cnn22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_max_pool2_layer_call_and_return_conditional_losses_121672
max_pool2/PartitionedCall�
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall"max_pool2/PartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_123412#
!dropout_4/StatefulPartitionedCall�
cnn3/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0
cnn3_12381
cnn3_12383*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_cnn3_layer_call_and_return_conditional_losses_123702
cnn3/StatefulPartitionedCall�
cnn32/StatefulPartitionedCallStatefulPartitionedCall%cnn3/StatefulPartitionedCall:output:0cnn32_12408cnn32_12410*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cnn32_layer_call_and_return_conditional_losses_123972
cnn32/StatefulPartitionedCall�
max_pool3/PartitionedCallPartitionedCall&cnn32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_max_pool3_layer_call_and_return_conditional_losses_121792
max_pool3/PartitionedCall�
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall"max_pool3/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_124262#
!dropout_5/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall*dropout_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_124502
flatten_1/PartitionedCall�
fc1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0	fc1_12480	fc1_12482*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_fc1_layer_call_and_return_conditional_losses_124692
fc1/StatefulPartitionedCall�
final/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0final_12507final_12509*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_final_layer_call_and_return_conditional_losses_124962
final/StatefulPartitionedCall�
IdentityIdentity&final/StatefulPartitionedCall:output:0^cnn1/StatefulPartitionedCall^cnn12/StatefulPartitionedCall^cnn2/StatefulPartitionedCall^cnn22/StatefulPartitionedCall^cnn3/StatefulPartitionedCall^cnn32/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall^fc1/StatefulPartitionedCall^final/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������  ::::::::::::::::2<
cnn1/StatefulPartitionedCallcnn1/StatefulPartitionedCall2>
cnn12/StatefulPartitionedCallcnn12/StatefulPartitionedCall2<
cnn2/StatefulPartitionedCallcnn2/StatefulPartitionedCall2>
cnn22/StatefulPartitionedCallcnn22/StatefulPartitionedCall2<
cnn3/StatefulPartitionedCallcnn3/StatefulPartitionedCall2>
cnn32/StatefulPartitionedCallcnn32/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2>
final/StatefulPartitionedCallfinal/StatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_2
�g
�
 __inference__wrapped_model_12149
input_22
.base_cifar_cnn1_conv2d_readvariableop_resource3
/base_cifar_cnn1_biasadd_readvariableop_resource3
/base_cifar_cnn12_conv2d_readvariableop_resource4
0base_cifar_cnn12_biasadd_readvariableop_resource2
.base_cifar_cnn2_conv2d_readvariableop_resource3
/base_cifar_cnn2_biasadd_readvariableop_resource3
/base_cifar_cnn22_conv2d_readvariableop_resource4
0base_cifar_cnn22_biasadd_readvariableop_resource2
.base_cifar_cnn3_conv2d_readvariableop_resource3
/base_cifar_cnn3_biasadd_readvariableop_resource3
/base_cifar_cnn32_conv2d_readvariableop_resource4
0base_cifar_cnn32_biasadd_readvariableop_resource1
-base_cifar_fc1_matmul_readvariableop_resource2
.base_cifar_fc1_biasadd_readvariableop_resource3
/base_cifar_final_matmul_readvariableop_resource4
0base_cifar_final_biasadd_readvariableop_resource
identity��&base_cifar/cnn1/BiasAdd/ReadVariableOp�%base_cifar/cnn1/Conv2D/ReadVariableOp�'base_cifar/cnn12/BiasAdd/ReadVariableOp�&base_cifar/cnn12/Conv2D/ReadVariableOp�&base_cifar/cnn2/BiasAdd/ReadVariableOp�%base_cifar/cnn2/Conv2D/ReadVariableOp�'base_cifar/cnn22/BiasAdd/ReadVariableOp�&base_cifar/cnn22/Conv2D/ReadVariableOp�&base_cifar/cnn3/BiasAdd/ReadVariableOp�%base_cifar/cnn3/Conv2D/ReadVariableOp�'base_cifar/cnn32/BiasAdd/ReadVariableOp�&base_cifar/cnn32/Conv2D/ReadVariableOp�%base_cifar/fc1/BiasAdd/ReadVariableOp�$base_cifar/fc1/MatMul/ReadVariableOp�'base_cifar/final/BiasAdd/ReadVariableOp�&base_cifar/final/MatMul/ReadVariableOp�
%base_cifar/cnn1/Conv2D/ReadVariableOpReadVariableOp.base_cifar_cnn1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02'
%base_cifar/cnn1/Conv2D/ReadVariableOp�
base_cifar/cnn1/Conv2DConv2Dinput_2-base_cifar/cnn1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
base_cifar/cnn1/Conv2D�
&base_cifar/cnn1/BiasAdd/ReadVariableOpReadVariableOp/base_cifar_cnn1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&base_cifar/cnn1/BiasAdd/ReadVariableOp�
base_cifar/cnn1/BiasAddBiasAddbase_cifar/cnn1/Conv2D:output:0.base_cifar/cnn1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
base_cifar/cnn1/BiasAdd�
base_cifar/cnn1/ReluRelu base_cifar/cnn1/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
base_cifar/cnn1/Relu�
&base_cifar/cnn12/Conv2D/ReadVariableOpReadVariableOp/base_cifar_cnn12_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02(
&base_cifar/cnn12/Conv2D/ReadVariableOp�
base_cifar/cnn12/Conv2DConv2D"base_cifar/cnn1/Relu:activations:0.base_cifar/cnn12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
base_cifar/cnn12/Conv2D�
'base_cifar/cnn12/BiasAdd/ReadVariableOpReadVariableOp0base_cifar_cnn12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'base_cifar/cnn12/BiasAdd/ReadVariableOp�
base_cifar/cnn12/BiasAddBiasAdd base_cifar/cnn12/Conv2D:output:0/base_cifar/cnn12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
base_cifar/cnn12/BiasAdd�
base_cifar/cnn12/ReluRelu!base_cifar/cnn12/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
base_cifar/cnn12/Relu�
base_cifar/max_pool1/MaxPoolMaxPool#base_cifar/cnn12/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
base_cifar/max_pool1/MaxPool�
base_cifar/dropout_3/IdentityIdentity%base_cifar/max_pool1/MaxPool:output:0*
T0*/
_output_shapes
:��������� 2
base_cifar/dropout_3/Identity�
%base_cifar/cnn2/Conv2D/ReadVariableOpReadVariableOp.base_cifar_cnn2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02'
%base_cifar/cnn2/Conv2D/ReadVariableOp�
base_cifar/cnn2/Conv2DConv2D&base_cifar/dropout_3/Identity:output:0-base_cifar/cnn2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
base_cifar/cnn2/Conv2D�
&base_cifar/cnn2/BiasAdd/ReadVariableOpReadVariableOp/base_cifar_cnn2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&base_cifar/cnn2/BiasAdd/ReadVariableOp�
base_cifar/cnn2/BiasAddBiasAddbase_cifar/cnn2/Conv2D:output:0.base_cifar/cnn2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
base_cifar/cnn2/BiasAdd�
base_cifar/cnn2/ReluRelu base_cifar/cnn2/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
base_cifar/cnn2/Relu�
&base_cifar/cnn22/Conv2D/ReadVariableOpReadVariableOp/base_cifar_cnn22_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02(
&base_cifar/cnn22/Conv2D/ReadVariableOp�
base_cifar/cnn22/Conv2DConv2D"base_cifar/cnn2/Relu:activations:0.base_cifar/cnn22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
base_cifar/cnn22/Conv2D�
'base_cifar/cnn22/BiasAdd/ReadVariableOpReadVariableOp0base_cifar_cnn22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'base_cifar/cnn22/BiasAdd/ReadVariableOp�
base_cifar/cnn22/BiasAddBiasAdd base_cifar/cnn22/Conv2D:output:0/base_cifar/cnn22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
base_cifar/cnn22/BiasAdd�
base_cifar/cnn22/ReluRelu!base_cifar/cnn22/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
base_cifar/cnn22/Relu�
base_cifar/max_pool2/MaxPoolMaxPool#base_cifar/cnn22/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
base_cifar/max_pool2/MaxPool�
base_cifar/dropout_4/IdentityIdentity%base_cifar/max_pool2/MaxPool:output:0*
T0*/
_output_shapes
:���������@2
base_cifar/dropout_4/Identity�
%base_cifar/cnn3/Conv2D/ReadVariableOpReadVariableOp.base_cifar_cnn3_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02'
%base_cifar/cnn3/Conv2D/ReadVariableOp�
base_cifar/cnn3/Conv2DConv2D&base_cifar/dropout_4/Identity:output:0-base_cifar/cnn3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
base_cifar/cnn3/Conv2D�
&base_cifar/cnn3/BiasAdd/ReadVariableOpReadVariableOp/base_cifar_cnn3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&base_cifar/cnn3/BiasAdd/ReadVariableOp�
base_cifar/cnn3/BiasAddBiasAddbase_cifar/cnn3/Conv2D:output:0.base_cifar/cnn3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
base_cifar/cnn3/BiasAdd�
base_cifar/cnn3/ReluRelu base_cifar/cnn3/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
base_cifar/cnn3/Relu�
&base_cifar/cnn32/Conv2D/ReadVariableOpReadVariableOp/base_cifar_cnn32_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02(
&base_cifar/cnn32/Conv2D/ReadVariableOp�
base_cifar/cnn32/Conv2DConv2D"base_cifar/cnn3/Relu:activations:0.base_cifar/cnn32/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
base_cifar/cnn32/Conv2D�
'base_cifar/cnn32/BiasAdd/ReadVariableOpReadVariableOp0base_cifar_cnn32_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'base_cifar/cnn32/BiasAdd/ReadVariableOp�
base_cifar/cnn32/BiasAddBiasAdd base_cifar/cnn32/Conv2D:output:0/base_cifar/cnn32/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
base_cifar/cnn32/BiasAdd�
base_cifar/cnn32/ReluRelu!base_cifar/cnn32/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
base_cifar/cnn32/Relu�
base_cifar/max_pool3/MaxPoolMaxPool#base_cifar/cnn32/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
base_cifar/max_pool3/MaxPool�
base_cifar/dropout_5/IdentityIdentity%base_cifar/max_pool3/MaxPool:output:0*
T0*0
_output_shapes
:����������2
base_cifar/dropout_5/Identity�
base_cifar/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
base_cifar/flatten_1/Const�
base_cifar/flatten_1/ReshapeReshape&base_cifar/dropout_5/Identity:output:0#base_cifar/flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
base_cifar/flatten_1/Reshape�
$base_cifar/fc1/MatMul/ReadVariableOpReadVariableOp-base_cifar_fc1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02&
$base_cifar/fc1/MatMul/ReadVariableOp�
base_cifar/fc1/MatMulMatMul%base_cifar/flatten_1/Reshape:output:0,base_cifar/fc1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
base_cifar/fc1/MatMul�
%base_cifar/fc1/BiasAdd/ReadVariableOpReadVariableOp.base_cifar_fc1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%base_cifar/fc1/BiasAdd/ReadVariableOp�
base_cifar/fc1/BiasAddBiasAddbase_cifar/fc1/MatMul:product:0-base_cifar/fc1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
base_cifar/fc1/BiasAdd�
base_cifar/fc1/ReluRelubase_cifar/fc1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
base_cifar/fc1/Relu�
&base_cifar/final/MatMul/ReadVariableOpReadVariableOp/base_cifar_final_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02(
&base_cifar/final/MatMul/ReadVariableOp�
base_cifar/final/MatMulMatMul!base_cifar/fc1/Relu:activations:0.base_cifar/final/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
base_cifar/final/MatMul�
'base_cifar/final/BiasAdd/ReadVariableOpReadVariableOp0base_cifar_final_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'base_cifar/final/BiasAdd/ReadVariableOp�
base_cifar/final/BiasAddBiasAdd!base_cifar/final/MatMul:product:0/base_cifar/final/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
base_cifar/final/BiasAdd�
base_cifar/final/SoftmaxSoftmax!base_cifar/final/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
base_cifar/final/Softmax�
IdentityIdentity"base_cifar/final/Softmax:softmax:0'^base_cifar/cnn1/BiasAdd/ReadVariableOp&^base_cifar/cnn1/Conv2D/ReadVariableOp(^base_cifar/cnn12/BiasAdd/ReadVariableOp'^base_cifar/cnn12/Conv2D/ReadVariableOp'^base_cifar/cnn2/BiasAdd/ReadVariableOp&^base_cifar/cnn2/Conv2D/ReadVariableOp(^base_cifar/cnn22/BiasAdd/ReadVariableOp'^base_cifar/cnn22/Conv2D/ReadVariableOp'^base_cifar/cnn3/BiasAdd/ReadVariableOp&^base_cifar/cnn3/Conv2D/ReadVariableOp(^base_cifar/cnn32/BiasAdd/ReadVariableOp'^base_cifar/cnn32/Conv2D/ReadVariableOp&^base_cifar/fc1/BiasAdd/ReadVariableOp%^base_cifar/fc1/MatMul/ReadVariableOp(^base_cifar/final/BiasAdd/ReadVariableOp'^base_cifar/final/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������  ::::::::::::::::2P
&base_cifar/cnn1/BiasAdd/ReadVariableOp&base_cifar/cnn1/BiasAdd/ReadVariableOp2N
%base_cifar/cnn1/Conv2D/ReadVariableOp%base_cifar/cnn1/Conv2D/ReadVariableOp2R
'base_cifar/cnn12/BiasAdd/ReadVariableOp'base_cifar/cnn12/BiasAdd/ReadVariableOp2P
&base_cifar/cnn12/Conv2D/ReadVariableOp&base_cifar/cnn12/Conv2D/ReadVariableOp2P
&base_cifar/cnn2/BiasAdd/ReadVariableOp&base_cifar/cnn2/BiasAdd/ReadVariableOp2N
%base_cifar/cnn2/Conv2D/ReadVariableOp%base_cifar/cnn2/Conv2D/ReadVariableOp2R
'base_cifar/cnn22/BiasAdd/ReadVariableOp'base_cifar/cnn22/BiasAdd/ReadVariableOp2P
&base_cifar/cnn22/Conv2D/ReadVariableOp&base_cifar/cnn22/Conv2D/ReadVariableOp2P
&base_cifar/cnn3/BiasAdd/ReadVariableOp&base_cifar/cnn3/BiasAdd/ReadVariableOp2N
%base_cifar/cnn3/Conv2D/ReadVariableOp%base_cifar/cnn3/Conv2D/ReadVariableOp2R
'base_cifar/cnn32/BiasAdd/ReadVariableOp'base_cifar/cnn32/BiasAdd/ReadVariableOp2P
&base_cifar/cnn32/Conv2D/ReadVariableOp&base_cifar/cnn32/Conv2D/ReadVariableOp2N
%base_cifar/fc1/BiasAdd/ReadVariableOp%base_cifar/fc1/BiasAdd/ReadVariableOp2L
$base_cifar/fc1/MatMul/ReadVariableOp$base_cifar/fc1/MatMul/ReadVariableOp2R
'base_cifar/final/BiasAdd/ReadVariableOp'base_cifar/final/BiasAdd/ReadVariableOp2P
&base_cifar/final/MatMul/ReadVariableOp&base_cifar/final/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_2
�
c
D__inference_dropout_3_layer_call_and_return_conditional_losses_12256

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:��������� 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� 2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:��������� 2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:��������� 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
y
$__inference_cnn1_layer_call_fn_13039

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_cnn1_layer_call_and_return_conditional_losses_122002
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������  ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�

�
*__inference_base_cifar_layer_call_fn_12653
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_base_cifar_layer_call_and_return_conditional_losses_126182
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������  ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_2
�A
�
E__inference_base_cifar_layer_call_and_return_conditional_losses_12618

inputs

cnn1_12570

cnn1_12572
cnn12_12575
cnn12_12577

cnn2_12582

cnn2_12584
cnn22_12587
cnn22_12589

cnn3_12594

cnn3_12596
cnn32_12599
cnn32_12601
	fc1_12607
	fc1_12609
final_12612
final_12614
identity��cnn1/StatefulPartitionedCall�cnn12/StatefulPartitionedCall�cnn2/StatefulPartitionedCall�cnn22/StatefulPartitionedCall�cnn3/StatefulPartitionedCall�cnn32/StatefulPartitionedCall�!dropout_3/StatefulPartitionedCall�!dropout_4/StatefulPartitionedCall�!dropout_5/StatefulPartitionedCall�fc1/StatefulPartitionedCall�final/StatefulPartitionedCall�
cnn1/StatefulPartitionedCallStatefulPartitionedCallinputs
cnn1_12570
cnn1_12572*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_cnn1_layer_call_and_return_conditional_losses_122002
cnn1/StatefulPartitionedCall�
cnn12/StatefulPartitionedCallStatefulPartitionedCall%cnn1/StatefulPartitionedCall:output:0cnn12_12575cnn12_12577*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cnn12_layer_call_and_return_conditional_losses_122272
cnn12/StatefulPartitionedCall�
max_pool1/PartitionedCallPartitionedCall&cnn12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_max_pool1_layer_call_and_return_conditional_losses_121552
max_pool1/PartitionedCall�
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall"max_pool1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_122562#
!dropout_3/StatefulPartitionedCall�
cnn2/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0
cnn2_12582
cnn2_12584*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_cnn2_layer_call_and_return_conditional_losses_122852
cnn2/StatefulPartitionedCall�
cnn22/StatefulPartitionedCallStatefulPartitionedCall%cnn2/StatefulPartitionedCall:output:0cnn22_12587cnn22_12589*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cnn22_layer_call_and_return_conditional_losses_123122
cnn22/StatefulPartitionedCall�
max_pool2/PartitionedCallPartitionedCall&cnn22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_max_pool2_layer_call_and_return_conditional_losses_121672
max_pool2/PartitionedCall�
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall"max_pool2/PartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_123412#
!dropout_4/StatefulPartitionedCall�
cnn3/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0
cnn3_12594
cnn3_12596*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_cnn3_layer_call_and_return_conditional_losses_123702
cnn3/StatefulPartitionedCall�
cnn32/StatefulPartitionedCallStatefulPartitionedCall%cnn3/StatefulPartitionedCall:output:0cnn32_12599cnn32_12601*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cnn32_layer_call_and_return_conditional_losses_123972
cnn32/StatefulPartitionedCall�
max_pool3/PartitionedCallPartitionedCall&cnn32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_max_pool3_layer_call_and_return_conditional_losses_121792
max_pool3/PartitionedCall�
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall"max_pool3/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_124262#
!dropout_5/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall*dropout_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_124502
flatten_1/PartitionedCall�
fc1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0	fc1_12607	fc1_12609*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_fc1_layer_call_and_return_conditional_losses_124692
fc1/StatefulPartitionedCall�
final/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0final_12612final_12614*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_final_layer_call_and_return_conditional_losses_124962
final/StatefulPartitionedCall�
IdentityIdentity&final/StatefulPartitionedCall:output:0^cnn1/StatefulPartitionedCall^cnn12/StatefulPartitionedCall^cnn2/StatefulPartitionedCall^cnn22/StatefulPartitionedCall^cnn3/StatefulPartitionedCall^cnn32/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall^fc1/StatefulPartitionedCall^final/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������  ::::::::::::::::2<
cnn1/StatefulPartitionedCallcnn1/StatefulPartitionedCall2>
cnn12/StatefulPartitionedCallcnn12/StatefulPartitionedCall2<
cnn2/StatefulPartitionedCallcnn2/StatefulPartitionedCall2>
cnn22/StatefulPartitionedCallcnn22/StatefulPartitionedCall2<
cnn3/StatefulPartitionedCallcnn3/StatefulPartitionedCall2>
cnn32/StatefulPartitionedCallcnn32/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2>
final/StatefulPartitionedCallfinal/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
E
)__inference_dropout_3_layer_call_fn_13086

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_122612
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
b
)__inference_dropout_5_layer_call_fn_13215

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_124262
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�R
�	
E__inference_base_cifar_layer_call_and_return_conditional_losses_12945

inputs'
#cnn1_conv2d_readvariableop_resource(
$cnn1_biasadd_readvariableop_resource(
$cnn12_conv2d_readvariableop_resource)
%cnn12_biasadd_readvariableop_resource'
#cnn2_conv2d_readvariableop_resource(
$cnn2_biasadd_readvariableop_resource(
$cnn22_conv2d_readvariableop_resource)
%cnn22_biasadd_readvariableop_resource'
#cnn3_conv2d_readvariableop_resource(
$cnn3_biasadd_readvariableop_resource(
$cnn32_conv2d_readvariableop_resource)
%cnn32_biasadd_readvariableop_resource&
"fc1_matmul_readvariableop_resource'
#fc1_biasadd_readvariableop_resource(
$final_matmul_readvariableop_resource)
%final_biasadd_readvariableop_resource
identity��cnn1/BiasAdd/ReadVariableOp�cnn1/Conv2D/ReadVariableOp�cnn12/BiasAdd/ReadVariableOp�cnn12/Conv2D/ReadVariableOp�cnn2/BiasAdd/ReadVariableOp�cnn2/Conv2D/ReadVariableOp�cnn22/BiasAdd/ReadVariableOp�cnn22/Conv2D/ReadVariableOp�cnn3/BiasAdd/ReadVariableOp�cnn3/Conv2D/ReadVariableOp�cnn32/BiasAdd/ReadVariableOp�cnn32/Conv2D/ReadVariableOp�fc1/BiasAdd/ReadVariableOp�fc1/MatMul/ReadVariableOp�final/BiasAdd/ReadVariableOp�final/MatMul/ReadVariableOp�
cnn1/Conv2D/ReadVariableOpReadVariableOp#cnn1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
cnn1/Conv2D/ReadVariableOp�
cnn1/Conv2DConv2Dinputs"cnn1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
cnn1/Conv2D�
cnn1/BiasAdd/ReadVariableOpReadVariableOp$cnn1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
cnn1/BiasAdd/ReadVariableOp�
cnn1/BiasAddBiasAddcnn1/Conv2D:output:0#cnn1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
cnn1/BiasAddo
	cnn1/ReluRelucnn1/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
	cnn1/Relu�
cnn12/Conv2D/ReadVariableOpReadVariableOp$cnn12_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
cnn12/Conv2D/ReadVariableOp�
cnn12/Conv2DConv2Dcnn1/Relu:activations:0#cnn12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
cnn12/Conv2D�
cnn12/BiasAdd/ReadVariableOpReadVariableOp%cnn12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
cnn12/BiasAdd/ReadVariableOp�
cnn12/BiasAddBiasAddcnn12/Conv2D:output:0$cnn12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
cnn12/BiasAddr

cnn12/ReluRelucnn12/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2

cnn12/Relu�
max_pool1/MaxPoolMaxPoolcnn12/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pool1/MaxPool�
dropout_3/IdentityIdentitymax_pool1/MaxPool:output:0*
T0*/
_output_shapes
:��������� 2
dropout_3/Identity�
cnn2/Conv2D/ReadVariableOpReadVariableOp#cnn2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
cnn2/Conv2D/ReadVariableOp�
cnn2/Conv2DConv2Ddropout_3/Identity:output:0"cnn2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
cnn2/Conv2D�
cnn2/BiasAdd/ReadVariableOpReadVariableOp$cnn2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
cnn2/BiasAdd/ReadVariableOp�
cnn2/BiasAddBiasAddcnn2/Conv2D:output:0#cnn2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
cnn2/BiasAddo
	cnn2/ReluRelucnn2/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
	cnn2/Relu�
cnn22/Conv2D/ReadVariableOpReadVariableOp$cnn22_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
cnn22/Conv2D/ReadVariableOp�
cnn22/Conv2DConv2Dcnn2/Relu:activations:0#cnn22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
cnn22/Conv2D�
cnn22/BiasAdd/ReadVariableOpReadVariableOp%cnn22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
cnn22/BiasAdd/ReadVariableOp�
cnn22/BiasAddBiasAddcnn22/Conv2D:output:0$cnn22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
cnn22/BiasAddr

cnn22/ReluRelucnn22/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2

cnn22/Relu�
max_pool2/MaxPoolMaxPoolcnn22/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pool2/MaxPool�
dropout_4/IdentityIdentitymax_pool2/MaxPool:output:0*
T0*/
_output_shapes
:���������@2
dropout_4/Identity�
cnn3/Conv2D/ReadVariableOpReadVariableOp#cnn3_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
cnn3/Conv2D/ReadVariableOp�
cnn3/Conv2DConv2Ddropout_4/Identity:output:0"cnn3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
cnn3/Conv2D�
cnn3/BiasAdd/ReadVariableOpReadVariableOp$cnn3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
cnn3/BiasAdd/ReadVariableOp�
cnn3/BiasAddBiasAddcnn3/Conv2D:output:0#cnn3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
cnn3/BiasAddp
	cnn3/ReluRelucnn3/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
	cnn3/Relu�
cnn32/Conv2D/ReadVariableOpReadVariableOp$cnn32_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
cnn32/Conv2D/ReadVariableOp�
cnn32/Conv2DConv2Dcnn3/Relu:activations:0#cnn32/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
cnn32/Conv2D�
cnn32/BiasAdd/ReadVariableOpReadVariableOp%cnn32_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
cnn32/BiasAdd/ReadVariableOp�
cnn32/BiasAddBiasAddcnn32/Conv2D:output:0$cnn32/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
cnn32/BiasAdds

cnn32/ReluRelucnn32/BiasAdd:output:0*
T0*0
_output_shapes
:����������2

cnn32/Relu�
max_pool3/MaxPoolMaxPoolcnn32/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pool3/MaxPool�
dropout_5/IdentityIdentitymax_pool3/MaxPool:output:0*
T0*0
_output_shapes
:����������2
dropout_5/Identitys
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_1/Const�
flatten_1/ReshapeReshapedropout_5/Identity:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_1/Reshape�
fc1/MatMul/ReadVariableOpReadVariableOp"fc1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
fc1/MatMul/ReadVariableOp�

fc1/MatMulMatMulflatten_1/Reshape:output:0!fc1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

fc1/MatMul�
fc1/BiasAdd/ReadVariableOpReadVariableOp#fc1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
fc1/BiasAdd/ReadVariableOp�
fc1/BiasAddBiasAddfc1/MatMul:product:0"fc1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
fc1/BiasAdde
fc1/ReluRelufc1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2

fc1/Relu�
final/MatMul/ReadVariableOpReadVariableOp$final_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02
final/MatMul/ReadVariableOp�
final/MatMulMatMulfc1/Relu:activations:0#final/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
final/MatMul�
final/BiasAdd/ReadVariableOpReadVariableOp%final_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
final/BiasAdd/ReadVariableOp�
final/BiasAddBiasAddfinal/MatMul:product:0$final/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
final/BiasAdds
final/SoftmaxSoftmaxfinal/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
final/Softmax�
IdentityIdentityfinal/Softmax:softmax:0^cnn1/BiasAdd/ReadVariableOp^cnn1/Conv2D/ReadVariableOp^cnn12/BiasAdd/ReadVariableOp^cnn12/Conv2D/ReadVariableOp^cnn2/BiasAdd/ReadVariableOp^cnn2/Conv2D/ReadVariableOp^cnn22/BiasAdd/ReadVariableOp^cnn22/Conv2D/ReadVariableOp^cnn3/BiasAdd/ReadVariableOp^cnn3/Conv2D/ReadVariableOp^cnn32/BiasAdd/ReadVariableOp^cnn32/Conv2D/ReadVariableOp^fc1/BiasAdd/ReadVariableOp^fc1/MatMul/ReadVariableOp^final/BiasAdd/ReadVariableOp^final/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������  ::::::::::::::::2:
cnn1/BiasAdd/ReadVariableOpcnn1/BiasAdd/ReadVariableOp28
cnn1/Conv2D/ReadVariableOpcnn1/Conv2D/ReadVariableOp2<
cnn12/BiasAdd/ReadVariableOpcnn12/BiasAdd/ReadVariableOp2:
cnn12/Conv2D/ReadVariableOpcnn12/Conv2D/ReadVariableOp2:
cnn2/BiasAdd/ReadVariableOpcnn2/BiasAdd/ReadVariableOp28
cnn2/Conv2D/ReadVariableOpcnn2/Conv2D/ReadVariableOp2<
cnn22/BiasAdd/ReadVariableOpcnn22/BiasAdd/ReadVariableOp2:
cnn22/Conv2D/ReadVariableOpcnn22/Conv2D/ReadVariableOp2:
cnn3/BiasAdd/ReadVariableOpcnn3/BiasAdd/ReadVariableOp28
cnn3/Conv2D/ReadVariableOpcnn3/Conv2D/ReadVariableOp2<
cnn32/BiasAdd/ReadVariableOpcnn32/BiasAdd/ReadVariableOp2:
cnn32/Conv2D/ReadVariableOpcnn32/Conv2D/ReadVariableOp28
fc1/BiasAdd/ReadVariableOpfc1/BiasAdd/ReadVariableOp26
fc1/MatMul/ReadVariableOpfc1/MatMul/ReadVariableOp2<
final/BiasAdd/ReadVariableOpfinal/BiasAdd/ReadVariableOp2:
final/MatMul/ReadVariableOpfinal/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
`
D__inference_max_pool2_layer_call_and_return_conditional_losses_12167

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_12261

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:��������� 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:��������� 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
@__inference_cnn32_layer_call_and_return_conditional_losses_13184

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
@__inference_cnn32_layer_call_and_return_conditional_losses_12397

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
@__inference_cnn22_layer_call_and_return_conditional_losses_13117

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
@__inference_cnn12_layer_call_and_return_conditional_losses_13050

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������   2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_28
serving_default_input_2:0���������  9
final0
StatefulPartitionedCall:0���������
tensorflow/serving/predict:��
��
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer_with_weights-7
layer-15
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
�_default_save_signature
+�&call_and_return_all_conditional_losses
�__call__"�~
_tf_keras_network�~{"class_name": "Functional", "name": "base_cifar", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "base_cifar", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "cnn1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "cnn1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "cnn12", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "cnn12", "inbound_nodes": [[["cnn1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pool1", "inbound_nodes": [[["cnn12", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout_3", "inbound_nodes": [[["max_pool1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "cnn2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "cnn2", "inbound_nodes": [[["dropout_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "cnn22", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "cnn22", "inbound_nodes": [[["cnn2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pool2", "inbound_nodes": [[["cnn22", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout_4", "inbound_nodes": [[["max_pool2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "cnn3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "cnn3", "inbound_nodes": [[["dropout_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "cnn32", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "cnn32", "inbound_nodes": [[["cnn3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pool3", "inbound_nodes": [[["cnn32", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout_5", "inbound_nodes": [[["max_pool3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["dropout_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc1", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "final", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "final", "inbound_nodes": [[["fc1", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["final", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "base_cifar", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "cnn1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "cnn1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "cnn12", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "cnn12", "inbound_nodes": [[["cnn1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pool1", "inbound_nodes": [[["cnn12", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout_3", "inbound_nodes": [[["max_pool1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "cnn2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "cnn2", "inbound_nodes": [[["dropout_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "cnn22", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "cnn22", "inbound_nodes": [[["cnn2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pool2", "inbound_nodes": [[["cnn22", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout_4", "inbound_nodes": [[["max_pool2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "cnn3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "cnn3", "inbound_nodes": [[["dropout_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "cnn32", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "cnn32", "inbound_nodes": [[["cnn3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pool3", "inbound_nodes": [[["cnn32", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout_5", "inbound_nodes": [[["max_pool3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["dropout_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc1", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "final", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "final", "inbound_nodes": [[["fc1", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["final", 0, 0]]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
�	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "cnn1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "cnn1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}}
�	

kernel
bias
	variables
 regularization_losses
!trainable_variables
"	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "cnn12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "cnn12", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
�
#	variables
$regularization_losses
%trainable_variables
&	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pool1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
'	variables
(regularization_losses
)trainable_variables
*	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
�	

+kernel
,bias
-	variables
.regularization_losses
/trainable_variables
0	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "cnn2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "cnn2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 32]}}
�	

1kernel
2bias
3	variables
4regularization_losses
5trainable_variables
6	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "cnn22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "cnn22", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
�
7	variables
8regularization_losses
9trainable_variables
:	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pool2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
;	variables
<regularization_losses
=trainable_variables
>	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
�	

?kernel
@bias
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "cnn3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "cnn3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 64]}}
�	

Ekernel
Fbias
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "cnn32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "cnn32", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
�
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pool3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
O	variables
Pregularization_losses
Qtrainable_variables
R	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
�
S	variables
Tregularization_losses
Utrainable_variables
V	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

Wkernel
Xbias
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "fc1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2048}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2048]}}
�

]kernel
^bias
_	variables
`regularization_losses
atrainable_variables
b	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "final", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "final", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
�
citer

dbeta_1

ebeta_2
	fdecay
glearning_ratem�m�m�m�+m�,m�1m�2m�?m�@m�Em�Fm�Wm�Xm�]m�^m�v�v�v�v�+v�,v�1v�2v�?v�@v�Ev�Fv�Wv�Xv�]v�^v�"
	optimizer
�
0
1
2
3
+4
,5
16
27
?8
@9
E10
F11
W12
X13
]14
^15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0
1
2
3
+4
,5
16
27
?8
@9
E10
F11
W12
X13
]14
^15"
trackable_list_wrapper
�

hlayers
	variables
ilayer_metrics
jlayer_regularization_losses
knon_trainable_variables
lmetrics
regularization_losses
trainable_variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
%:# 2cnn1/kernel
: 2	cnn1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�

mlayers
	variables
nlayer_metrics
olayer_regularization_losses
pnon_trainable_variables
qmetrics
regularization_losses
trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
&:$  2cnn12/kernel
: 2
cnn12/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�

rlayers
	variables
slayer_metrics
tlayer_regularization_losses
unon_trainable_variables
vmetrics
 regularization_losses
!trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

wlayers
#	variables
xlayer_metrics
ylayer_regularization_losses
znon_trainable_variables
{metrics
$regularization_losses
%trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

|layers
'	variables
}layer_metrics
~layer_regularization_losses
non_trainable_variables
�metrics
(regularization_losses
)trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
%:# @2cnn2/kernel
:@2	cnn2/bias
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
�
�layers
-	variables
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�metrics
.regularization_losses
/trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
&:$@@2cnn22/kernel
:@2
cnn22/bias
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
�
�layers
3	variables
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�metrics
4regularization_losses
5trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layers
7	variables
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�metrics
8regularization_losses
9trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layers
;	variables
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�metrics
<regularization_losses
=trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
&:$@�2cnn3/kernel
:�2	cnn3/bias
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
�
�layers
A	variables
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�metrics
Bregularization_losses
Ctrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(:&��2cnn32/kernel
:�2
cnn32/bias
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
�
�layers
G	variables
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�metrics
Hregularization_losses
Itrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layers
K	variables
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�metrics
Lregularization_losses
Mtrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layers
O	variables
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�metrics
Pregularization_losses
Qtrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layers
S	variables
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�metrics
Tregularization_losses
Utrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:
��2
fc1/kernel
:�2fc1/bias
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
�
�layers
Y	variables
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�metrics
Zregularization_losses
[trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	�
2final/kernel
:
2
final/bias
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
�
�layers
_	variables
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�metrics
`regularization_losses
atrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "sparse_categorical_accuracy", "dtype": "float32", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
*:( 2Adam/cnn1/kernel/m
: 2Adam/cnn1/bias/m
+:)  2Adam/cnn12/kernel/m
: 2Adam/cnn12/bias/m
*:( @2Adam/cnn2/kernel/m
:@2Adam/cnn2/bias/m
+:)@@2Adam/cnn22/kernel/m
:@2Adam/cnn22/bias/m
+:)@�2Adam/cnn3/kernel/m
:�2Adam/cnn3/bias/m
-:+��2Adam/cnn32/kernel/m
:�2Adam/cnn32/bias/m
#:!
��2Adam/fc1/kernel/m
:�2Adam/fc1/bias/m
$:"	�
2Adam/final/kernel/m
:
2Adam/final/bias/m
*:( 2Adam/cnn1/kernel/v
: 2Adam/cnn1/bias/v
+:)  2Adam/cnn12/kernel/v
: 2Adam/cnn12/bias/v
*:( @2Adam/cnn2/kernel/v
:@2Adam/cnn2/bias/v
+:)@@2Adam/cnn22/kernel/v
:@2Adam/cnn22/bias/v
+:)@�2Adam/cnn3/kernel/v
:�2Adam/cnn3/bias/v
-:+��2Adam/cnn32/kernel/v
:�2Adam/cnn32/bias/v
#:!
��2Adam/fc1/kernel/v
:�2Adam/fc1/bias/v
$:"	�
2Adam/final/kernel/v
:
2Adam/final/bias/v
�2�
 __inference__wrapped_model_12149�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
input_2���������  
�2�
E__inference_base_cifar_layer_call_and_return_conditional_losses_12945
E__inference_base_cifar_layer_call_and_return_conditional_losses_12564
E__inference_base_cifar_layer_call_and_return_conditional_losses_12877
E__inference_base_cifar_layer_call_and_return_conditional_losses_12513�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_base_cifar_layer_call_fn_13019
*__inference_base_cifar_layer_call_fn_12653
*__inference_base_cifar_layer_call_fn_12982
*__inference_base_cifar_layer_call_fn_12741�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
?__inference_cnn1_layer_call_and_return_conditional_losses_13030�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_cnn1_layer_call_fn_13039�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_cnn12_layer_call_and_return_conditional_losses_13050�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_cnn12_layer_call_fn_13059�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_max_pool1_layer_call_and_return_conditional_losses_12155�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
)__inference_max_pool1_layer_call_fn_12161�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
D__inference_dropout_3_layer_call_and_return_conditional_losses_13071
D__inference_dropout_3_layer_call_and_return_conditional_losses_13076�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_dropout_3_layer_call_fn_13086
)__inference_dropout_3_layer_call_fn_13081�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
?__inference_cnn2_layer_call_and_return_conditional_losses_13097�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_cnn2_layer_call_fn_13106�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_cnn22_layer_call_and_return_conditional_losses_13117�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_cnn22_layer_call_fn_13126�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_max_pool2_layer_call_and_return_conditional_losses_12167�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
)__inference_max_pool2_layer_call_fn_12173�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
D__inference_dropout_4_layer_call_and_return_conditional_losses_13143
D__inference_dropout_4_layer_call_and_return_conditional_losses_13138�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_dropout_4_layer_call_fn_13153
)__inference_dropout_4_layer_call_fn_13148�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
?__inference_cnn3_layer_call_and_return_conditional_losses_13164�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_cnn3_layer_call_fn_13173�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_cnn32_layer_call_and_return_conditional_losses_13184�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_cnn32_layer_call_fn_13193�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_max_pool3_layer_call_and_return_conditional_losses_12179�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
)__inference_max_pool3_layer_call_fn_12185�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
D__inference_dropout_5_layer_call_and_return_conditional_losses_13205
D__inference_dropout_5_layer_call_and_return_conditional_losses_13210�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_dropout_5_layer_call_fn_13220
)__inference_dropout_5_layer_call_fn_13215�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_flatten_1_layer_call_and_return_conditional_losses_13226�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_flatten_1_layer_call_fn_13231�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
>__inference_fc1_layer_call_and_return_conditional_losses_13242�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
#__inference_fc1_layer_call_fn_13251�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_final_layer_call_and_return_conditional_losses_13262�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_final_layer_call_fn_13271�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_signature_wrapper_12788input_2"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
 __inference__wrapped_model_12149{+,12?@EFWX]^8�5
.�+
)�&
input_2���������  
� "-�*
(
final�
final���������
�
E__inference_base_cifar_layer_call_and_return_conditional_losses_12513{+,12?@EFWX]^@�=
6�3
)�&
input_2���������  
p

 
� "%�"
�
0���������

� �
E__inference_base_cifar_layer_call_and_return_conditional_losses_12564{+,12?@EFWX]^@�=
6�3
)�&
input_2���������  
p 

 
� "%�"
�
0���������

� �
E__inference_base_cifar_layer_call_and_return_conditional_losses_12877z+,12?@EFWX]^?�<
5�2
(�%
inputs���������  
p

 
� "%�"
�
0���������

� �
E__inference_base_cifar_layer_call_and_return_conditional_losses_12945z+,12?@EFWX]^?�<
5�2
(�%
inputs���������  
p 

 
� "%�"
�
0���������

� �
*__inference_base_cifar_layer_call_fn_12653n+,12?@EFWX]^@�=
6�3
)�&
input_2���������  
p

 
� "����������
�
*__inference_base_cifar_layer_call_fn_12741n+,12?@EFWX]^@�=
6�3
)�&
input_2���������  
p 

 
� "����������
�
*__inference_base_cifar_layer_call_fn_12982m+,12?@EFWX]^?�<
5�2
(�%
inputs���������  
p

 
� "����������
�
*__inference_base_cifar_layer_call_fn_13019m+,12?@EFWX]^?�<
5�2
(�%
inputs���������  
p 

 
� "����������
�
@__inference_cnn12_layer_call_and_return_conditional_losses_13050l7�4
-�*
(�%
inputs���������   
� "-�*
#� 
0���������   
� �
%__inference_cnn12_layer_call_fn_13059_7�4
-�*
(�%
inputs���������   
� " ����������   �
?__inference_cnn1_layer_call_and_return_conditional_losses_13030l7�4
-�*
(�%
inputs���������  
� "-�*
#� 
0���������   
� �
$__inference_cnn1_layer_call_fn_13039_7�4
-�*
(�%
inputs���������  
� " ����������   �
@__inference_cnn22_layer_call_and_return_conditional_losses_13117l127�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
%__inference_cnn22_layer_call_fn_13126_127�4
-�*
(�%
inputs���������@
� " ����������@�
?__inference_cnn2_layer_call_and_return_conditional_losses_13097l+,7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������@
� �
$__inference_cnn2_layer_call_fn_13106_+,7�4
-�*
(�%
inputs��������� 
� " ����������@�
@__inference_cnn32_layer_call_and_return_conditional_losses_13184nEF8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
%__inference_cnn32_layer_call_fn_13193aEF8�5
.�+
)�&
inputs����������
� "!������������
?__inference_cnn3_layer_call_and_return_conditional_losses_13164m?@7�4
-�*
(�%
inputs���������@
� ".�+
$�!
0����������
� �
$__inference_cnn3_layer_call_fn_13173`?@7�4
-�*
(�%
inputs���������@
� "!������������
D__inference_dropout_3_layer_call_and_return_conditional_losses_13071l;�8
1�.
(�%
inputs��������� 
p
� "-�*
#� 
0��������� 
� �
D__inference_dropout_3_layer_call_and_return_conditional_losses_13076l;�8
1�.
(�%
inputs��������� 
p 
� "-�*
#� 
0��������� 
� �
)__inference_dropout_3_layer_call_fn_13081_;�8
1�.
(�%
inputs��������� 
p
� " ���������� �
)__inference_dropout_3_layer_call_fn_13086_;�8
1�.
(�%
inputs��������� 
p 
� " ���������� �
D__inference_dropout_4_layer_call_and_return_conditional_losses_13138l;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
D__inference_dropout_4_layer_call_and_return_conditional_losses_13143l;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
)__inference_dropout_4_layer_call_fn_13148_;�8
1�.
(�%
inputs���������@
p
� " ����������@�
)__inference_dropout_4_layer_call_fn_13153_;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
D__inference_dropout_5_layer_call_and_return_conditional_losses_13205n<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
D__inference_dropout_5_layer_call_and_return_conditional_losses_13210n<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
)__inference_dropout_5_layer_call_fn_13215a<�9
2�/
)�&
inputs����������
p
� "!������������
)__inference_dropout_5_layer_call_fn_13220a<�9
2�/
)�&
inputs����������
p 
� "!������������
>__inference_fc1_layer_call_and_return_conditional_losses_13242^WX0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� x
#__inference_fc1_layer_call_fn_13251QWX0�-
&�#
!�
inputs����������
� "������������
@__inference_final_layer_call_and_return_conditional_losses_13262]]^0�-
&�#
!�
inputs����������
� "%�"
�
0���������

� y
%__inference_final_layer_call_fn_13271P]^0�-
&�#
!�
inputs����������
� "����������
�
D__inference_flatten_1_layer_call_and_return_conditional_losses_13226b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������
� �
)__inference_flatten_1_layer_call_fn_13231U8�5
.�+
)�&
inputs����������
� "������������
D__inference_max_pool1_layer_call_and_return_conditional_losses_12155�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
)__inference_max_pool1_layer_call_fn_12161�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
D__inference_max_pool2_layer_call_and_return_conditional_losses_12167�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
)__inference_max_pool2_layer_call_fn_12173�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
D__inference_max_pool3_layer_call_and_return_conditional_losses_12179�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
)__inference_max_pool3_layer_call_fn_12185�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
#__inference_signature_wrapper_12788�+,12?@EFWX]^C�@
� 
9�6
4
input_2)�&
input_2���������  "-�*
(
final�
final���������
