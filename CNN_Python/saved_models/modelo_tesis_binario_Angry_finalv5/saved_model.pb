И
Їі
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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

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

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

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
0
Sigmoid
x"T
y"T"
Ttype:

2
С
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
executor_typestring Ј
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58ли

t
dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias/v
m
"dense_5/bias/v/Read/ReadVariableOpReadVariableOpdense_5/bias/v*
_output_shapes
:*
dtype0
|
dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_namedense_5/kernel/v
u
$dense_5/kernel/v/Read/ReadVariableOpReadVariableOpdense_5/kernel/v*
_output_shapes

: *
dtype0
t
dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_4/bias/v
m
"dense_4/bias/v/Read/ReadVariableOpReadVariableOpdense_4/bias/v*
_output_shapes
: *
dtype0
|
dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_4/kernel/v
u
$dense_4/kernel/v/Read/ReadVariableOpReadVariableOpdense_4/kernel/v*
_output_shapes

:  *
dtype0
t
dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_3/bias/v
m
"dense_3/bias/v/Read/ReadVariableOpReadVariableOpdense_3/bias/v*
_output_shapes
: *
dtype0
}
dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р *!
shared_namedense_3/kernel/v
v
$dense_3/kernel/v/Read/ReadVariableOpReadVariableOpdense_3/kernel/v*
_output_shapes
:	р *
dtype0
v
conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_5/bias/v
o
#conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpconv2d_5/bias/v*
_output_shapes
:*
dtype0

conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_5/kernel/v

%conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_5/kernel/v*&
_output_shapes
:*
dtype0
v
conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/bias/v
o
#conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpconv2d_4/bias/v*
_output_shapes
:*
dtype0

conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_4/kernel/v

%conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_4/kernel/v*&
_output_shapes
: *
dtype0
v
conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_3/bias/v
o
#conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpconv2d_3/bias/v*
_output_shapes
: *
dtype0

conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_3/kernel/v

%conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_3/kernel/v*&
_output_shapes
: *
dtype0
t
dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias/m
m
"dense_5/bias/m/Read/ReadVariableOpReadVariableOpdense_5/bias/m*
_output_shapes
:*
dtype0
|
dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_namedense_5/kernel/m
u
$dense_5/kernel/m/Read/ReadVariableOpReadVariableOpdense_5/kernel/m*
_output_shapes

: *
dtype0
t
dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_4/bias/m
m
"dense_4/bias/m/Read/ReadVariableOpReadVariableOpdense_4/bias/m*
_output_shapes
: *
dtype0
|
dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_4/kernel/m
u
$dense_4/kernel/m/Read/ReadVariableOpReadVariableOpdense_4/kernel/m*
_output_shapes

:  *
dtype0
t
dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_3/bias/m
m
"dense_3/bias/m/Read/ReadVariableOpReadVariableOpdense_3/bias/m*
_output_shapes
: *
dtype0
}
dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р *!
shared_namedense_3/kernel/m
v
$dense_3/kernel/m/Read/ReadVariableOpReadVariableOpdense_3/kernel/m*
_output_shapes
:	р *
dtype0
v
conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_5/bias/m
o
#conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpconv2d_5/bias/m*
_output_shapes
:*
dtype0

conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_5/kernel/m

%conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_5/kernel/m*&
_output_shapes
:*
dtype0
v
conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/bias/m
o
#conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpconv2d_4/bias/m*
_output_shapes
:*
dtype0

conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_4/kernel/m

%conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_4/kernel/m*&
_output_shapes
: *
dtype0
v
conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_3/bias/m
o
#conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpconv2d_3/bias/m*
_output_shapes
: *
dtype0

conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_3/kernel/m

%conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_3/kernel/m*&
_output_shapes
: *
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
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
\
iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameiter
U
iter/Read/ReadVariableOpReadVariableOpiter*
_output_shapes
: *
dtype0	
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

: *
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
: *
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:  *
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
: *
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р *
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	р *
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:*
dtype0

conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:*
dtype0

conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
: *
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
: *
dtype0

conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
: *
dtype0

serving_default_conv2d_3_inputPlaceholder*/
_output_shapes
:џџџџџџџџџ?*
dtype0*$
shape:џџџџџџџџџ?

StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_3_inputconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_43916

NoOpNoOp
Жe
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ёd
valueчdBфd Bнd
ј
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
Ш
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*

	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses* 
Ш
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias
 ,_jit_compiled_convolution_op*
Ш
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias
 5_jit_compiled_convolution_op*

6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses* 

<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses* 
І
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

Hkernel
Ibias*
Ѕ
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
P_random_generator* 
І
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

Wkernel
Xbias*
Ѕ
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
__random_generator* 
І
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses

fkernel
gbias*
Z
0
1
*2
+3
34
45
H6
I7
W8
X9
f10
g11*
Z
0
1
*2
+3
34
45
H6
I7
W8
X9
f10
g11*

h0
i1* 
А
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
otrace_0
ptrace_1
qtrace_2
rtrace_3* 
6
strace_0
ttrace_1
utrace_2
vtrace_3* 
* 
Д
witer

xbeta_1

ybeta_2
	zdecay
{learning_ratemлmм*mн+mо3mп4mрHmсImтWmуXmфfmхgmцvчvш*vщ+vъ3vы4vьHvэIvюWvяXv№fvёgvђ*

|serving_default* 

0
1*

0
1*
	
h0* 

}non_trainable_variables

~layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0* 

trace_0* 
_Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

*0
+1*

*0
+1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

trace_0* 

trace_0* 
_Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

30
41*

30
41*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

trace_0* 

trace_0* 
_Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

 non_trainable_variables
Ёlayers
Ђmetrics
 Ѓlayer_regularization_losses
Єlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses* 

Ѕtrace_0* 

Іtrace_0* 

H0
I1*

H0
I1*
	
i0* 

Їnon_trainable_variables
Јlayers
Љmetrics
 Њlayer_regularization_losses
Ћlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

Ќtrace_0* 

­trace_0* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Ўnon_trainable_variables
Џlayers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 

Гtrace_0
Дtrace_1* 

Еtrace_0
Жtrace_1* 
* 

W0
X1*

W0
X1*
* 

Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*

Мtrace_0* 

Нtrace_0* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses* 

Уtrace_0
Фtrace_1* 

Хtrace_0
Цtrace_1* 
* 

f0
g1*

f0
g1*
* 

Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses*

Ьtrace_0* 

Эtrace_0* 
^X
VARIABLE_VALUEdense_5/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_5/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

Юtrace_0* 

Яtrace_0* 
* 
R
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
10*

а0
б1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
GA
VARIABLE_VALUEiter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
	
h0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
i0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
в	variables
г	keras_api

дtotal

еcount*
M
ж	variables
з	keras_api

иtotal

йcount
к
_fn_kwargs*

д0
е1*

в	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

и0
й1*

ж	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
}w
VARIABLE_VALUEconv2d_3/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEconv2d_3/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEconv2d_4/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEconv2d_4/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEconv2d_5/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEconv2d_5/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEdense_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEdense_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEdense_4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEdense_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEdense_5/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEdense_5/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEconv2d_3/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEconv2d_3/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEconv2d_4/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEconv2d_4/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEconv2d_5/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEconv2d_5/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEdense_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEdense_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEdense_4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEdense_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEdense_5/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEdense_5/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ј
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpiter/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp%conv2d_3/kernel/m/Read/ReadVariableOp#conv2d_3/bias/m/Read/ReadVariableOp%conv2d_4/kernel/m/Read/ReadVariableOp#conv2d_4/bias/m/Read/ReadVariableOp%conv2d_5/kernel/m/Read/ReadVariableOp#conv2d_5/bias/m/Read/ReadVariableOp$dense_3/kernel/m/Read/ReadVariableOp"dense_3/bias/m/Read/ReadVariableOp$dense_4/kernel/m/Read/ReadVariableOp"dense_4/bias/m/Read/ReadVariableOp$dense_5/kernel/m/Read/ReadVariableOp"dense_5/bias/m/Read/ReadVariableOp%conv2d_3/kernel/v/Read/ReadVariableOp#conv2d_3/bias/v/Read/ReadVariableOp%conv2d_4/kernel/v/Read/ReadVariableOp#conv2d_4/bias/v/Read/ReadVariableOp%conv2d_5/kernel/v/Read/ReadVariableOp#conv2d_5/bias/v/Read/ReadVariableOp$dense_3/kernel/v/Read/ReadVariableOp"dense_3/bias/v/Read/ReadVariableOp$dense_4/kernel/v/Read/ReadVariableOp"dense_4/bias/v/Read/ReadVariableOp$dense_5/kernel/v/Read/ReadVariableOp"dense_5/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_44505
я
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasiterbeta_1beta_2decaylearning_ratetotal_1count_1totalcountconv2d_3/kernel/mconv2d_3/bias/mconv2d_4/kernel/mconv2d_4/bias/mconv2d_5/kernel/mconv2d_5/bias/mdense_3/kernel/mdense_3/bias/mdense_4/kernel/mdense_4/bias/mdense_5/kernel/mdense_5/bias/mconv2d_3/kernel/vconv2d_3/bias/vconv2d_4/kernel/vconv2d_4/bias/vconv2d_5/kernel/vconv2d_5/bias/vdense_3/kernel/vdense_3/bias/vdense_4/kernel/vdense_4/bias/vdense_5/kernel/vdense_5/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_44650	

Р
#__inference_signature_wrapper_43916
conv2d_3_input!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5:	р 
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10:
identityЂStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_43330o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:џџџџџџџџџ?: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ?
(
_user_specified_nameconv2d_3_input
ёY

__inference__traced_save_44505
file_prefix.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop#
savev2_iter_read_readvariableop	%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop0
,savev2_conv2d_3_kernel_m_read_readvariableop.
*savev2_conv2d_3_bias_m_read_readvariableop0
,savev2_conv2d_4_kernel_m_read_readvariableop.
*savev2_conv2d_4_bias_m_read_readvariableop0
,savev2_conv2d_5_kernel_m_read_readvariableop.
*savev2_conv2d_5_bias_m_read_readvariableop/
+savev2_dense_3_kernel_m_read_readvariableop-
)savev2_dense_3_bias_m_read_readvariableop/
+savev2_dense_4_kernel_m_read_readvariableop-
)savev2_dense_4_bias_m_read_readvariableop/
+savev2_dense_5_kernel_m_read_readvariableop-
)savev2_dense_5_bias_m_read_readvariableop0
,savev2_conv2d_3_kernel_v_read_readvariableop.
*savev2_conv2d_3_bias_v_read_readvariableop0
,savev2_conv2d_4_kernel_v_read_readvariableop.
*savev2_conv2d_4_bias_v_read_readvariableop0
,savev2_conv2d_5_kernel_v_read_readvariableop.
*savev2_conv2d_5_bias_v_read_readvariableop/
+savev2_dense_3_kernel_v_read_readvariableop-
)savev2_dense_3_bias_v_read_readvariableop/
+savev2_dense_4_kernel_v_read_readvariableop-
)savev2_dense_4_bias_v_read_readvariableop/
+savev2_dense_5_kernel_v_read_readvariableop-
)savev2_dense_5_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ѓ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*Ь
valueТBП.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЩ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ю
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableopsavev2_iter_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop,savev2_conv2d_3_kernel_m_read_readvariableop*savev2_conv2d_3_bias_m_read_readvariableop,savev2_conv2d_4_kernel_m_read_readvariableop*savev2_conv2d_4_bias_m_read_readvariableop,savev2_conv2d_5_kernel_m_read_readvariableop*savev2_conv2d_5_bias_m_read_readvariableop+savev2_dense_3_kernel_m_read_readvariableop)savev2_dense_3_bias_m_read_readvariableop+savev2_dense_4_kernel_m_read_readvariableop)savev2_dense_4_bias_m_read_readvariableop+savev2_dense_5_kernel_m_read_readvariableop)savev2_dense_5_bias_m_read_readvariableop,savev2_conv2d_3_kernel_v_read_readvariableop*savev2_conv2d_3_bias_v_read_readvariableop,savev2_conv2d_4_kernel_v_read_readvariableop*savev2_conv2d_4_bias_v_read_readvariableop,savev2_conv2d_5_kernel_v_read_readvariableop*savev2_conv2d_5_bias_v_read_readvariableop+savev2_dense_3_kernel_v_read_readvariableop)savev2_dense_3_bias_v_read_readvariableop+savev2_dense_4_kernel_v_read_readvariableop)savev2_dense_4_bias_v_read_readvariableop+savev2_dense_5_kernel_v_read_readvariableop)savev2_dense_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *<
dtypes2
02.	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: : : : ::::	р : :  : : :: : : : : : : : : : : : ::::	р : :  : : :: : : ::::	р : :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	р : 

_output_shapes
: :$	 

_output_shapes

:  : 


_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	р : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$  

_output_shapes

: : !

_output_shapes
::,"(
&
_output_shapes
: : #

_output_shapes
: :,$(
&
_output_shapes
: : %

_output_shapes
::,&(
&
_output_shapes
:: '

_output_shapes
::%(!

_output_shapes
:	р : )

_output_shapes
: :$* 

_output_shapes

:  : +

_output_shapes
: :$, 

_output_shapes

: : -

_output_shapes
::.

_output_shapes
: 
С

'__inference_dense_5_layer_call_fn_44318

inputs
unknown: 
	unknown_0:
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_43489o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ц
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_43424

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџр   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџрY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџр"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ќ
C__inference_conv2d_4_layer_call_and_return_conditional_losses_44170

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Б:
З
G__inference_sequential_1_layer_call_and_return_conditional_losses_43824
conv2d_3_input(
conv2d_3_43780: 
conv2d_3_43782: (
conv2d_4_43786: 
conv2d_4_43788:(
conv2d_5_43791:
conv2d_5_43793: 
dense_3_43798:	р 
dense_3_43800: 
dense_4_43804:  
dense_4_43806: 
dense_5_43810: 
dense_5_43812:
identityЂ conv2d_3/StatefulPartitionedCallЂ1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpЂ conv2d_4/StatefulPartitionedCallЂ conv2d_5/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂ0dense_3/kernel/Regularizer/L2Loss/ReadVariableOpЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCall
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputconv2d_3_43780conv2d_3_43782*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_43376ѓ
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_43339
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_4_43786conv2d_4_43788*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_43394
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_43791conv2d_5_43793*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_43411ѓ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_43351п
flatten_1/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџр* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_43424
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_43798dense_3_43800*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_43441о
dropout_2/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_43452
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_4_43804dense_4_43806*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_43465о
dropout_3/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_43476
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_5_43810dense_5_43812*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_43489
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_3_43780*&
_output_shapes
: *
dtype0
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ы? 
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
0dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_3_43798*
_output_shapes
:	р *
dtype0
!dense_3/kernel/Regularizer/L2LossL2Loss8dense_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ы?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0*dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџќ
NoOpNoOp!^conv2d_3/StatefulPartitionedCall2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall1^dense_3/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:џџџџџџџџџ?: : : : : : : : : : : : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2d
0dense_3/kernel/Regularizer/L2Loss/ReadVariableOp0dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ?
(
_user_specified_nameconv2d_3_input
ьH


G__inference_sequential_1_layer_call_and_return_conditional_losses_44042

inputsA
'conv2d_3_conv2d_readvariableop_resource: 6
(conv2d_3_biasadd_readvariableop_resource: A
'conv2d_4_conv2d_readvariableop_resource: 6
(conv2d_4_biasadd_readvariableop_resource:A
'conv2d_5_conv2d_readvariableop_resource:6
(conv2d_5_biasadd_readvariableop_resource:9
&dense_3_matmul_readvariableop_resource:	р 5
'dense_3_biasadd_readvariableop_resource: 8
&dense_4_matmul_readvariableop_resource:  5
'dense_4_biasadd_readvariableop_resource: 8
&dense_5_matmul_readvariableop_resource: 5
'dense_5_biasadd_readvariableop_resource:
identityЂconv2d_3/BiasAdd/ReadVariableOpЂconv2d_3/Conv2D/ReadVariableOpЂ1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpЂconv2d_4/BiasAdd/ReadVariableOpЂconv2d_4/Conv2D/ReadVariableOpЂconv2d_5/BiasAdd/ReadVariableOpЂconv2d_5/Conv2D/ReadVariableOpЂdense_3/BiasAdd/ReadVariableOpЂdense_3/MatMul/ReadVariableOpЂ0dense_3/kernel/Regularizer/L2Loss/ReadVariableOpЂdense_4/BiasAdd/ReadVariableOpЂdense_4/MatMul/ReadVariableOpЂdense_5/BiasAdd/ReadVariableOpЂdense_5/MatMul/ReadVariableOp
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ћ
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ? *
paddingSAME*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ? j
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ? Ќ
max_pooling2d_2/MaxPoolMaxPoolconv2d_3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides

conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Х
conv2d_4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџj
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Р
conv2d_5/Conv2DConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџj
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџЌ
max_pooling2d_3/MaxPoolMaxPoolconv2d_5/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџр   
flatten_1/ReshapeReshape max_pooling2d_3/MaxPool:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџр
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	р *
dtype0
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ l
dropout_2/IdentityIdentitydense_3/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_4/MatMulMatMuldropout_2/Identity:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ l
dropout_3/IdentityIdentitydense_4/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_5/MatMulMatMuldropout_3/Identity:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџf
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџЁ
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ы? 
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
0dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	р *
dtype0
!dense_3/kernel/Regularizer/L2LossL2Loss8dense_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ы?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0*dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentitydense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЙ
NoOpNoOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp1^dense_3/kernel/Regularizer/L2Loss/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:џџџџџџџџџ?: : : : : : : : : : : : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2d
0dense_3/kernel/Regularizer/L2Loss/ReadVariableOp0dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ?
 
_user_specified_nameinputs


ѓ
B__inference_dense_5_layer_call_and_return_conditional_losses_44329

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
	
Џ
__inference_loss_fn_1_44347L
9dense_3_kernel_regularizer_l2loss_readvariableop_resource:	р 
identityЂ0dense_3/kernel/Regularizer/L2Loss/ReadVariableOpЋ
0dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp9dense_3_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes
:	р *
dtype0
!dense_3/kernel/Regularizer/L2LossL2Loss8dense_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ы?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0*dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentity"dense_3/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: y
NoOpNoOp1^dense_3/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_3/kernel/Regularizer/L2Loss/ReadVariableOp0dense_3/kernel/Regularizer/L2Loss/ReadVariableOp
В
Щ
,__inference_sequential_1_layer_call_fn_43531
conv2d_3_input!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5:	р 
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10:
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_43504o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:џџџџџџџџџ?: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ?
(
_user_specified_nameconv2d_3_input
В
Щ
,__inference_sequential_1_layer_call_fn_43777
conv2d_3_input!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5:	р 
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10:
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_43721o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:џџџџџџџџџ?: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ?
(
_user_specified_nameconv2d_3_input
Й
K
/__inference_max_pooling2d_3_layer_call_fn_44195

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_43351
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

С
,__inference_sequential_1_layer_call_fn_43982

inputs!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5:	р 
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10:
identityЂStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_43721o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:џџџџџџџџџ?: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ?
 
_user_specified_nameinputs
ы

(__inference_conv2d_5_layer_call_fn_44179

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_43411w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ВK
а
 __inference__wrapped_model_43330
conv2d_3_inputN
4sequential_1_conv2d_3_conv2d_readvariableop_resource: C
5sequential_1_conv2d_3_biasadd_readvariableop_resource: N
4sequential_1_conv2d_4_conv2d_readvariableop_resource: C
5sequential_1_conv2d_4_biasadd_readvariableop_resource:N
4sequential_1_conv2d_5_conv2d_readvariableop_resource:C
5sequential_1_conv2d_5_biasadd_readvariableop_resource:F
3sequential_1_dense_3_matmul_readvariableop_resource:	р B
4sequential_1_dense_3_biasadd_readvariableop_resource: E
3sequential_1_dense_4_matmul_readvariableop_resource:  B
4sequential_1_dense_4_biasadd_readvariableop_resource: E
3sequential_1_dense_5_matmul_readvariableop_resource: B
4sequential_1_dense_5_biasadd_readvariableop_resource:
identityЂ,sequential_1/conv2d_3/BiasAdd/ReadVariableOpЂ+sequential_1/conv2d_3/Conv2D/ReadVariableOpЂ,sequential_1/conv2d_4/BiasAdd/ReadVariableOpЂ+sequential_1/conv2d_4/Conv2D/ReadVariableOpЂ,sequential_1/conv2d_5/BiasAdd/ReadVariableOpЂ+sequential_1/conv2d_5/Conv2D/ReadVariableOpЂ+sequential_1/dense_3/BiasAdd/ReadVariableOpЂ*sequential_1/dense_3/MatMul/ReadVariableOpЂ+sequential_1/dense_4/BiasAdd/ReadVariableOpЂ*sequential_1/dense_4/MatMul/ReadVariableOpЂ+sequential_1/dense_5/BiasAdd/ReadVariableOpЂ*sequential_1/dense_5/MatMul/ReadVariableOpЈ
+sequential_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Э
sequential_1/conv2d_3/Conv2DConv2Dconv2d_3_input3sequential_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ? *
paddingSAME*
strides

,sequential_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0П
sequential_1/conv2d_3/BiasAddBiasAdd%sequential_1/conv2d_3/Conv2D:output:04sequential_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ? 
sequential_1/conv2d_3/ReluRelu&sequential_1/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ? Ц
$sequential_1/max_pooling2d_2/MaxPoolMaxPool(sequential_1/conv2d_3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
Ј
+sequential_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ь
sequential_1/conv2d_4/Conv2DConv2D-sequential_1/max_pooling2d_2/MaxPool:output:03sequential_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

,sequential_1/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
sequential_1/conv2d_4/BiasAddBiasAdd%sequential_1/conv2d_4/Conv2D:output:04sequential_1/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
sequential_1/conv2d_4/ReluRelu&sequential_1/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџЈ
+sequential_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ч
sequential_1/conv2d_5/Conv2DConv2D(sequential_1/conv2d_4/Relu:activations:03sequential_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

,sequential_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
sequential_1/conv2d_5/BiasAddBiasAdd%sequential_1/conv2d_5/Conv2D:output:04sequential_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
sequential_1/conv2d_5/ReluRelu&sequential_1/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџЦ
$sequential_1/max_pooling2d_3/MaxPoolMaxPool(sequential_1/conv2d_5/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
m
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџр   В
sequential_1/flatten_1/ReshapeReshape-sequential_1/max_pooling2d_3/MaxPool:output:0%sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџр
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource*
_output_shapes
:	р *
dtype0Д
sequential_1/dense_3/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Е
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ z
sequential_1/dense_3/ReluRelu%sequential_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
sequential_1/dropout_2/IdentityIdentity'sequential_1/dense_3/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 
*sequential_1/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Е
sequential_1/dense_4/MatMulMatMul(sequential_1/dropout_2/Identity:output:02sequential_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
+sequential_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Е
sequential_1/dense_4/BiasAddBiasAdd%sequential_1/dense_4/MatMul:product:03sequential_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ z
sequential_1/dense_4/ReluRelu%sequential_1/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
sequential_1/dropout_3/IdentityIdentity'sequential_1/dense_4/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 
*sequential_1/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Е
sequential_1/dense_5/MatMulMatMul(sequential_1/dropout_3/Identity:output:02sequential_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
+sequential_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Е
sequential_1/dense_5/BiasAddBiasAdd%sequential_1/dense_5/MatMul:product:03sequential_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
sequential_1/dense_5/SigmoidSigmoid%sequential_1/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџo
IdentityIdentity sequential_1/dense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџю
NoOpNoOp-^sequential_1/conv2d_3/BiasAdd/ReadVariableOp,^sequential_1/conv2d_3/Conv2D/ReadVariableOp-^sequential_1/conv2d_4/BiasAdd/ReadVariableOp,^sequential_1/conv2d_4/Conv2D/ReadVariableOp-^sequential_1/conv2d_5/BiasAdd/ReadVariableOp,^sequential_1/conv2d_5/Conv2D/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp+^sequential_1/dense_3/MatMul/ReadVariableOp,^sequential_1/dense_4/BiasAdd/ReadVariableOp+^sequential_1/dense_4/MatMul/ReadVariableOp,^sequential_1/dense_5/BiasAdd/ReadVariableOp+^sequential_1/dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:џџџџџџџџџ?: : : : : : : : : : : : 2\
,sequential_1/conv2d_3/BiasAdd/ReadVariableOp,sequential_1/conv2d_3/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_3/Conv2D/ReadVariableOp+sequential_1/conv2d_3/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_4/BiasAdd/ReadVariableOp,sequential_1/conv2d_4/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_4/Conv2D/ReadVariableOp+sequential_1/conv2d_4/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_5/BiasAdd/ReadVariableOp,sequential_1/conv2d_5/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_5/Conv2D/ReadVariableOp+sequential_1/conv2d_5/Conv2D/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2X
*sequential_1/dense_3/MatMul/ReadVariableOp*sequential_1/dense_3/MatMul/ReadVariableOp2Z
+sequential_1/dense_4/BiasAdd/ReadVariableOp+sequential_1/dense_4/BiasAdd/ReadVariableOp2X
*sequential_1/dense_4/MatMul/ReadVariableOp*sequential_1/dense_4/MatMul/ReadVariableOp2Z
+sequential_1/dense_5/BiasAdd/ReadVariableOp+sequential_1/dense_5/BiasAdd/ReadVariableOp2X
*sequential_1/dense_5/MatMul/ReadVariableOp*sequential_1/dense_5/MatMul/ReadVariableOp:_ [
/
_output_shapes
:џџџџџџџџџ?
(
_user_specified_nameconv2d_3_input

ќ
C__inference_conv2d_5_layer_call_and_return_conditional_losses_43411

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
з
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_44297

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


ѓ
B__inference_dense_5_layer_call_and_return_conditional_losses_43489

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ЙП
Ё
!__inference__traced_restore_44650
file_prefix:
 assignvariableop_conv2d_3_kernel: .
 assignvariableop_1_conv2d_3_bias: <
"assignvariableop_2_conv2d_4_kernel: .
 assignvariableop_3_conv2d_4_bias:<
"assignvariableop_4_conv2d_5_kernel:.
 assignvariableop_5_conv2d_5_bias:4
!assignvariableop_6_dense_3_kernel:	р -
assignvariableop_7_dense_3_bias: 3
!assignvariableop_8_dense_4_kernel:  -
assignvariableop_9_dense_4_bias: 4
"assignvariableop_10_dense_5_kernel: .
 assignvariableop_11_dense_5_bias:"
assignvariableop_12_iter:	 $
assignvariableop_13_beta_1: $
assignvariableop_14_beta_2: #
assignvariableop_15_decay: +
!assignvariableop_16_learning_rate: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: #
assignvariableop_19_total: #
assignvariableop_20_count: ?
%assignvariableop_21_conv2d_3_kernel_m: 1
#assignvariableop_22_conv2d_3_bias_m: ?
%assignvariableop_23_conv2d_4_kernel_m: 1
#assignvariableop_24_conv2d_4_bias_m:?
%assignvariableop_25_conv2d_5_kernel_m:1
#assignvariableop_26_conv2d_5_bias_m:7
$assignvariableop_27_dense_3_kernel_m:	р 0
"assignvariableop_28_dense_3_bias_m: 6
$assignvariableop_29_dense_4_kernel_m:  0
"assignvariableop_30_dense_4_bias_m: 6
$assignvariableop_31_dense_5_kernel_m: 0
"assignvariableop_32_dense_5_bias_m:?
%assignvariableop_33_conv2d_3_kernel_v: 1
#assignvariableop_34_conv2d_3_bias_v: ?
%assignvariableop_35_conv2d_4_kernel_v: 1
#assignvariableop_36_conv2d_4_bias_v:?
%assignvariableop_37_conv2d_5_kernel_v:1
#assignvariableop_38_conv2d_5_bias_v:7
$assignvariableop_39_dense_3_kernel_v:	р 0
"assignvariableop_40_dense_3_bias_v: 6
$assignvariableop_41_dense_4_kernel_v:  0
"assignvariableop_42_dense_4_bias_v: 6
$assignvariableop_43_dense_5_kernel_v: 0
"assignvariableop_44_dense_5_bias_v:
identity_46ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9І
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*Ь
valueТBП.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЬ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ю
_output_shapesЛ
И::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOpAssignVariableOp assignvariableop_conv2d_3_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_3_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_4_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_4_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_5_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_5_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_4_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_4_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_5_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_5_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:Б
AssignVariableOp_12AssignVariableOpassignvariableop_12_iterIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOp_13AssignVariableOpassignvariableop_13_beta_1Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOp_14AssignVariableOpassignvariableop_14_beta_2Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_15AssignVariableOpassignvariableop_15_decayIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_16AssignVariableOp!assignvariableop_16_learning_rateIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_21AssignVariableOp%assignvariableop_21_conv2d_3_kernel_mIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_22AssignVariableOp#assignvariableop_22_conv2d_3_bias_mIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_23AssignVariableOp%assignvariableop_23_conv2d_4_kernel_mIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv2d_4_bias_mIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_25AssignVariableOp%assignvariableop_25_conv2d_5_kernel_mIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_26AssignVariableOp#assignvariableop_26_conv2d_5_bias_mIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_27AssignVariableOp$assignvariableop_27_dense_3_kernel_mIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_28AssignVariableOp"assignvariableop_28_dense_3_bias_mIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_29AssignVariableOp$assignvariableop_29_dense_4_kernel_mIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_30AssignVariableOp"assignvariableop_30_dense_4_bias_mIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_31AssignVariableOp$assignvariableop_31_dense_5_kernel_mIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_32AssignVariableOp"assignvariableop_32_dense_5_bias_mIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_33AssignVariableOp%assignvariableop_33_conv2d_3_kernel_vIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_34AssignVariableOp#assignvariableop_34_conv2d_3_bias_vIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_35AssignVariableOp%assignvariableop_35_conv2d_4_kernel_vIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_36AssignVariableOp#assignvariableop_36_conv2d_4_bias_vIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_37AssignVariableOp%assignvariableop_37_conv2d_5_kernel_vIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_38AssignVariableOp#assignvariableop_38_conv2d_5_bias_vIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_39AssignVariableOp$assignvariableop_39_dense_3_kernel_vIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_40AssignVariableOp"assignvariableop_40_dense_3_bias_vIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_41AssignVariableOp$assignvariableop_41_dense_4_kernel_vIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_42AssignVariableOp"assignvariableop_42_dense_4_bias_vIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_43AssignVariableOp$assignvariableop_43_dense_5_kernel_vIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_44AssignVariableOp"assignvariableop_44_dense_5_bias_vIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ­
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_46IdentityIdentity_45:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ц
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_44211

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџр   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџрY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџр"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

А
C__inference_conv2d_3_layer_call_and_return_conditional_losses_43376

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЂ1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ? X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ? 
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ы? 
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ? Ћ
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ?
 
_user_specified_nameinputs
ы

(__inference_conv2d_3_layer_call_fn_44125

inputs!
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_43376w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ?: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ?
 
_user_specified_nameinputs
Й
K
/__inference_max_pooling2d_2_layer_call_fn_44145

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_43339
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_43339

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Џ	
И
__inference_loss_fn_0_44338T
:conv2d_3_kernel_regularizer_l2loss_readvariableop_resource: 
identityЂ1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpД
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:conv2d_3_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
: *
dtype0
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ы? 
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv2d_3/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp
з
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_43452

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


c
D__inference_dropout_2_layer_call_and_return_conditional_losses_43594

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  Ш?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ьQИ>І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
X


G__inference_sequential_1_layer_call_and_return_conditional_losses_44116

inputsA
'conv2d_3_conv2d_readvariableop_resource: 6
(conv2d_3_biasadd_readvariableop_resource: A
'conv2d_4_conv2d_readvariableop_resource: 6
(conv2d_4_biasadd_readvariableop_resource:A
'conv2d_5_conv2d_readvariableop_resource:6
(conv2d_5_biasadd_readvariableop_resource:9
&dense_3_matmul_readvariableop_resource:	р 5
'dense_3_biasadd_readvariableop_resource: 8
&dense_4_matmul_readvariableop_resource:  5
'dense_4_biasadd_readvariableop_resource: 8
&dense_5_matmul_readvariableop_resource: 5
'dense_5_biasadd_readvariableop_resource:
identityЂconv2d_3/BiasAdd/ReadVariableOpЂconv2d_3/Conv2D/ReadVariableOpЂ1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpЂconv2d_4/BiasAdd/ReadVariableOpЂconv2d_4/Conv2D/ReadVariableOpЂconv2d_5/BiasAdd/ReadVariableOpЂconv2d_5/Conv2D/ReadVariableOpЂdense_3/BiasAdd/ReadVariableOpЂdense_3/MatMul/ReadVariableOpЂ0dense_3/kernel/Regularizer/L2Loss/ReadVariableOpЂdense_4/BiasAdd/ReadVariableOpЂdense_4/MatMul/ReadVariableOpЂdense_5/BiasAdd/ReadVariableOpЂdense_5/MatMul/ReadVariableOp
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ћ
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ? *
paddingSAME*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ? j
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ? Ќ
max_pooling2d_2/MaxPoolMaxPoolconv2d_3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides

conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Х
conv2d_4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџj
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Р
conv2d_5/Conv2DConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџj
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџЌ
max_pooling2d_3/MaxPoolMaxPoolconv2d_5/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџр   
flatten_1/ReshapeReshape max_pooling2d_3/MaxPool:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџр
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	р *
dtype0
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ \
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  Ш?
dropout_2/dropout/MulMuldense_3/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ a
dropout_2/dropout/ShapeShapedense_3/Relu:activations:0*
T0*
_output_shapes
: 
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ьQИ>Ф
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ ^
dropout_2/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Л
dropout_2/dropout/SelectV2SelectV2"dropout_2/dropout/GreaterEqual:z:0dropout_2/dropout/Mul:z:0"dropout_2/dropout/Const_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_4/MatMulMatMul#dropout_2/dropout/SelectV2:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ \
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  Ш?
dropout_3/dropout/MulMuldense_4/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ a
dropout_3/dropout/ShapeShapedense_4/Relu:activations:0*
T0*
_output_shapes
: 
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ьQИ>Ф
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ ^
dropout_3/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Л
dropout_3/dropout/SelectV2SelectV2"dropout_3/dropout/GreaterEqual:z:0dropout_3/dropout/Mul:z:0"dropout_3/dropout/Const_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_5/MatMulMatMul#dropout_3/dropout/SelectV2:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџf
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџЁ
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ы? 
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
0dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	р *
dtype0
!dense_3/kernel/Regularizer/L2LossL2Loss8dense_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ы?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0*dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentitydense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЙ
NoOpNoOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp1^dense_3/kernel/Regularizer/L2Loss/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:џџџџџџџџџ?: : : : : : : : : : : : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2d
0dense_3/kernel/Regularizer/L2Loss/ReadVariableOp0dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ?
 
_user_specified_nameinputs
Ф

'__inference_dense_3_layer_call_fn_44220

inputs
unknown:	р 
	unknown_0: 
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_43441o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџр
 
_user_specified_nameinputs
С

'__inference_dense_4_layer_call_fn_44271

inputs
unknown:  
	unknown_0: 
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_43465o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


c
D__inference_dropout_3_layer_call_and_return_conditional_losses_44309

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  Ш?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ьQИ>І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_44150

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ђ
b
)__inference_dropout_2_layer_call_fn_44245

inputs
identityЂStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_43594o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
:
Џ
G__inference_sequential_1_layer_call_and_return_conditional_losses_43504

inputs(
conv2d_3_43377: 
conv2d_3_43379: (
conv2d_4_43395: 
conv2d_4_43397:(
conv2d_5_43412:
conv2d_5_43414: 
dense_3_43442:	р 
dense_3_43444: 
dense_4_43466:  
dense_4_43468: 
dense_5_43490: 
dense_5_43492:
identityЂ conv2d_3/StatefulPartitionedCallЂ1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpЂ conv2d_4/StatefulPartitionedCallЂ conv2d_5/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂ0dense_3/kernel/Regularizer/L2Loss/ReadVariableOpЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCallј
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_43377conv2d_3_43379*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_43376ѓ
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_43339
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_4_43395conv2d_4_43397*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_43394
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_43412conv2d_5_43414*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_43411ѓ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_43351п
flatten_1/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџр* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_43424
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_43442dense_3_43444*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_43441о
dropout_2/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_43452
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_4_43466dense_4_43468*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_43465о
dropout_3/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_43476
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_5_43490dense_5_43492*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_43489
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_3_43377*&
_output_shapes
: *
dtype0
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ы? 
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
0dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_3_43442*
_output_shapes
:	р *
dtype0
!dense_3/kernel/Regularizer/L2LossL2Loss8dense_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ы?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0*dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџќ
NoOpNoOp!^conv2d_3/StatefulPartitionedCall2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall1^dense_3/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:џџџџџџџџџ?: : : : : : : : : : : : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2d
0dense_3/kernel/Regularizer/L2Loss/ReadVariableOp0dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ?
 
_user_specified_nameinputs

А
C__inference_conv2d_3_layer_call_and_return_conditional_losses_44140

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЂ1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ? X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ? 
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ы? 
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ? Ћ
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ?
 
_user_specified_nameinputs


c
D__inference_dropout_2_layer_call_and_return_conditional_losses_44262

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  Ш?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ьQИ>І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


ѓ
B__inference_dense_4_layer_call_and_return_conditional_losses_43465

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


c
D__inference_dropout_3_layer_call_and_return_conditional_losses_43561

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  Ш?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ьQИ>І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

С
,__inference_sequential_1_layer_call_fn_43953

inputs!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5:	р 
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10:
identityЂStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_43504o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:џџџџџџџџџ?: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ?
 
_user_specified_nameinputs
=
ї
G__inference_sequential_1_layer_call_and_return_conditional_losses_43721

inputs(
conv2d_3_43677: 
conv2d_3_43679: (
conv2d_4_43683: 
conv2d_4_43685:(
conv2d_5_43688:
conv2d_5_43690: 
dense_3_43695:	р 
dense_3_43697: 
dense_4_43701:  
dense_4_43703: 
dense_5_43707: 
dense_5_43709:
identityЂ conv2d_3/StatefulPartitionedCallЂ1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpЂ conv2d_4/StatefulPartitionedCallЂ conv2d_5/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂ0dense_3/kernel/Regularizer/L2Loss/ReadVariableOpЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCallЂ!dropout_2/StatefulPartitionedCallЂ!dropout_3/StatefulPartitionedCallј
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_43677conv2d_3_43679*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_43376ѓ
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_43339
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_4_43683conv2d_4_43685*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_43394
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_43688conv2d_5_43690*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_43411ѓ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_43351п
flatten_1/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџр* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_43424
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_43695dense_3_43697*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_43441ю
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_43594
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_4_43701dense_4_43703*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_43465
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_43561
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_5_43707dense_5_43709*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_43489
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_3_43677*&
_output_shapes
: *
dtype0
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ы? 
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
0dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_3_43695*
_output_shapes
:	р *
dtype0
!dense_3/kernel/Regularizer/L2LossL2Loss8dense_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ы?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0*dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџФ
NoOpNoOp!^conv2d_3/StatefulPartitionedCall2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall1^dense_3/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:џџџџџџџџџ?: : : : : : : : : : : : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2d
0dense_3/kernel/Regularizer/L2Loss/ReadVariableOp0dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ?
 
_user_specified_nameinputs
ы

(__inference_conv2d_4_layer_call_fn_44159

inputs!
unknown: 
	unknown_0:
identityЂStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_43394w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
 
E
)__inference_dropout_3_layer_call_fn_44287

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_43476`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
з
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_43476

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

Ї
B__inference_dense_3_layer_call_and_return_conditional_losses_44235

inputs1
matmul_readvariableop_resource:	р -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ0dense_3/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
0dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р *
dtype0
!dense_3/kernel/Regularizer/L2LossL2Loss8dense_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ы?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0*dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ Њ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_3/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_3/kernel/Regularizer/L2Loss/ReadVariableOp0dense_3/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр
 
_user_specified_nameinputs

ќ
C__inference_conv2d_4_layer_call_and_return_conditional_losses_43394

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

ќ
C__inference_conv2d_5_layer_call_and_return_conditional_losses_44190

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_44200

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Ї
B__inference_dense_3_layer_call_and_return_conditional_losses_43441

inputs1
matmul_readvariableop_resource:	р -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ0dense_3/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
0dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р *
dtype0
!dense_3/kernel/Regularizer/L2LossL2Loss8dense_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ы?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0*dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ Њ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_3/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_3/kernel/Regularizer/L2Loss/ReadVariableOp0dense_3/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_43351

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ѕ=
џ
G__inference_sequential_1_layer_call_and_return_conditional_losses_43871
conv2d_3_input(
conv2d_3_43827: 
conv2d_3_43829: (
conv2d_4_43833: 
conv2d_4_43835:(
conv2d_5_43838:
conv2d_5_43840: 
dense_3_43845:	р 
dense_3_43847: 
dense_4_43851:  
dense_4_43853: 
dense_5_43857: 
dense_5_43859:
identityЂ conv2d_3/StatefulPartitionedCallЂ1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpЂ conv2d_4/StatefulPartitionedCallЂ conv2d_5/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂ0dense_3/kernel/Regularizer/L2Loss/ReadVariableOpЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCallЂ!dropout_2/StatefulPartitionedCallЂ!dropout_3/StatefulPartitionedCall
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputconv2d_3_43827conv2d_3_43829*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_43376ѓ
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_43339
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_4_43833conv2d_4_43835*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_43394
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_43838conv2d_5_43840*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_43411ѓ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_43351п
flatten_1/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџр* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_43424
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_43845dense_3_43847*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_43441ю
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_43594
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_4_43851dense_4_43853*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_43465
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_43561
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_5_43857dense_5_43859*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_43489
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_3_43827*&
_output_shapes
: *
dtype0
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ы? 
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
0dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_3_43845*
_output_shapes
:	р *
dtype0
!dense_3/kernel/Regularizer/L2LossL2Loss8dense_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ы?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0*dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџФ
NoOpNoOp!^conv2d_3/StatefulPartitionedCall2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall1^dense_3/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:џџџџџџџџџ?: : : : : : : : : : : : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2d
0dense_3/kernel/Regularizer/L2Loss/ReadVariableOp0dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ?
(
_user_specified_nameconv2d_3_input
ђ
b
)__inference_dropout_3_layer_call_fn_44292

inputs
identityЂStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_43561o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
В
E
)__inference_flatten_1_layer_call_fn_44205

inputs
identityГ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџр* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_43424a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџр"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


ѓ
B__inference_dense_4_layer_call_and_return_conditional_losses_44282

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
з
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_44250

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
 
E
)__inference_dropout_2_layer_call_fn_44240

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_43452`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs"
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Р
serving_defaultЌ
Q
conv2d_3_input?
 serving_default_conv2d_3_input:0џџџџџџџџџ?;
dense_50
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:Й

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
н
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
н
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias
 ,_jit_compiled_convolution_op"
_tf_keras_layer
н
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias
 5_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

Hkernel
Ibias"
_tf_keras_layer
М
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
P_random_generator"
_tf_keras_layer
Л
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

Wkernel
Xbias"
_tf_keras_layer
М
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
__random_generator"
_tf_keras_layer
Л
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses

fkernel
gbias"
_tf_keras_layer
v
0
1
*2
+3
34
45
H6
I7
W8
X9
f10
g11"
trackable_list_wrapper
v
0
1
*2
+3
34
45
H6
I7
W8
X9
f10
g11"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
Ъ
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
х
otrace_0
ptrace_1
qtrace_2
rtrace_32њ
,__inference_sequential_1_layer_call_fn_43531
,__inference_sequential_1_layer_call_fn_43953
,__inference_sequential_1_layer_call_fn_43982
,__inference_sequential_1_layer_call_fn_43777П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zotrace_0zptrace_1zqtrace_2zrtrace_3
б
strace_0
ttrace_1
utrace_2
vtrace_32ц
G__inference_sequential_1_layer_call_and_return_conditional_losses_44042
G__inference_sequential_1_layer_call_and_return_conditional_losses_44116
G__inference_sequential_1_layer_call_and_return_conditional_losses_43824
G__inference_sequential_1_layer_call_and_return_conditional_losses_43871П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zstrace_0zttrace_1zutrace_2zvtrace_3
вBЯ
 __inference__wrapped_model_43330conv2d_3_input"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
У
witer

xbeta_1

ybeta_2
	zdecay
{learning_ratemлmм*mн+mо3mп4mрHmсImтWmуXmфfmхgmцvчvш*vщ+vъ3vы4vьHvэIvюWvяXv№fvёgvђ"
	optimizer
,
|serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
h0"
trackable_list_wrapper
Џ
}non_trainable_variables

~layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ю
trace_02Я
(__inference_conv2d_3_layer_call_fn_44125Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ъ
C__inference_conv2d_3_layer_call_and_return_conditional_losses_44140Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
):' 2conv2d_3/kernel
: 2conv2d_3/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
ѕ
trace_02ж
/__inference_max_pooling2d_2_layer_call_fn_44145Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ё
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_44150Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
ю
trace_02Я
(__inference_conv2d_4_layer_call_fn_44159Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ъ
C__inference_conv2d_4_layer_call_and_return_conditional_losses_44170Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
):' 2conv2d_4/kernel
:2conv2d_4/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
ю
trace_02Я
(__inference_conv2d_5_layer_call_fn_44179Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ъ
C__inference_conv2d_5_layer_call_and_return_conditional_losses_44190Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
):'2conv2d_5/kernel
:2conv2d_5/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
ѕ
trace_02ж
/__inference_max_pooling2d_3_layer_call_fn_44195Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ё
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_44200Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
 non_trainable_variables
Ёlayers
Ђmetrics
 Ѓlayer_regularization_losses
Єlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
я
Ѕtrace_02а
)__inference_flatten_1_layer_call_fn_44205Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЅtrace_0

Іtrace_02ы
D__inference_flatten_1_layer_call_and_return_conditional_losses_44211Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zІtrace_0
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
'
i0"
trackable_list_wrapper
В
Їnon_trainable_variables
Јlayers
Љmetrics
 Њlayer_regularization_losses
Ћlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
э
Ќtrace_02Ю
'__inference_dense_3_layer_call_fn_44220Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЌtrace_0

­trace_02щ
B__inference_dense_3_layer_call_and_return_conditional_losses_44235Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z­trace_0
!:	р 2dense_3/kernel
: 2dense_3/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ўnon_trainable_variables
Џlayers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
Ч
Гtrace_0
Дtrace_12
)__inference_dropout_2_layer_call_fn_44240
)__inference_dropout_2_layer_call_fn_44245Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zГtrace_0zДtrace_1
§
Еtrace_0
Жtrace_12Т
D__inference_dropout_2_layer_call_and_return_conditional_losses_44250
D__inference_dropout_2_layer_call_and_return_conditional_losses_44262Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЕtrace_0zЖtrace_1
"
_generic_user_object
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
э
Мtrace_02Ю
'__inference_dense_4_layer_call_fn_44271Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zМtrace_0

Нtrace_02щ
B__inference_dense_4_layer_call_and_return_conditional_losses_44282Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zНtrace_0
 :  2dense_4/kernel
: 2dense_4/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
Ч
Уtrace_0
Фtrace_12
)__inference_dropout_3_layer_call_fn_44287
)__inference_dropout_3_layer_call_fn_44292Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zУtrace_0zФtrace_1
§
Хtrace_0
Цtrace_12Т
D__inference_dropout_3_layer_call_and_return_conditional_losses_44297
D__inference_dropout_3_layer_call_and_return_conditional_losses_44309Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zХtrace_0zЦtrace_1
"
_generic_user_object
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
э
Ьtrace_02Ю
'__inference_dense_5_layer_call_fn_44318Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЬtrace_0

Эtrace_02щ
B__inference_dense_5_layer_call_and_return_conditional_losses_44329Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЭtrace_0
 : 2dense_5/kernel
:2dense_5/bias
Ю
Юtrace_02Џ
__inference_loss_fn_0_44338
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ zЮtrace_0
Ю
Яtrace_02Џ
__inference_loss_fn_1_44347
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ zЯtrace_0
 "
trackable_list_wrapper
n
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
10"
trackable_list_wrapper
0
а0
б1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
,__inference_sequential_1_layer_call_fn_43531conv2d_3_input"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
,__inference_sequential_1_layer_call_fn_43953inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
,__inference_sequential_1_layer_call_fn_43982inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
,__inference_sequential_1_layer_call_fn_43777conv2d_3_input"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
G__inference_sequential_1_layer_call_and_return_conditional_losses_44042inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
G__inference_sequential_1_layer_call_and_return_conditional_losses_44116inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 B
G__inference_sequential_1_layer_call_and_return_conditional_losses_43824conv2d_3_input"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 B
G__inference_sequential_1_layer_call_and_return_conditional_losses_43871conv2d_3_input"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
:	 (2iter
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
бBЮ
#__inference_signature_wrapper_43916conv2d_3_input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
h0"
trackable_list_wrapper
 "
trackable_dict_wrapper
мBй
(__inference_conv2d_3_layer_call_fn_44125inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
C__inference_conv2d_3_layer_call_and_return_conditional_losses_44140inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
уBр
/__inference_max_pooling2d_2_layer_call_fn_44145inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_44150inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
мBй
(__inference_conv2d_4_layer_call_fn_44159inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
C__inference_conv2d_4_layer_call_and_return_conditional_losses_44170inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
мBй
(__inference_conv2d_5_layer_call_fn_44179inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
C__inference_conv2d_5_layer_call_and_return_conditional_losses_44190inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
уBр
/__inference_max_pooling2d_3_layer_call_fn_44195inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_44200inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
нBк
)__inference_flatten_1_layer_call_fn_44205inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
D__inference_flatten_1_layer_call_and_return_conditional_losses_44211inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
i0"
trackable_list_wrapper
 "
trackable_dict_wrapper
лBи
'__inference_dense_3_layer_call_fn_44220inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
B__inference_dense_3_layer_call_and_return_conditional_losses_44235inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
юBы
)__inference_dropout_2_layer_call_fn_44240inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
)__inference_dropout_2_layer_call_fn_44245inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_dropout_2_layer_call_and_return_conditional_losses_44250inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_dropout_2_layer_call_and_return_conditional_losses_44262inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
лBи
'__inference_dense_4_layer_call_fn_44271inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
B__inference_dense_4_layer_call_and_return_conditional_losses_44282inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
юBы
)__inference_dropout_3_layer_call_fn_44287inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
)__inference_dropout_3_layer_call_fn_44292inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_dropout_3_layer_call_and_return_conditional_losses_44297inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_dropout_3_layer_call_and_return_conditional_losses_44309inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
лBи
'__inference_dense_5_layer_call_fn_44318inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
B__inference_dense_5_layer_call_and_return_conditional_losses_44329inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ВBЏ
__inference_loss_fn_0_44338"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
ВBЏ
__inference_loss_fn_1_44347"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
R
в	variables
г	keras_api

дtotal

еcount"
_tf_keras_metric
c
ж	variables
з	keras_api

иtotal

йcount
к
_fn_kwargs"
_tf_keras_metric
0
д0
е1"
trackable_list_wrapper
.
в	variables"
_generic_user_object
:  (2total
:  (2count
0
и0
й1"
trackable_list_wrapper
.
ж	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
):' 2conv2d_3/kernel/m
: 2conv2d_3/bias/m
):' 2conv2d_4/kernel/m
:2conv2d_4/bias/m
):'2conv2d_5/kernel/m
:2conv2d_5/bias/m
!:	р 2dense_3/kernel/m
: 2dense_3/bias/m
 :  2dense_4/kernel/m
: 2dense_4/bias/m
 : 2dense_5/kernel/m
:2dense_5/bias/m
):' 2conv2d_3/kernel/v
: 2conv2d_3/bias/v
):' 2conv2d_4/kernel/v
:2conv2d_4/bias/v
):'2conv2d_5/kernel/v
:2conv2d_5/bias/v
!:	р 2dense_3/kernel/v
: 2dense_3/bias/v
 :  2dense_4/kernel/v
: 2dense_4/bias/v
 : 2dense_5/kernel/v
:2dense_5/bias/vЇ
 __inference__wrapped_model_43330*+34HIWXfg?Ђ<
5Ђ2
0-
conv2d_3_inputџџџџџџџџџ?
Њ "1Њ.
,
dense_5!
dense_5џџџџџџџџџК
C__inference_conv2d_3_layer_call_and_return_conditional_losses_44140s7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ?
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ? 
 
(__inference_conv2d_3_layer_call_fn_44125h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ?
Њ ")&
unknownџџџџџџџџџ? К
C__inference_conv2d_4_layer_call_and_return_conditional_losses_44170s*+7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ
 
(__inference_conv2d_4_layer_call_fn_44159h*+7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ ")&
unknownџџџџџџџџџК
C__inference_conv2d_5_layer_call_and_return_conditional_losses_44190s347Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ
 
(__inference_conv2d_5_layer_call_fn_44179h347Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ ")&
unknownџџџџџџџџџЊ
B__inference_dense_3_layer_call_and_return_conditional_losses_44235dHI0Ђ-
&Ђ#
!
inputsџџџџџџџџџр
Њ ",Ђ)
"
tensor_0џџџџџџџџџ 
 
'__inference_dense_3_layer_call_fn_44220YHI0Ђ-
&Ђ#
!
inputsџџџџџџџџџр
Њ "!
unknownџџџџџџџџџ Љ
B__inference_dense_4_layer_call_and_return_conditional_losses_44282cWX/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ 
 
'__inference_dense_4_layer_call_fn_44271XWX/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "!
unknownџџџџџџџџџ Љ
B__inference_dense_5_layer_call_and_return_conditional_losses_44329cfg/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
'__inference_dense_5_layer_call_fn_44318Xfg/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "!
unknownџџџџџџџџџЋ
D__inference_dropout_2_layer_call_and_return_conditional_losses_44250c3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ 
 Ћ
D__inference_dropout_2_layer_call_and_return_conditional_losses_44262c3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p
Њ ",Ђ)
"
tensor_0џџџџџџџџџ 
 
)__inference_dropout_2_layer_call_fn_44240X3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p 
Њ "!
unknownџџџџџџџџџ 
)__inference_dropout_2_layer_call_fn_44245X3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p
Њ "!
unknownџџџџџџџџџ Ћ
D__inference_dropout_3_layer_call_and_return_conditional_losses_44297c3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ 
 Ћ
D__inference_dropout_3_layer_call_and_return_conditional_losses_44309c3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p
Њ ",Ђ)
"
tensor_0џџџџџџџџџ 
 
)__inference_dropout_3_layer_call_fn_44287X3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p 
Њ "!
unknownџџџџџџџџџ 
)__inference_dropout_3_layer_call_fn_44292X3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p
Њ "!
unknownџџџџџџџџџ А
D__inference_flatten_1_layer_call_and_return_conditional_losses_44211h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
tensor_0џџџџџџџџџр
 
)__inference_flatten_1_layer_call_fn_44205]7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ ""
unknownџџџџџџџџџрC
__inference_loss_fn_0_44338$Ђ

Ђ 
Њ "
unknown C
__inference_loss_fn_1_44347$HЂ

Ђ 
Њ "
unknown є
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_44150ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ю
/__inference_max_pooling2d_2_layer_call_fn_44145RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџє
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_44200ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ю
/__inference_max_pooling2d_3_layer_call_fn_44195RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџб
G__inference_sequential_1_layer_call_and_return_conditional_losses_43824*+34HIWXfgGЂD
=Ђ:
0-
conv2d_3_inputџџџџџџџџџ?
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 б
G__inference_sequential_1_layer_call_and_return_conditional_losses_43871*+34HIWXfgGЂD
=Ђ:
0-
conv2d_3_inputџџџџџџџџџ?
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 Ш
G__inference_sequential_1_layer_call_and_return_conditional_losses_44042}*+34HIWXfg?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ?
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 Ш
G__inference_sequential_1_layer_call_and_return_conditional_losses_44116}*+34HIWXfg?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ?
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 Њ
,__inference_sequential_1_layer_call_fn_43531z*+34HIWXfgGЂD
=Ђ:
0-
conv2d_3_inputџџџџџџџџџ?
p 

 
Њ "!
unknownџџџџџџџџџЊ
,__inference_sequential_1_layer_call_fn_43777z*+34HIWXfgGЂD
=Ђ:
0-
conv2d_3_inputџџџџџџџџџ?
p

 
Њ "!
unknownџџџџџџџџџЂ
,__inference_sequential_1_layer_call_fn_43953r*+34HIWXfg?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ?
p 

 
Њ "!
unknownџџџџџџџџџЂ
,__inference_sequential_1_layer_call_fn_43982r*+34HIWXfg?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ?
p

 
Њ "!
unknownџџџџџџџџџМ
#__inference_signature_wrapper_43916*+34HIWXfgQЂN
Ђ 
GЊD
B
conv2d_3_input0-
conv2d_3_inputџџџџџџџџџ?"1Њ.
,
dense_5!
dense_5џџџџџџџџџ