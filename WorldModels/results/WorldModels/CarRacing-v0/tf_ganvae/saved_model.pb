ǭ#
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
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
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8??
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
?
enc_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameenc_conv1/kernel
}
$enc_conv1/kernel/Read/ReadVariableOpReadVariableOpenc_conv1/kernel*&
_output_shapes
: *
dtype0
t
enc_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameenc_conv1/bias
m
"enc_conv1/bias/Read/ReadVariableOpReadVariableOpenc_conv1/bias*
_output_shapes
: *
dtype0
?
enc_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameenc_conv2/kernel
}
$enc_conv2/kernel/Read/ReadVariableOpReadVariableOpenc_conv2/kernel*&
_output_shapes
: @*
dtype0
t
enc_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameenc_conv2/bias
m
"enc_conv2/bias/Read/ReadVariableOpReadVariableOpenc_conv2/bias*
_output_shapes
:@*
dtype0
?
enc_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*!
shared_nameenc_conv3/kernel
~
$enc_conv3/kernel/Read/ReadVariableOpReadVariableOpenc_conv3/kernel*'
_output_shapes
:@?*
dtype0
u
enc_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameenc_conv3/bias
n
"enc_conv3/bias/Read/ReadVariableOpReadVariableOpenc_conv3/bias*
_output_shapes	
:?*
dtype0
?
enc_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameenc_conv4/kernel

$enc_conv4/kernel/Read/ReadVariableOpReadVariableOpenc_conv4/kernel*(
_output_shapes
:??*
dtype0
u
enc_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameenc_conv4/bias
n
"enc_conv4/bias/Read/ReadVariableOpReadVariableOpenc_conv4/bias*
_output_shapes	
:?*
dtype0
}
enc_fc_mu/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *!
shared_nameenc_fc_mu/kernel
v
$enc_fc_mu/kernel/Read/ReadVariableOpReadVariableOpenc_fc_mu/kernel*
_output_shapes
:	? *
dtype0
t
enc_fc_mu/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameenc_fc_mu/bias
m
"enc_fc_mu/bias/Read/ReadVariableOpReadVariableOpenc_fc_mu/bias*
_output_shapes
: *
dtype0
?
enc_fc_log_var/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *&
shared_nameenc_fc_log_var/kernel
?
)enc_fc_log_var/kernel/Read/ReadVariableOpReadVariableOpenc_fc_log_var/kernel*
_output_shapes
:	? *
dtype0
~
enc_fc_log_var/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameenc_fc_log_var/bias
w
'enc_fc_log_var/bias/Read/ReadVariableOpReadVariableOpenc_fc_log_var/bias*
_output_shapes
: *
dtype0

dec_dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*"
shared_namedec_dense1/kernel
x
%dec_dense1/kernel/Read/ReadVariableOpReadVariableOpdec_dense1/kernel*
_output_shapes
:	 ?*
dtype0
w
dec_dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namedec_dense1/bias
p
#dec_dense1/bias/Read/ReadVariableOpReadVariableOpdec_dense1/bias*
_output_shapes	
:?*
dtype0
?
dec_deconv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*#
shared_namedec_deconv1/kernel
?
&dec_deconv1/kernel/Read/ReadVariableOpReadVariableOpdec_deconv1/kernel*(
_output_shapes
:??*
dtype0
y
dec_deconv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_namedec_deconv1/bias
r
$dec_deconv1/bias/Read/ReadVariableOpReadVariableOpdec_deconv1/bias*
_output_shapes	
:?*
dtype0
?
dec_deconv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*#
shared_namedec_deconv2/kernel
?
&dec_deconv2/kernel/Read/ReadVariableOpReadVariableOpdec_deconv2/kernel*'
_output_shapes
:@?*
dtype0
x
dec_deconv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_namedec_deconv2/bias
q
$dec_deconv2/bias/Read/ReadVariableOpReadVariableOpdec_deconv2/bias*
_output_shapes
:@*
dtype0
?
dec_deconv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*#
shared_namedec_deconv3/kernel
?
&dec_deconv3/kernel/Read/ReadVariableOpReadVariableOpdec_deconv3/kernel*&
_output_shapes
: @*
dtype0
x
dec_deconv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_namedec_deconv3/bias
q
$dec_deconv3/bias/Read/ReadVariableOpReadVariableOpdec_deconv3/bias*
_output_shapes
: *
dtype0
?
dec_deconv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namedec_deconv4/kernel
?
&dec_deconv4/kernel/Read/ReadVariableOpReadVariableOpdec_deconv4/kernel*&
_output_shapes
: *
dtype0
x
dec_deconv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namedec_deconv4/bias
q
$dec_deconv4/bias/Read/ReadVariableOpReadVariableOpdec_deconv4/bias*
_output_shapes
:*
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
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
?
Adam/enc_conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/enc_conv1/kernel/m
?
+Adam/enc_conv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_conv1/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/enc_conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/enc_conv1/bias/m
{
)Adam/enc_conv1/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_conv1/bias/m*
_output_shapes
: *
dtype0
?
Adam/enc_conv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/enc_conv2/kernel/m
?
+Adam/enc_conv2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_conv2/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/enc_conv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/enc_conv2/bias/m
{
)Adam/enc_conv2/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_conv2/bias/m*
_output_shapes
:@*
dtype0
?
Adam/enc_conv3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/enc_conv3/kernel/m
?
+Adam/enc_conv3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_conv3/kernel/m*'
_output_shapes
:@?*
dtype0
?
Adam/enc_conv3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/enc_conv3/bias/m
|
)Adam/enc_conv3/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_conv3/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/enc_conv4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/enc_conv4/kernel/m
?
+Adam/enc_conv4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_conv4/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/enc_conv4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/enc_conv4/bias/m
|
)Adam/enc_conv4/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_conv4/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/enc_fc_mu/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *(
shared_nameAdam/enc_fc_mu/kernel/m
?
+Adam/enc_fc_mu/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_fc_mu/kernel/m*
_output_shapes
:	? *
dtype0
?
Adam/enc_fc_mu/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/enc_fc_mu/bias/m
{
)Adam/enc_fc_mu/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_fc_mu/bias/m*
_output_shapes
: *
dtype0
?
Adam/enc_fc_log_var/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *-
shared_nameAdam/enc_fc_log_var/kernel/m
?
0Adam/enc_fc_log_var/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_fc_log_var/kernel/m*
_output_shapes
:	? *
dtype0
?
Adam/enc_fc_log_var/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameAdam/enc_fc_log_var/bias/m
?
.Adam/enc_fc_log_var/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_fc_log_var/bias/m*
_output_shapes
: *
dtype0
?
Adam/dec_dense1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*)
shared_nameAdam/dec_dense1/kernel/m
?
,Adam/dec_dense1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_dense1/kernel/m*
_output_shapes
:	 ?*
dtype0
?
Adam/dec_dense1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameAdam/dec_dense1/bias/m
~
*Adam/dec_dense1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_dense1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dec_deconv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??**
shared_nameAdam/dec_deconv1/kernel/m
?
-Adam/dec_deconv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_deconv1/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/dec_deconv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/dec_deconv1/bias/m
?
+Adam/dec_deconv1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_deconv1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dec_deconv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?**
shared_nameAdam/dec_deconv2/kernel/m
?
-Adam/dec_deconv2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_deconv2/kernel/m*'
_output_shapes
:@?*
dtype0
?
Adam/dec_deconv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/dec_deconv2/bias/m

+Adam/dec_deconv2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_deconv2/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dec_deconv3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameAdam/dec_deconv3/kernel/m
?
-Adam/dec_deconv3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_deconv3/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/dec_deconv3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/dec_deconv3/bias/m

+Adam/dec_deconv3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_deconv3/bias/m*
_output_shapes
: *
dtype0
?
Adam/dec_deconv4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/dec_deconv4/kernel/m
?
-Adam/dec_deconv4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_deconv4/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/dec_deconv4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/dec_deconv4/bias/m

+Adam/dec_deconv4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_deconv4/bias/m*
_output_shapes
:*
dtype0
?
Adam/enc_conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/enc_conv1/kernel/v
?
+Adam/enc_conv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_conv1/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/enc_conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/enc_conv1/bias/v
{
)Adam/enc_conv1/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_conv1/bias/v*
_output_shapes
: *
dtype0
?
Adam/enc_conv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/enc_conv2/kernel/v
?
+Adam/enc_conv2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_conv2/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/enc_conv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/enc_conv2/bias/v
{
)Adam/enc_conv2/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_conv2/bias/v*
_output_shapes
:@*
dtype0
?
Adam/enc_conv3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/enc_conv3/kernel/v
?
+Adam/enc_conv3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_conv3/kernel/v*'
_output_shapes
:@?*
dtype0
?
Adam/enc_conv3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/enc_conv3/bias/v
|
)Adam/enc_conv3/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_conv3/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/enc_conv4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/enc_conv4/kernel/v
?
+Adam/enc_conv4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_conv4/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/enc_conv4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/enc_conv4/bias/v
|
)Adam/enc_conv4/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_conv4/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/enc_fc_mu/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *(
shared_nameAdam/enc_fc_mu/kernel/v
?
+Adam/enc_fc_mu/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_fc_mu/kernel/v*
_output_shapes
:	? *
dtype0
?
Adam/enc_fc_mu/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/enc_fc_mu/bias/v
{
)Adam/enc_fc_mu/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_fc_mu/bias/v*
_output_shapes
: *
dtype0
?
Adam/enc_fc_log_var/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *-
shared_nameAdam/enc_fc_log_var/kernel/v
?
0Adam/enc_fc_log_var/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_fc_log_var/kernel/v*
_output_shapes
:	? *
dtype0
?
Adam/enc_fc_log_var/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameAdam/enc_fc_log_var/bias/v
?
.Adam/enc_fc_log_var/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_fc_log_var/bias/v*
_output_shapes
: *
dtype0
?
Adam/dec_dense1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*)
shared_nameAdam/dec_dense1/kernel/v
?
,Adam/dec_dense1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_dense1/kernel/v*
_output_shapes
:	 ?*
dtype0
?
Adam/dec_dense1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameAdam/dec_dense1/bias/v
~
*Adam/dec_dense1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_dense1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dec_deconv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??**
shared_nameAdam/dec_deconv1/kernel/v
?
-Adam/dec_deconv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_deconv1/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/dec_deconv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/dec_deconv1/bias/v
?
+Adam/dec_deconv1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_deconv1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dec_deconv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?**
shared_nameAdam/dec_deconv2/kernel/v
?
-Adam/dec_deconv2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_deconv2/kernel/v*'
_output_shapes
:@?*
dtype0
?
Adam/dec_deconv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/dec_deconv2/bias/v

+Adam/dec_deconv2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_deconv2/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dec_deconv3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameAdam/dec_deconv3/kernel/v
?
-Adam/dec_deconv3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_deconv3/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/dec_deconv3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/dec_deconv3/bias/v

+Adam/dec_deconv3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_deconv3/bias/v*
_output_shapes
: *
dtype0
?
Adam/dec_deconv4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/dec_deconv4/kernel/v
?
-Adam/dec_deconv4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_deconv4/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/dec_deconv4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/dec_deconv4/bias/v

+Adam/dec_deconv4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_deconv4/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?~
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?}
value?}B?} B?}
?
	optimizer
inference_net_base

mu_net

logvar_net
generative_net
loss
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
?
iter

beta_1

beta_2
	decay
learning_rate.m?/m?0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?.v?/v?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
trainable_variables
	variables
regularization_losses
	keras_api
y
layer_with_weights-0
layer-0
trainable_variables
	variables
regularization_losses
	keras_api
y
layer_with_weights-0
layer-0
 trainable_variables
!	variables
"regularization_losses
#	keras_api
?
$layer_with_weights-0
$layer-0
%layer-1
&layer_with_weights-1
&layer-2
'layer_with_weights-2
'layer-3
(layer_with_weights-3
(layer-4
)layer_with_weights-4
)layer-5
*trainable_variables
+	variables
,regularization_losses
-	keras_api
 
?
.0
/1
02
13
24
35
46
57
68
79
810
911
:12
;13
<14
=15
>16
?17
@18
A19
B20
C21
?
.0
/1
02
13
24
35
46
57
68
79
810
911
:12
;13
<14
=15
>16
?17
@18
A19
B20
C21
 
?
Dlayer_regularization_losses
Emetrics
Flayer_metrics
trainable_variables
Gnon_trainable_variables
	variables
	regularization_losses

Hlayers
 
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
h

.kernel
/bias
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
h

0kernel
1bias
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
h

2kernel
3bias
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
h

4kernel
5bias
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
R
Ytrainable_variables
Z	variables
[regularization_losses
\	keras_api
8
.0
/1
02
13
24
35
46
57
8
.0
/1
02
13
24
35
46
57
 
?
]layer_regularization_losses
^metrics
_layer_metrics
trainable_variables
`non_trainable_variables
	variables
regularization_losses

alayers
h

6kernel
7bias
btrainable_variables
c	variables
dregularization_losses
e	keras_api

60
71

60
71
 
?
flayer_regularization_losses
gmetrics
hlayer_metrics
trainable_variables
inon_trainable_variables
	variables
regularization_losses

jlayers
h

8kernel
9bias
ktrainable_variables
l	variables
mregularization_losses
n	keras_api

80
91

80
91
 
?
olayer_regularization_losses
pmetrics
qlayer_metrics
 trainable_variables
rnon_trainable_variables
!	variables
"regularization_losses

slayers
h

:kernel
;bias
ttrainable_variables
u	variables
vregularization_losses
w	keras_api
R
xtrainable_variables
y	variables
zregularization_losses
{	keras_api
h

<kernel
=bias
|trainable_variables
}	variables
~regularization_losses
	keras_api
l

>kernel
?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

@kernel
Abias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

Bkernel
Cbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
F
:0
;1
<2
=3
>4
?5
@6
A7
B8
C9
F
:0
;1
<2
=3
>4
?5
@6
A7
B8
C9
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
*trainable_variables
?non_trainable_variables
+	variables
,regularization_losses
?layers
VT
VARIABLE_VALUEenc_conv1/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEenc_conv1/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEenc_conv2/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEenc_conv2/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEenc_conv3/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEenc_conv3/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEenc_conv4/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEenc_conv4/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEenc_fc_mu/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEenc_fc_mu/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEenc_fc_log_var/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEenc_fc_log_var/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdec_dense1/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdec_dense1/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdec_deconv1/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdec_deconv1/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdec_deconv2/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdec_deconv2/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdec_deconv3/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdec_deconv3/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdec_deconv4/kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdec_deconv4/bias1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
?2
 
 

0
1
2
3

.0
/1

.0
/1
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
Itrainable_variables
?non_trainable_variables
J	variables
Kregularization_losses
?layers

00
11

00
11
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
Mtrainable_variables
?non_trainable_variables
N	variables
Oregularization_losses
?layers

20
31

20
31
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
Qtrainable_variables
?non_trainable_variables
R	variables
Sregularization_losses
?layers

40
51

40
51
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
Utrainable_variables
?non_trainable_variables
V	variables
Wregularization_losses
?layers
 
 
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
Ytrainable_variables
?non_trainable_variables
Z	variables
[regularization_losses
?layers
 
 
 
 
#
0
1
2
3
4

60
71

60
71
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
btrainable_variables
?non_trainable_variables
c	variables
dregularization_losses
?layers
 
 
 
 

0

80
91

80
91
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
ktrainable_variables
?non_trainable_variables
l	variables
mregularization_losses
?layers
 
 
 
 

0

:0
;1

:0
;1
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
ttrainable_variables
?non_trainable_variables
u	variables
vregularization_losses
?layers
 
 
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
xtrainable_variables
?non_trainable_variables
y	variables
zregularization_losses
?layers

<0
=1

<0
=1
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
|trainable_variables
?non_trainable_variables
}	variables
~regularization_losses
?layers

>0
?1

>0
?1
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?regularization_losses
?layers

@0
A1

@0
A1
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?regularization_losses
?layers

B0
C1

B0
C1
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?regularization_losses
?layers
 
 
 
 
*
$0
%1
&2
'3
(4
)5
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
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
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
yw
VARIABLE_VALUEAdam/enc_conv1/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/enc_conv1/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/enc_conv2/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/enc_conv2/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/enc_conv3/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/enc_conv3/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/enc_conv4/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/enc_conv4/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/enc_fc_mu/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/enc_fc_mu/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/enc_fc_log_var/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/enc_fc_log_var/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dec_dense1/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dec_dense1/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dec_deconv1/kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dec_deconv1/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dec_deconv2/kernel/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dec_deconv2/bias/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dec_deconv3/kernel/mMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dec_deconv3/bias/mMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dec_deconv4/kernel/mMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dec_deconv4/bias/mMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/enc_conv1/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/enc_conv1/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/enc_conv2/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/enc_conv2/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/enc_conv3/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/enc_conv3/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/enc_conv4/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/enc_conv4/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/enc_fc_mu/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/enc_fc_mu/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/enc_fc_log_var/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/enc_fc_log_var/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dec_dense1/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dec_dense1/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dec_deconv1/kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dec_deconv1/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dec_deconv2/kernel/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dec_deconv2/bias/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dec_deconv3/kernel/vMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dec_deconv3/bias/vMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dec_deconv4/kernel/vMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dec_deconv4/bias/vMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????@@*
dtype0*$
shape:?????????@@
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1enc_conv1/kernelenc_conv1/biasenc_conv2/kernelenc_conv2/biasenc_conv3/kernelenc_conv3/biasenc_conv4/kernelenc_conv4/biasenc_fc_mu/kernelenc_fc_mu/biasenc_fc_log_var/kernelenc_fc_log_var/biasdec_dense1/kerneldec_dense1/biasdec_deconv1/kerneldec_deconv1/biasdec_deconv2/kerneldec_deconv2/biasdec_deconv3/kerneldec_deconv3/biasdec_deconv4/kerneldec_deconv4/bias*"
Tin
2*
Tout
2*B
_output_shapes0
.:?????????@:?????????@@*8
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*-
f(R&
$__inference_signature_wrapper_878633
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp$enc_conv1/kernel/Read/ReadVariableOp"enc_conv1/bias/Read/ReadVariableOp$enc_conv2/kernel/Read/ReadVariableOp"enc_conv2/bias/Read/ReadVariableOp$enc_conv3/kernel/Read/ReadVariableOp"enc_conv3/bias/Read/ReadVariableOp$enc_conv4/kernel/Read/ReadVariableOp"enc_conv4/bias/Read/ReadVariableOp$enc_fc_mu/kernel/Read/ReadVariableOp"enc_fc_mu/bias/Read/ReadVariableOp)enc_fc_log_var/kernel/Read/ReadVariableOp'enc_fc_log_var/bias/Read/ReadVariableOp%dec_dense1/kernel/Read/ReadVariableOp#dec_dense1/bias/Read/ReadVariableOp&dec_deconv1/kernel/Read/ReadVariableOp$dec_deconv1/bias/Read/ReadVariableOp&dec_deconv2/kernel/Read/ReadVariableOp$dec_deconv2/bias/Read/ReadVariableOp&dec_deconv3/kernel/Read/ReadVariableOp$dec_deconv3/bias/Read/ReadVariableOp&dec_deconv4/kernel/Read/ReadVariableOp$dec_deconv4/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp+Adam/enc_conv1/kernel/m/Read/ReadVariableOp)Adam/enc_conv1/bias/m/Read/ReadVariableOp+Adam/enc_conv2/kernel/m/Read/ReadVariableOp)Adam/enc_conv2/bias/m/Read/ReadVariableOp+Adam/enc_conv3/kernel/m/Read/ReadVariableOp)Adam/enc_conv3/bias/m/Read/ReadVariableOp+Adam/enc_conv4/kernel/m/Read/ReadVariableOp)Adam/enc_conv4/bias/m/Read/ReadVariableOp+Adam/enc_fc_mu/kernel/m/Read/ReadVariableOp)Adam/enc_fc_mu/bias/m/Read/ReadVariableOp0Adam/enc_fc_log_var/kernel/m/Read/ReadVariableOp.Adam/enc_fc_log_var/bias/m/Read/ReadVariableOp,Adam/dec_dense1/kernel/m/Read/ReadVariableOp*Adam/dec_dense1/bias/m/Read/ReadVariableOp-Adam/dec_deconv1/kernel/m/Read/ReadVariableOp+Adam/dec_deconv1/bias/m/Read/ReadVariableOp-Adam/dec_deconv2/kernel/m/Read/ReadVariableOp+Adam/dec_deconv2/bias/m/Read/ReadVariableOp-Adam/dec_deconv3/kernel/m/Read/ReadVariableOp+Adam/dec_deconv3/bias/m/Read/ReadVariableOp-Adam/dec_deconv4/kernel/m/Read/ReadVariableOp+Adam/dec_deconv4/bias/m/Read/ReadVariableOp+Adam/enc_conv1/kernel/v/Read/ReadVariableOp)Adam/enc_conv1/bias/v/Read/ReadVariableOp+Adam/enc_conv2/kernel/v/Read/ReadVariableOp)Adam/enc_conv2/bias/v/Read/ReadVariableOp+Adam/enc_conv3/kernel/v/Read/ReadVariableOp)Adam/enc_conv3/bias/v/Read/ReadVariableOp+Adam/enc_conv4/kernel/v/Read/ReadVariableOp)Adam/enc_conv4/bias/v/Read/ReadVariableOp+Adam/enc_fc_mu/kernel/v/Read/ReadVariableOp)Adam/enc_fc_mu/bias/v/Read/ReadVariableOp0Adam/enc_fc_log_var/kernel/v/Read/ReadVariableOp.Adam/enc_fc_log_var/bias/v/Read/ReadVariableOp,Adam/dec_dense1/kernel/v/Read/ReadVariableOp*Adam/dec_dense1/bias/v/Read/ReadVariableOp-Adam/dec_deconv1/kernel/v/Read/ReadVariableOp+Adam/dec_deconv1/bias/v/Read/ReadVariableOp-Adam/dec_deconv2/kernel/v/Read/ReadVariableOp+Adam/dec_deconv2/bias/v/Read/ReadVariableOp-Adam/dec_deconv3/kernel/v/Read/ReadVariableOp+Adam/dec_deconv3/bias/v/Read/ReadVariableOp-Adam/dec_deconv4/kernel/v/Read/ReadVariableOp+Adam/dec_deconv4/bias/v/Read/ReadVariableOpConst*Z
TinS
Q2O	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*(
f#R!
__inference__traced_save_880586
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateenc_conv1/kernelenc_conv1/biasenc_conv2/kernelenc_conv2/biasenc_conv3/kernelenc_conv3/biasenc_conv4/kernelenc_conv4/biasenc_fc_mu/kernelenc_fc_mu/biasenc_fc_log_var/kernelenc_fc_log_var/biasdec_dense1/kerneldec_dense1/biasdec_deconv1/kerneldec_deconv1/biasdec_deconv2/kerneldec_deconv2/biasdec_deconv3/kerneldec_deconv3/biasdec_deconv4/kerneldec_deconv4/biastotalcounttotal_1count_1total_2count_2Adam/enc_conv1/kernel/mAdam/enc_conv1/bias/mAdam/enc_conv2/kernel/mAdam/enc_conv2/bias/mAdam/enc_conv3/kernel/mAdam/enc_conv3/bias/mAdam/enc_conv4/kernel/mAdam/enc_conv4/bias/mAdam/enc_fc_mu/kernel/mAdam/enc_fc_mu/bias/mAdam/enc_fc_log_var/kernel/mAdam/enc_fc_log_var/bias/mAdam/dec_dense1/kernel/mAdam/dec_dense1/bias/mAdam/dec_deconv1/kernel/mAdam/dec_deconv1/bias/mAdam/dec_deconv2/kernel/mAdam/dec_deconv2/bias/mAdam/dec_deconv3/kernel/mAdam/dec_deconv3/bias/mAdam/dec_deconv4/kernel/mAdam/dec_deconv4/bias/mAdam/enc_conv1/kernel/vAdam/enc_conv1/bias/vAdam/enc_conv2/kernel/vAdam/enc_conv2/bias/vAdam/enc_conv3/kernel/vAdam/enc_conv3/bias/vAdam/enc_conv4/kernel/vAdam/enc_conv4/bias/vAdam/enc_fc_mu/kernel/vAdam/enc_fc_mu/bias/vAdam/enc_fc_log_var/kernel/vAdam/enc_fc_log_var/bias/vAdam/dec_dense1/kernel/vAdam/dec_dense1/bias/vAdam/dec_deconv1/kernel/vAdam/dec_deconv1/bias/vAdam/dec_deconv2/kernel/vAdam/dec_deconv2/bias/vAdam/dec_deconv3/kernel/vAdam/dec_deconv3/bias/vAdam/dec_deconv4/kernel/vAdam/dec_deconv4/bias/v*Y
TinR
P2N*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference__traced_restore_880829ۓ
?
_
C__inference_reshape_layer_call_and_return_conditional_losses_877906

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_877446

inputs
enc_conv1_877424
enc_conv1_877426
enc_conv2_877429
enc_conv2_877431
enc_conv3_877434
enc_conv3_877436
enc_conv4_877439
enc_conv4_877441
identity??!enc_conv1/StatefulPartitionedCall?!enc_conv2/StatefulPartitionedCall?!enc_conv3/StatefulPartitionedCall?!enc_conv4/StatefulPartitionedCall?
!enc_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsenc_conv1_877424enc_conv1_877426*
Tin
2*
Tout
2*/
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_conv1_layer_call_and_return_conditional_losses_8772782#
!enc_conv1/StatefulPartitionedCall?
!enc_conv2/StatefulPartitionedCallStatefulPartitionedCall*enc_conv1/StatefulPartitionedCall:output:0enc_conv2_877429enc_conv2_877431*
Tin
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_conv2_layer_call_and_return_conditional_losses_8773002#
!enc_conv2/StatefulPartitionedCall?
!enc_conv3/StatefulPartitionedCallStatefulPartitionedCall*enc_conv2/StatefulPartitionedCall:output:0enc_conv3_877434enc_conv3_877436*
Tin
2*
Tout
2*0
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_conv3_layer_call_and_return_conditional_losses_8773222#
!enc_conv3/StatefulPartitionedCall?
!enc_conv4/StatefulPartitionedCallStatefulPartitionedCall*enc_conv3/StatefulPartitionedCall:output:0enc_conv4_877439enc_conv4_877441*
Tin
2*
Tout
2*0
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_conv4_layer_call_and_return_conditional_losses_8773442#
!enc_conv4/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall*enc_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_8773842
flatten/PartitionedCall?
IdentityIdentity flatten/PartitionedCall:output:0"^enc_conv1/StatefulPartitionedCall"^enc_conv2/StatefulPartitionedCall"^enc_conv3/StatefulPartitionedCall"^enc_conv4/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????@@::::::::2F
!enc_conv1/StatefulPartitionedCall!enc_conv1/StatefulPartitionedCall2F
!enc_conv2/StatefulPartitionedCall!enc_conv2/StatefulPartitionedCall2F
!enc_conv3/StatefulPartitionedCall!enc_conv3/StatefulPartitionedCall2F
!enc_conv4/StatefulPartitionedCall!enc_conv4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_877384

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_2_layer_call_fn_877647
input_3
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_8776402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: 
?%
?
G__inference_dec_deconv2_layer_call_and_return_conditional_losses_877753

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3?
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????:::j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?!
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_877965
input_4
dec_dense1_877938
dec_dense1_877940
dec_deconv1_877944
dec_deconv1_877946
dec_deconv2_877949
dec_deconv2_877951
dec_deconv3_877954
dec_deconv3_877956
dec_deconv4_877959
dec_deconv4_877961
identity??#dec_deconv1/StatefulPartitionedCall?#dec_deconv2/StatefulPartitionedCall?#dec_deconv3/StatefulPartitionedCall?#dec_deconv4/StatefulPartitionedCall?"dec_dense1/StatefulPartitionedCall?
"dec_dense1/StatefulPartitionedCallStatefulPartitionedCallinput_4dec_dense1_877938dec_dense1_877940*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_dec_dense1_layer_call_and_return_conditional_losses_8778762$
"dec_dense1/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall+dec_dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_8779062
reshape/PartitionedCall?
#dec_deconv1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0dec_deconv1_877944dec_deconv1_877946*
Tin
2*
Tout
2*B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dec_deconv1_layer_call_and_return_conditional_losses_8777042%
#dec_deconv1/StatefulPartitionedCall?
#dec_deconv2/StatefulPartitionedCallStatefulPartitionedCall,dec_deconv1/StatefulPartitionedCall:output:0dec_deconv2_877949dec_deconv2_877951*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dec_deconv2_layer_call_and_return_conditional_losses_8777532%
#dec_deconv2/StatefulPartitionedCall?
#dec_deconv3/StatefulPartitionedCallStatefulPartitionedCall,dec_deconv2/StatefulPartitionedCall:output:0dec_deconv3_877954dec_deconv3_877956*
Tin
2*
Tout
2*A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dec_deconv3_layer_call_and_return_conditional_losses_8778022%
#dec_deconv3/StatefulPartitionedCall?
#dec_deconv4/StatefulPartitionedCallStatefulPartitionedCall,dec_deconv3/StatefulPartitionedCall:output:0dec_deconv4_877959dec_deconv4_877961*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dec_deconv4_layer_call_and_return_conditional_losses_8778512%
#dec_deconv4/StatefulPartitionedCall?
IdentityIdentity,dec_deconv4/StatefulPartitionedCall:output:0$^dec_deconv1/StatefulPartitionedCall$^dec_deconv2/StatefulPartitionedCall$^dec_deconv3/StatefulPartitionedCall$^dec_deconv4/StatefulPartitionedCall#^dec_dense1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:????????? ::::::::::2J
#dec_deconv1/StatefulPartitionedCall#dec_deconv1/StatefulPartitionedCall2J
#dec_deconv2/StatefulPartitionedCall#dec_deconv2/StatefulPartitionedCall2J
#dec_deconv3/StatefulPartitionedCall#dec_deconv3/StatefulPartitionedCall2J
#dec_deconv4/StatefulPartitionedCall#dec_deconv4/StatefulPartitionedCall2H
"dec_dense1/StatefulPartitionedCall"dec_dense1/StatefulPartitionedCall:P L
'
_output_shapes
:????????? 
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_877551
input_2
enc_fc_mu_877545
enc_fc_mu_877547
identity??!enc_fc_mu/StatefulPartitionedCall?
!enc_fc_mu/StatefulPartitionedCallStatefulPartitionedCallinput_2enc_fc_mu_877545enc_fc_mu_877547*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_fc_mu_layer_call_and_return_conditional_losses_8775252#
!enc_fc_mu/StatefulPartitionedCall?
IdentityIdentity*enc_fc_mu/StatefulPartitionedCall:output:0"^enc_fc_mu/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2F
!enc_fc_mu/StatefulPartitionedCall!enc_fc_mu/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: 
?
_
C__inference_reshape_layer_call_and_return_conditional_losses_880322

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_enc_fc_log_var_layer_call_and_return_conditional_losses_880279

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
??
?
C__inference_gan_vae_layer_call_and_return_conditional_losses_879587
input_17
3sequential_enc_conv1_conv2d_readvariableop_resource8
4sequential_enc_conv1_biasadd_readvariableop_resource7
3sequential_enc_conv2_conv2d_readvariableop_resource8
4sequential_enc_conv2_biasadd_readvariableop_resource7
3sequential_enc_conv3_conv2d_readvariableop_resource8
4sequential_enc_conv3_biasadd_readvariableop_resource7
3sequential_enc_conv4_conv2d_readvariableop_resource8
4sequential_enc_conv4_biasadd_readvariableop_resource9
5sequential_1_enc_fc_mu_matmul_readvariableop_resource:
6sequential_1_enc_fc_mu_biasadd_readvariableop_resource>
:sequential_2_enc_fc_log_var_matmul_readvariableop_resource?
;sequential_2_enc_fc_log_var_biasadd_readvariableop_resource:
6sequential_3_dec_dense1_matmul_readvariableop_resource;
7sequential_3_dec_dense1_biasadd_readvariableop_resourceE
Asequential_3_dec_deconv1_conv2d_transpose_readvariableop_resource<
8sequential_3_dec_deconv1_biasadd_readvariableop_resourceE
Asequential_3_dec_deconv2_conv2d_transpose_readvariableop_resource<
8sequential_3_dec_deconv2_biasadd_readvariableop_resourceE
Asequential_3_dec_deconv3_conv2d_transpose_readvariableop_resource<
8sequential_3_dec_deconv3_biasadd_readvariableop_resourceE
Asequential_3_dec_deconv4_conv2d_transpose_readvariableop_resource<
8sequential_3_dec_deconv4_biasadd_readvariableop_resource
identity

identity_1??
*sequential/enc_conv1/Conv2D/ReadVariableOpReadVariableOp3sequential_enc_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*sequential/enc_conv1/Conv2D/ReadVariableOp?
sequential/enc_conv1/Conv2DConv2Dinput_12sequential/enc_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
sequential/enc_conv1/Conv2D?
+sequential/enc_conv1/BiasAdd/ReadVariableOpReadVariableOp4sequential_enc_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential/enc_conv1/BiasAdd/ReadVariableOp?
sequential/enc_conv1/BiasAddBiasAdd$sequential/enc_conv1/Conv2D:output:03sequential/enc_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential/enc_conv1/BiasAdd?
sequential/enc_conv1/ReluRelu%sequential/enc_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential/enc_conv1/Relu?
*sequential/enc_conv2/Conv2D/ReadVariableOpReadVariableOp3sequential_enc_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*sequential/enc_conv2/Conv2D/ReadVariableOp?
sequential/enc_conv2/Conv2DConv2D'sequential/enc_conv1/Relu:activations:02sequential/enc_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
sequential/enc_conv2/Conv2D?
+sequential/enc_conv2/BiasAdd/ReadVariableOpReadVariableOp4sequential_enc_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential/enc_conv2/BiasAdd/ReadVariableOp?
sequential/enc_conv2/BiasAddBiasAdd$sequential/enc_conv2/Conv2D:output:03sequential/enc_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential/enc_conv2/BiasAdd?
sequential/enc_conv2/ReluRelu%sequential/enc_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential/enc_conv2/Relu?
*sequential/enc_conv3/Conv2D/ReadVariableOpReadVariableOp3sequential_enc_conv3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02,
*sequential/enc_conv3/Conv2D/ReadVariableOp?
sequential/enc_conv3/Conv2DConv2D'sequential/enc_conv2/Relu:activations:02sequential/enc_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential/enc_conv3/Conv2D?
+sequential/enc_conv3/BiasAdd/ReadVariableOpReadVariableOp4sequential_enc_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential/enc_conv3/BiasAdd/ReadVariableOp?
sequential/enc_conv3/BiasAddBiasAdd$sequential/enc_conv3/Conv2D:output:03sequential/enc_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential/enc_conv3/BiasAdd?
sequential/enc_conv3/ReluRelu%sequential/enc_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential/enc_conv3/Relu?
*sequential/enc_conv4/Conv2D/ReadVariableOpReadVariableOp3sequential_enc_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*sequential/enc_conv4/Conv2D/ReadVariableOp?
sequential/enc_conv4/Conv2DConv2D'sequential/enc_conv3/Relu:activations:02sequential/enc_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential/enc_conv4/Conv2D?
+sequential/enc_conv4/BiasAdd/ReadVariableOpReadVariableOp4sequential_enc_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential/enc_conv4/BiasAdd/ReadVariableOp?
sequential/enc_conv4/BiasAddBiasAdd$sequential/enc_conv4/Conv2D:output:03sequential/enc_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential/enc_conv4/BiasAdd?
sequential/enc_conv4/ReluRelu%sequential/enc_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential/enc_conv4/Relu?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape'sequential/enc_conv4/Relu:activations:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/flatten/Reshape?
,sequential_1/enc_fc_mu/MatMul/ReadVariableOpReadVariableOp5sequential_1_enc_fc_mu_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02.
,sequential_1/enc_fc_mu/MatMul/ReadVariableOp?
sequential_1/enc_fc_mu/MatMulMatMul#sequential/flatten/Reshape:output:04sequential_1/enc_fc_mu/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_1/enc_fc_mu/MatMul?
-sequential_1/enc_fc_mu/BiasAdd/ReadVariableOpReadVariableOp6sequential_1_enc_fc_mu_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_1/enc_fc_mu/BiasAdd/ReadVariableOp?
sequential_1/enc_fc_mu/BiasAddBiasAdd'sequential_1/enc_fc_mu/MatMul:product:05sequential_1/enc_fc_mu/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2 
sequential_1/enc_fc_mu/BiasAdd?
1sequential_2/enc_fc_log_var/MatMul/ReadVariableOpReadVariableOp:sequential_2_enc_fc_log_var_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype023
1sequential_2/enc_fc_log_var/MatMul/ReadVariableOp?
"sequential_2/enc_fc_log_var/MatMulMatMul#sequential/flatten/Reshape:output:09sequential_2/enc_fc_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2$
"sequential_2/enc_fc_log_var/MatMul?
2sequential_2/enc_fc_log_var/BiasAdd/ReadVariableOpReadVariableOp;sequential_2_enc_fc_log_var_biasadd_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_2/enc_fc_log_var/BiasAdd/ReadVariableOp?
#sequential_2/enc_fc_log_var/BiasAddBiasAdd,sequential_2/enc_fc_log_var/MatMul:product:0:sequential_2/enc_fc_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2%
#sequential_2/enc_fc_log_var/BiasAdde
ShapeShape'sequential_1/enc_fc_mu/BiasAdd:output:0*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:????????? *
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:????????? 2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:????????? 2
random_normalS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/y?
mulMul,sequential_2/enc_fc_log_var/BiasAdd:output:0mul/y:output:0*
T0*'
_output_shapes
:????????? 2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:????????? 2
Expc
mul_1Mulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:????????? 2
mul_1y
addAddV2	mul_1:z:0'sequential_1/enc_fc_mu/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
add?
-sequential_3/dec_dense1/MatMul/ReadVariableOpReadVariableOp6sequential_3_dec_dense1_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02/
-sequential_3/dec_dense1/MatMul/ReadVariableOp?
sequential_3/dec_dense1/MatMulMatMuladd:z:05sequential_3/dec_dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_3/dec_dense1/MatMul?
.sequential_3/dec_dense1/BiasAdd/ReadVariableOpReadVariableOp7sequential_3_dec_dense1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.sequential_3/dec_dense1/BiasAdd/ReadVariableOp?
sequential_3/dec_dense1/BiasAddBiasAdd(sequential_3/dec_dense1/MatMul:product:06sequential_3/dec_dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
sequential_3/dec_dense1/BiasAdd?
sequential_3/dec_dense1/ReluRelu(sequential_3/dec_dense1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_3/dec_dense1/Relu?
sequential_3/reshape/ShapeShape*sequential_3/dec_dense1/Relu:activations:0*
T0*
_output_shapes
:2
sequential_3/reshape/Shape?
(sequential_3/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_3/reshape/strided_slice/stack?
*sequential_3/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_3/reshape/strided_slice/stack_1?
*sequential_3/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_3/reshape/strided_slice/stack_2?
"sequential_3/reshape/strided_sliceStridedSlice#sequential_3/reshape/Shape:output:01sequential_3/reshape/strided_slice/stack:output:03sequential_3/reshape/strided_slice/stack_1:output:03sequential_3/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_3/reshape/strided_slice?
$sequential_3/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_3/reshape/Reshape/shape/1?
$sequential_3/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_3/reshape/Reshape/shape/2?
$sequential_3/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2&
$sequential_3/reshape/Reshape/shape/3?
"sequential_3/reshape/Reshape/shapePack+sequential_3/reshape/strided_slice:output:0-sequential_3/reshape/Reshape/shape/1:output:0-sequential_3/reshape/Reshape/shape/2:output:0-sequential_3/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"sequential_3/reshape/Reshape/shape?
sequential_3/reshape/ReshapeReshape*sequential_3/dec_dense1/Relu:activations:0+sequential_3/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
sequential_3/reshape/Reshape?
sequential_3/dec_deconv1/ShapeShape%sequential_3/reshape/Reshape:output:0*
T0*
_output_shapes
:2 
sequential_3/dec_deconv1/Shape?
,sequential_3/dec_deconv1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_3/dec_deconv1/strided_slice/stack?
.sequential_3/dec_deconv1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv1/strided_slice/stack_1?
.sequential_3/dec_deconv1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv1/strided_slice/stack_2?
&sequential_3/dec_deconv1/strided_sliceStridedSlice'sequential_3/dec_deconv1/Shape:output:05sequential_3/dec_deconv1/strided_slice/stack:output:07sequential_3/dec_deconv1/strided_slice/stack_1:output:07sequential_3/dec_deconv1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_3/dec_deconv1/strided_slice?
.sequential_3/dec_deconv1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv1/strided_slice_1/stack?
0sequential_3/dec_deconv1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_1/stack_1?
0sequential_3/dec_deconv1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_1/stack_2?
(sequential_3/dec_deconv1/strided_slice_1StridedSlice'sequential_3/dec_deconv1/Shape:output:07sequential_3/dec_deconv1/strided_slice_1/stack:output:09sequential_3/dec_deconv1/strided_slice_1/stack_1:output:09sequential_3/dec_deconv1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv1/strided_slice_1?
.sequential_3/dec_deconv1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv1/strided_slice_2/stack?
0sequential_3/dec_deconv1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_2/stack_1?
0sequential_3/dec_deconv1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_2/stack_2?
(sequential_3/dec_deconv1/strided_slice_2StridedSlice'sequential_3/dec_deconv1/Shape:output:07sequential_3/dec_deconv1/strided_slice_2/stack:output:09sequential_3/dec_deconv1/strided_slice_2/stack_1:output:09sequential_3/dec_deconv1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv1/strided_slice_2?
sequential_3/dec_deconv1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv1/mul/y?
sequential_3/dec_deconv1/mulMul1sequential_3/dec_deconv1/strided_slice_1:output:0'sequential_3/dec_deconv1/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv1/mul?
sequential_3/dec_deconv1/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv1/add/y?
sequential_3/dec_deconv1/addAddV2 sequential_3/dec_deconv1/mul:z:0'sequential_3/dec_deconv1/add/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv1/add?
 sequential_3/dec_deconv1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv1/mul_1/y?
sequential_3/dec_deconv1/mul_1Mul1sequential_3/dec_deconv1/strided_slice_2:output:0)sequential_3/dec_deconv1/mul_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv1/mul_1?
 sequential_3/dec_deconv1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv1/add_1/y?
sequential_3/dec_deconv1/add_1AddV2"sequential_3/dec_deconv1/mul_1:z:0)sequential_3/dec_deconv1/add_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv1/add_1?
 sequential_3/dec_deconv1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2"
 sequential_3/dec_deconv1/stack/3?
sequential_3/dec_deconv1/stackPack/sequential_3/dec_deconv1/strided_slice:output:0 sequential_3/dec_deconv1/add:z:0"sequential_3/dec_deconv1/add_1:z:0)sequential_3/dec_deconv1/stack/3:output:0*
N*
T0*
_output_shapes
:2 
sequential_3/dec_deconv1/stack?
.sequential_3/dec_deconv1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_3/dec_deconv1/strided_slice_3/stack?
0sequential_3/dec_deconv1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_3/stack_1?
0sequential_3/dec_deconv1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_3/stack_2?
(sequential_3/dec_deconv1/strided_slice_3StridedSlice'sequential_3/dec_deconv1/stack:output:07sequential_3/dec_deconv1/strided_slice_3/stack:output:09sequential_3/dec_deconv1/strided_slice_3/stack_1:output:09sequential_3/dec_deconv1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv1/strided_slice_3?
8sequential_3/dec_deconv1/conv2d_transpose/ReadVariableOpReadVariableOpAsequential_3_dec_deconv1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02:
8sequential_3/dec_deconv1/conv2d_transpose/ReadVariableOp?
)sequential_3/dec_deconv1/conv2d_transposeConv2DBackpropInput'sequential_3/dec_deconv1/stack:output:0@sequential_3/dec_deconv1/conv2d_transpose/ReadVariableOp:value:0%sequential_3/reshape/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2+
)sequential_3/dec_deconv1/conv2d_transpose?
/sequential_3/dec_deconv1/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dec_deconv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/sequential_3/dec_deconv1/BiasAdd/ReadVariableOp?
 sequential_3/dec_deconv1/BiasAddBiasAdd2sequential_3/dec_deconv1/conv2d_transpose:output:07sequential_3/dec_deconv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2"
 sequential_3/dec_deconv1/BiasAdd?
sequential_3/dec_deconv1/ReluRelu)sequential_3/dec_deconv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential_3/dec_deconv1/Relu?
sequential_3/dec_deconv2/ShapeShape+sequential_3/dec_deconv1/Relu:activations:0*
T0*
_output_shapes
:2 
sequential_3/dec_deconv2/Shape?
,sequential_3/dec_deconv2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_3/dec_deconv2/strided_slice/stack?
.sequential_3/dec_deconv2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv2/strided_slice/stack_1?
.sequential_3/dec_deconv2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv2/strided_slice/stack_2?
&sequential_3/dec_deconv2/strided_sliceStridedSlice'sequential_3/dec_deconv2/Shape:output:05sequential_3/dec_deconv2/strided_slice/stack:output:07sequential_3/dec_deconv2/strided_slice/stack_1:output:07sequential_3/dec_deconv2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_3/dec_deconv2/strided_slice?
.sequential_3/dec_deconv2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv2/strided_slice_1/stack?
0sequential_3/dec_deconv2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_1/stack_1?
0sequential_3/dec_deconv2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_1/stack_2?
(sequential_3/dec_deconv2/strided_slice_1StridedSlice'sequential_3/dec_deconv2/Shape:output:07sequential_3/dec_deconv2/strided_slice_1/stack:output:09sequential_3/dec_deconv2/strided_slice_1/stack_1:output:09sequential_3/dec_deconv2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv2/strided_slice_1?
.sequential_3/dec_deconv2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv2/strided_slice_2/stack?
0sequential_3/dec_deconv2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_2/stack_1?
0sequential_3/dec_deconv2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_2/stack_2?
(sequential_3/dec_deconv2/strided_slice_2StridedSlice'sequential_3/dec_deconv2/Shape:output:07sequential_3/dec_deconv2/strided_slice_2/stack:output:09sequential_3/dec_deconv2/strided_slice_2/stack_1:output:09sequential_3/dec_deconv2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv2/strided_slice_2?
sequential_3/dec_deconv2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv2/mul/y?
sequential_3/dec_deconv2/mulMul1sequential_3/dec_deconv2/strided_slice_1:output:0'sequential_3/dec_deconv2/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv2/mul?
sequential_3/dec_deconv2/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv2/add/y?
sequential_3/dec_deconv2/addAddV2 sequential_3/dec_deconv2/mul:z:0'sequential_3/dec_deconv2/add/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv2/add?
 sequential_3/dec_deconv2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv2/mul_1/y?
sequential_3/dec_deconv2/mul_1Mul1sequential_3/dec_deconv2/strided_slice_2:output:0)sequential_3/dec_deconv2/mul_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv2/mul_1?
 sequential_3/dec_deconv2/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv2/add_1/y?
sequential_3/dec_deconv2/add_1AddV2"sequential_3/dec_deconv2/mul_1:z:0)sequential_3/dec_deconv2/add_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv2/add_1?
 sequential_3/dec_deconv2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2"
 sequential_3/dec_deconv2/stack/3?
sequential_3/dec_deconv2/stackPack/sequential_3/dec_deconv2/strided_slice:output:0 sequential_3/dec_deconv2/add:z:0"sequential_3/dec_deconv2/add_1:z:0)sequential_3/dec_deconv2/stack/3:output:0*
N*
T0*
_output_shapes
:2 
sequential_3/dec_deconv2/stack?
.sequential_3/dec_deconv2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_3/dec_deconv2/strided_slice_3/stack?
0sequential_3/dec_deconv2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_3/stack_1?
0sequential_3/dec_deconv2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_3/stack_2?
(sequential_3/dec_deconv2/strided_slice_3StridedSlice'sequential_3/dec_deconv2/stack:output:07sequential_3/dec_deconv2/strided_slice_3/stack:output:09sequential_3/dec_deconv2/strided_slice_3/stack_1:output:09sequential_3/dec_deconv2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv2/strided_slice_3?
8sequential_3/dec_deconv2/conv2d_transpose/ReadVariableOpReadVariableOpAsequential_3_dec_deconv2_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02:
8sequential_3/dec_deconv2/conv2d_transpose/ReadVariableOp?
)sequential_3/dec_deconv2/conv2d_transposeConv2DBackpropInput'sequential_3/dec_deconv2/stack:output:0@sequential_3/dec_deconv2/conv2d_transpose/ReadVariableOp:value:0+sequential_3/dec_deconv1/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2+
)sequential_3/dec_deconv2/conv2d_transpose?
/sequential_3/dec_deconv2/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dec_deconv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/sequential_3/dec_deconv2/BiasAdd/ReadVariableOp?
 sequential_3/dec_deconv2/BiasAddBiasAdd2sequential_3/dec_deconv2/conv2d_transpose:output:07sequential_3/dec_deconv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2"
 sequential_3/dec_deconv2/BiasAdd?
sequential_3/dec_deconv2/ReluRelu)sequential_3/dec_deconv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_3/dec_deconv2/Relu?
sequential_3/dec_deconv3/ShapeShape+sequential_3/dec_deconv2/Relu:activations:0*
T0*
_output_shapes
:2 
sequential_3/dec_deconv3/Shape?
,sequential_3/dec_deconv3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_3/dec_deconv3/strided_slice/stack?
.sequential_3/dec_deconv3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv3/strided_slice/stack_1?
.sequential_3/dec_deconv3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv3/strided_slice/stack_2?
&sequential_3/dec_deconv3/strided_sliceStridedSlice'sequential_3/dec_deconv3/Shape:output:05sequential_3/dec_deconv3/strided_slice/stack:output:07sequential_3/dec_deconv3/strided_slice/stack_1:output:07sequential_3/dec_deconv3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_3/dec_deconv3/strided_slice?
.sequential_3/dec_deconv3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv3/strided_slice_1/stack?
0sequential_3/dec_deconv3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_1/stack_1?
0sequential_3/dec_deconv3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_1/stack_2?
(sequential_3/dec_deconv3/strided_slice_1StridedSlice'sequential_3/dec_deconv3/Shape:output:07sequential_3/dec_deconv3/strided_slice_1/stack:output:09sequential_3/dec_deconv3/strided_slice_1/stack_1:output:09sequential_3/dec_deconv3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv3/strided_slice_1?
.sequential_3/dec_deconv3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv3/strided_slice_2/stack?
0sequential_3/dec_deconv3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_2/stack_1?
0sequential_3/dec_deconv3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_2/stack_2?
(sequential_3/dec_deconv3/strided_slice_2StridedSlice'sequential_3/dec_deconv3/Shape:output:07sequential_3/dec_deconv3/strided_slice_2/stack:output:09sequential_3/dec_deconv3/strided_slice_2/stack_1:output:09sequential_3/dec_deconv3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv3/strided_slice_2?
sequential_3/dec_deconv3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv3/mul/y?
sequential_3/dec_deconv3/mulMul1sequential_3/dec_deconv3/strided_slice_1:output:0'sequential_3/dec_deconv3/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv3/mul?
sequential_3/dec_deconv3/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv3/add/y?
sequential_3/dec_deconv3/addAddV2 sequential_3/dec_deconv3/mul:z:0'sequential_3/dec_deconv3/add/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv3/add?
 sequential_3/dec_deconv3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv3/mul_1/y?
sequential_3/dec_deconv3/mul_1Mul1sequential_3/dec_deconv3/strided_slice_2:output:0)sequential_3/dec_deconv3/mul_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv3/mul_1?
 sequential_3/dec_deconv3/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv3/add_1/y?
sequential_3/dec_deconv3/add_1AddV2"sequential_3/dec_deconv3/mul_1:z:0)sequential_3/dec_deconv3/add_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv3/add_1?
 sequential_3/dec_deconv3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2"
 sequential_3/dec_deconv3/stack/3?
sequential_3/dec_deconv3/stackPack/sequential_3/dec_deconv3/strided_slice:output:0 sequential_3/dec_deconv3/add:z:0"sequential_3/dec_deconv3/add_1:z:0)sequential_3/dec_deconv3/stack/3:output:0*
N*
T0*
_output_shapes
:2 
sequential_3/dec_deconv3/stack?
.sequential_3/dec_deconv3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_3/dec_deconv3/strided_slice_3/stack?
0sequential_3/dec_deconv3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_3/stack_1?
0sequential_3/dec_deconv3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_3/stack_2?
(sequential_3/dec_deconv3/strided_slice_3StridedSlice'sequential_3/dec_deconv3/stack:output:07sequential_3/dec_deconv3/strided_slice_3/stack:output:09sequential_3/dec_deconv3/strided_slice_3/stack_1:output:09sequential_3/dec_deconv3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv3/strided_slice_3?
8sequential_3/dec_deconv3/conv2d_transpose/ReadVariableOpReadVariableOpAsequential_3_dec_deconv3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02:
8sequential_3/dec_deconv3/conv2d_transpose/ReadVariableOp?
)sequential_3/dec_deconv3/conv2d_transposeConv2DBackpropInput'sequential_3/dec_deconv3/stack:output:0@sequential_3/dec_deconv3/conv2d_transpose/ReadVariableOp:value:0+sequential_3/dec_deconv2/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2+
)sequential_3/dec_deconv3/conv2d_transpose?
/sequential_3/dec_deconv3/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dec_deconv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential_3/dec_deconv3/BiasAdd/ReadVariableOp?
 sequential_3/dec_deconv3/BiasAddBiasAdd2sequential_3/dec_deconv3/conv2d_transpose:output:07sequential_3/dec_deconv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2"
 sequential_3/dec_deconv3/BiasAdd?
sequential_3/dec_deconv3/ReluRelu)sequential_3/dec_deconv3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_3/dec_deconv3/Relu?
sequential_3/dec_deconv4/ShapeShape+sequential_3/dec_deconv3/Relu:activations:0*
T0*
_output_shapes
:2 
sequential_3/dec_deconv4/Shape?
,sequential_3/dec_deconv4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_3/dec_deconv4/strided_slice/stack?
.sequential_3/dec_deconv4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv4/strided_slice/stack_1?
.sequential_3/dec_deconv4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv4/strided_slice/stack_2?
&sequential_3/dec_deconv4/strided_sliceStridedSlice'sequential_3/dec_deconv4/Shape:output:05sequential_3/dec_deconv4/strided_slice/stack:output:07sequential_3/dec_deconv4/strided_slice/stack_1:output:07sequential_3/dec_deconv4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_3/dec_deconv4/strided_slice?
.sequential_3/dec_deconv4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv4/strided_slice_1/stack?
0sequential_3/dec_deconv4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_1/stack_1?
0sequential_3/dec_deconv4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_1/stack_2?
(sequential_3/dec_deconv4/strided_slice_1StridedSlice'sequential_3/dec_deconv4/Shape:output:07sequential_3/dec_deconv4/strided_slice_1/stack:output:09sequential_3/dec_deconv4/strided_slice_1/stack_1:output:09sequential_3/dec_deconv4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv4/strided_slice_1?
.sequential_3/dec_deconv4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv4/strided_slice_2/stack?
0sequential_3/dec_deconv4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_2/stack_1?
0sequential_3/dec_deconv4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_2/stack_2?
(sequential_3/dec_deconv4/strided_slice_2StridedSlice'sequential_3/dec_deconv4/Shape:output:07sequential_3/dec_deconv4/strided_slice_2/stack:output:09sequential_3/dec_deconv4/strided_slice_2/stack_1:output:09sequential_3/dec_deconv4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv4/strided_slice_2?
sequential_3/dec_deconv4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv4/mul/y?
sequential_3/dec_deconv4/mulMul1sequential_3/dec_deconv4/strided_slice_1:output:0'sequential_3/dec_deconv4/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv4/mul?
sequential_3/dec_deconv4/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv4/add/y?
sequential_3/dec_deconv4/addAddV2 sequential_3/dec_deconv4/mul:z:0'sequential_3/dec_deconv4/add/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv4/add?
 sequential_3/dec_deconv4/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv4/mul_1/y?
sequential_3/dec_deconv4/mul_1Mul1sequential_3/dec_deconv4/strided_slice_2:output:0)sequential_3/dec_deconv4/mul_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv4/mul_1?
 sequential_3/dec_deconv4/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv4/add_1/y?
sequential_3/dec_deconv4/add_1AddV2"sequential_3/dec_deconv4/mul_1:z:0)sequential_3/dec_deconv4/add_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv4/add_1?
 sequential_3/dec_deconv4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv4/stack/3?
sequential_3/dec_deconv4/stackPack/sequential_3/dec_deconv4/strided_slice:output:0 sequential_3/dec_deconv4/add:z:0"sequential_3/dec_deconv4/add_1:z:0)sequential_3/dec_deconv4/stack/3:output:0*
N*
T0*
_output_shapes
:2 
sequential_3/dec_deconv4/stack?
.sequential_3/dec_deconv4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_3/dec_deconv4/strided_slice_3/stack?
0sequential_3/dec_deconv4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_3/stack_1?
0sequential_3/dec_deconv4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_3/stack_2?
(sequential_3/dec_deconv4/strided_slice_3StridedSlice'sequential_3/dec_deconv4/stack:output:07sequential_3/dec_deconv4/strided_slice_3/stack:output:09sequential_3/dec_deconv4/strided_slice_3/stack_1:output:09sequential_3/dec_deconv4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv4/strided_slice_3?
8sequential_3/dec_deconv4/conv2d_transpose/ReadVariableOpReadVariableOpAsequential_3_dec_deconv4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02:
8sequential_3/dec_deconv4/conv2d_transpose/ReadVariableOp?
)sequential_3/dec_deconv4/conv2d_transposeConv2DBackpropInput'sequential_3/dec_deconv4/stack:output:0@sequential_3/dec_deconv4/conv2d_transpose/ReadVariableOp:value:0+sequential_3/dec_deconv3/Relu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingVALID*
strides
2+
)sequential_3/dec_deconv4/conv2d_transpose?
/sequential_3/dec_deconv4/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dec_deconv4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_3/dec_deconv4/BiasAdd/ReadVariableOp?
 sequential_3/dec_deconv4/BiasAddBiasAdd2sequential_3/dec_deconv4/conv2d_transpose:output:07sequential_3/dec_deconv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2"
 sequential_3/dec_deconv4/BiasAdd?
 sequential_3/dec_deconv4/SigmoidSigmoid)sequential_3/dec_deconv4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2"
 sequential_3/dec_deconv4/Sigmoide
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2'sequential_1/enc_fc_mu/BiasAdd:output:0,sequential_2/enc_fc_log_var/BiasAdd:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity$sequential_3/dec_deconv4/Sigmoid:y:0*
T0*/
_output_shapes
:?????????@@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesu
s:?????????@@:::::::::::::::::::::::X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: 
?
?
,__inference_dec_deconv2_layer_call_fn_877763

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dec_deconv2_layer_call_and_return_conditional_losses_8777532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_dec_deconv3_layer_call_fn_877812

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dec_deconv3_layer_call_and_return_conditional_losses_8778022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?	
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_879819

inputs,
(enc_fc_mu_matmul_readvariableop_resource-
)enc_fc_mu_biasadd_readvariableop_resource
identity??
enc_fc_mu/MatMul/ReadVariableOpReadVariableOp(enc_fc_mu_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02!
enc_fc_mu/MatMul/ReadVariableOp?
enc_fc_mu/MatMulMatMulinputs'enc_fc_mu/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
enc_fc_mu/MatMul?
 enc_fc_mu/BiasAdd/ReadVariableOpReadVariableOp)enc_fc_mu_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 enc_fc_mu/BiasAdd/ReadVariableOp?
enc_fc_mu/BiasAddBiasAddenc_fc_mu/MatMul:product:0(enc_fc_mu/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
enc_fc_mu/BiasAddn
IdentityIdentityenc_fc_mu/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
H__inference_sequential_2_layer_call_and_return_conditional_losses_877640

inputs
enc_fc_log_var_877634
enc_fc_log_var_877636
identity??&enc_fc_log_var/StatefulPartitionedCall?
&enc_fc_log_var/StatefulPartitionedCallStatefulPartitionedCallinputsenc_fc_log_var_877634enc_fc_log_var_877636*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_enc_fc_log_var_layer_call_and_return_conditional_losses_8776022(
&enc_fc_log_var/StatefulPartitionedCall?
IdentityIdentity/enc_fc_log_var/StatefulPartitionedCall:output:0'^enc_fc_log_var/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2P
&enc_fc_log_var/StatefulPartitionedCall&enc_fc_log_var/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
E__inference_enc_fc_mu_layer_call_and_return_conditional_losses_877525

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_dec_deconv4_layer_call_fn_877861

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dec_deconv4_layer_call_and_return_conditional_losses_8778512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_877542
input_2
enc_fc_mu_877536
enc_fc_mu_877538
identity??!enc_fc_mu/StatefulPartitionedCall?
!enc_fc_mu/StatefulPartitionedCallStatefulPartitionedCallinput_2enc_fc_mu_877536enc_fc_mu_877538*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_fc_mu_layer_call_and_return_conditional_losses_8775252#
!enc_fc_mu/StatefulPartitionedCall?
IdentityIdentity*enc_fc_mu/StatefulPartitionedCall:output:0"^enc_fc_mu/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2F
!enc_fc_mu/StatefulPartitionedCall!enc_fc_mu/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_gan_vae_layer_call_fn_879638
input_1
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

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*T
_output_shapesB
@:?????????@:+???????????????????????????*8
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_gan_vae_layer_call_and_return_conditional_losses_8784062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesu
s:?????????@@::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: 
?
?
F__inference_dec_dense1_layer_call_and_return_conditional_losses_877876

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :::O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
-__inference_sequential_1_layer_call_fn_879837

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_8775812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?

*__inference_enc_conv3_layer_call_fn_877332

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_conv3_layer_call_and_return_conditional_losses_8773222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
-__inference_sequential_3_layer_call_fn_880239

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
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_8780532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:????????? ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
??
?)
"__inference__traced_restore_880829
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate'
#assignvariableop_5_enc_conv1_kernel%
!assignvariableop_6_enc_conv1_bias'
#assignvariableop_7_enc_conv2_kernel%
!assignvariableop_8_enc_conv2_bias'
#assignvariableop_9_enc_conv3_kernel&
"assignvariableop_10_enc_conv3_bias(
$assignvariableop_11_enc_conv4_kernel&
"assignvariableop_12_enc_conv4_bias(
$assignvariableop_13_enc_fc_mu_kernel&
"assignvariableop_14_enc_fc_mu_bias-
)assignvariableop_15_enc_fc_log_var_kernel+
'assignvariableop_16_enc_fc_log_var_bias)
%assignvariableop_17_dec_dense1_kernel'
#assignvariableop_18_dec_dense1_bias*
&assignvariableop_19_dec_deconv1_kernel(
$assignvariableop_20_dec_deconv1_bias*
&assignvariableop_21_dec_deconv2_kernel(
$assignvariableop_22_dec_deconv2_bias*
&assignvariableop_23_dec_deconv3_kernel(
$assignvariableop_24_dec_deconv3_bias*
&assignvariableop_25_dec_deconv4_kernel(
$assignvariableop_26_dec_deconv4_bias
assignvariableop_27_total
assignvariableop_28_count
assignvariableop_29_total_1
assignvariableop_30_count_1
assignvariableop_31_total_2
assignvariableop_32_count_2/
+assignvariableop_33_adam_enc_conv1_kernel_m-
)assignvariableop_34_adam_enc_conv1_bias_m/
+assignvariableop_35_adam_enc_conv2_kernel_m-
)assignvariableop_36_adam_enc_conv2_bias_m/
+assignvariableop_37_adam_enc_conv3_kernel_m-
)assignvariableop_38_adam_enc_conv3_bias_m/
+assignvariableop_39_adam_enc_conv4_kernel_m-
)assignvariableop_40_adam_enc_conv4_bias_m/
+assignvariableop_41_adam_enc_fc_mu_kernel_m-
)assignvariableop_42_adam_enc_fc_mu_bias_m4
0assignvariableop_43_adam_enc_fc_log_var_kernel_m2
.assignvariableop_44_adam_enc_fc_log_var_bias_m0
,assignvariableop_45_adam_dec_dense1_kernel_m.
*assignvariableop_46_adam_dec_dense1_bias_m1
-assignvariableop_47_adam_dec_deconv1_kernel_m/
+assignvariableop_48_adam_dec_deconv1_bias_m1
-assignvariableop_49_adam_dec_deconv2_kernel_m/
+assignvariableop_50_adam_dec_deconv2_bias_m1
-assignvariableop_51_adam_dec_deconv3_kernel_m/
+assignvariableop_52_adam_dec_deconv3_bias_m1
-assignvariableop_53_adam_dec_deconv4_kernel_m/
+assignvariableop_54_adam_dec_deconv4_bias_m/
+assignvariableop_55_adam_enc_conv1_kernel_v-
)assignvariableop_56_adam_enc_conv1_bias_v/
+assignvariableop_57_adam_enc_conv2_kernel_v-
)assignvariableop_58_adam_enc_conv2_bias_v/
+assignvariableop_59_adam_enc_conv3_kernel_v-
)assignvariableop_60_adam_enc_conv3_bias_v/
+assignvariableop_61_adam_enc_conv4_kernel_v-
)assignvariableop_62_adam_enc_conv4_bias_v/
+assignvariableop_63_adam_enc_fc_mu_kernel_v-
)assignvariableop_64_adam_enc_fc_mu_bias_v4
0assignvariableop_65_adam_enc_fc_log_var_kernel_v2
.assignvariableop_66_adam_enc_fc_log_var_bias_v0
,assignvariableop_67_adam_dec_dense1_kernel_v.
*assignvariableop_68_adam_dec_dense1_bias_v1
-assignvariableop_69_adam_dec_deconv1_kernel_v/
+assignvariableop_70_adam_dec_deconv1_bias_v1
-assignvariableop_71_adam_dec_deconv2_kernel_v/
+assignvariableop_72_adam_dec_deconv2_bias_v1
-assignvariableop_73_adam_dec_deconv3_kernel_v/
+assignvariableop_74_adam_dec_deconv3_bias_v1
-assignvariableop_75_adam_dec_deconv4_kernel_v/
+assignvariableop_76_adam_dec_deconv4_bias_v
identity_78??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?)
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:M*
dtype0*?(
value?(B?(MB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:M*
dtype0*?
value?B?MB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*[
dtypesQ
O2M	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_enc_conv1_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_enc_conv1_biasIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_enc_conv2_kernelIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_enc_conv2_biasIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_enc_conv3_kernelIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_enc_conv3_biasIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp$assignvariableop_11_enc_conv4_kernelIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_enc_conv4_biasIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp$assignvariableop_13_enc_fc_mu_kernelIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_enc_fc_mu_biasIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp)assignvariableop_15_enc_fc_log_var_kernelIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_enc_fc_log_var_biasIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp%assignvariableop_17_dec_dense1_kernelIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dec_dense1_biasIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp&assignvariableop_19_dec_deconv1_kernelIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp$assignvariableop_20_dec_deconv1_biasIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp&assignvariableop_21_dec_deconv2_kernelIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp$assignvariableop_22_dec_deconv2_biasIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp&assignvariableop_23_dec_deconv3_kernelIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp$assignvariableop_24_dec_deconv3_biasIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp&assignvariableop_25_dec_deconv4_kernelIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp$assignvariableop_26_dec_deconv4_biasIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_1Identity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_1Identity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_total_2Identity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_count_2Identity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_enc_conv1_kernel_mIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_enc_conv1_bias_mIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_enc_conv2_kernel_mIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_enc_conv2_bias_mIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_enc_conv3_kernel_mIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_enc_conv3_bias_mIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_enc_conv4_kernel_mIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_enc_conv4_bias_mIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_enc_fc_mu_kernel_mIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_enc_fc_mu_bias_mIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp0assignvariableop_43_adam_enc_fc_log_var_kernel_mIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp.assignvariableop_44_adam_enc_fc_log_var_bias_mIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp,assignvariableop_45_adam_dec_dense1_kernel_mIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_dec_dense1_bias_mIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp-assignvariableop_47_adam_dec_deconv1_kernel_mIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp+assignvariableop_48_adam_dec_deconv1_bias_mIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp-assignvariableop_49_adam_dec_deconv2_kernel_mIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp+assignvariableop_50_adam_dec_deconv2_bias_mIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp-assignvariableop_51_adam_dec_deconv3_kernel_mIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp+assignvariableop_52_adam_dec_deconv3_bias_mIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp-assignvariableop_53_adam_dec_deconv4_kernel_mIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp+assignvariableop_54_adam_dec_deconv4_bias_mIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_enc_conv1_kernel_vIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_enc_conv1_bias_vIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_enc_conv2_kernel_vIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_enc_conv2_bias_vIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp+assignvariableop_59_adam_enc_conv3_kernel_vIdentity_59:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_enc_conv3_bias_vIdentity_60:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_60_
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp+assignvariableop_61_adam_enc_conv4_kernel_vIdentity_61:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_enc_conv4_bias_vIdentity_62:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_62_
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_enc_fc_mu_kernel_vIdentity_63:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_63_
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_enc_fc_mu_bias_vIdentity_64:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_64_
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp0assignvariableop_65_adam_enc_fc_log_var_kernel_vIdentity_65:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_65_
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp.assignvariableop_66_adam_enc_fc_log_var_bias_vIdentity_66:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_66_
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp,assignvariableop_67_adam_dec_dense1_kernel_vIdentity_67:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_67_
Identity_68IdentityRestoreV2:tensors:68*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp*assignvariableop_68_adam_dec_dense1_bias_vIdentity_68:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_68_
Identity_69IdentityRestoreV2:tensors:69*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp-assignvariableop_69_adam_dec_deconv1_kernel_vIdentity_69:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_69_
Identity_70IdentityRestoreV2:tensors:70*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp+assignvariableop_70_adam_dec_deconv1_bias_vIdentity_70:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_70_
Identity_71IdentityRestoreV2:tensors:71*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp-assignvariableop_71_adam_dec_deconv2_kernel_vIdentity_71:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_71_
Identity_72IdentityRestoreV2:tensors:72*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp+assignvariableop_72_adam_dec_deconv2_bias_vIdentity_72:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_72_
Identity_73IdentityRestoreV2:tensors:73*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp-assignvariableop_73_adam_dec_deconv3_kernel_vIdentity_73:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_73_
Identity_74IdentityRestoreV2:tensors:74*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp+assignvariableop_74_adam_dec_deconv3_bias_vIdentity_74:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_74_
Identity_75IdentityRestoreV2:tensors:75*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp-assignvariableop_75_adam_dec_deconv4_kernel_vIdentity_75:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_75_
Identity_76IdentityRestoreV2:tensors:76*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp+assignvariableop_76_adam_dec_deconv4_bias_vIdentity_76:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_76?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_77Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_77?
Identity_78IdentityIdentity_77:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_78"#
identity_78Identity_78:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :@

_output_shapes
: :A

_output_shapes
: :B

_output_shapes
: :C

_output_shapes
: :D

_output_shapes
: :E

_output_shapes
: :F

_output_shapes
: :G

_output_shapes
: :H

_output_shapes
: :I

_output_shapes
: :J

_output_shapes
: :K

_output_shapes
: :L

_output_shapes
: :M

_output_shapes
: 
?

?
E__inference_enc_conv4_layer_call_and_return_conditional_losses_877344

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????:::j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?&
?
G__inference_dec_deconv1_layer_call_and_return_conditional_losses_877704

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????:::j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?%
?
G__inference_dec_deconv4_layer_call_and_return_conditional_losses_877851

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoidy
IdentityIdentitySigmoid:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? :::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
-__inference_sequential_1_layer_call_fn_877588
input_2
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_8775812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: 
?1
?
C__inference_gan_vae_layer_call_and_return_conditional_losses_878406

inputs
sequential_878343
sequential_878345
sequential_878347
sequential_878349
sequential_878351
sequential_878353
sequential_878355
sequential_878357
sequential_1_878360
sequential_1_878362
sequential_2_878365
sequential_2_878367
sequential_3_878381
sequential_3_878383
sequential_3_878385
sequential_3_878387
sequential_3_878389
sequential_3_878391
sequential_3_878393
sequential_3_878395
sequential_3_878397
sequential_3_878399
identity

identity_1??"sequential/StatefulPartitionedCall?$sequential_1/StatefulPartitionedCall?$sequential_2/StatefulPartitionedCall?$sequential_3/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_878343sequential_878345sequential_878347sequential_878349sequential_878351sequential_878353sequential_878355sequential_878357*
Tin
2	*
Tout
2*(
_output_shapes
:??????????**
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_8774462$
"sequential/StatefulPartitionedCall?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_878360sequential_1_878362*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_8775632&
$sequential_1/StatefulPartitionedCall?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_2_878365sequential_2_878367*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_8776402&
$sequential_2/StatefulPartitionedCallk
ShapeShape-sequential_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:????????? *
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:????????? 2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:????????? 2
random_normalS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/y?
mulMul-sequential_2/StatefulPartitionedCall:output:0mul/y:output:0*
T0*'
_output_shapes
:????????? 2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:????????? 2
Expc
mul_1Mulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:????????? 2
mul_1
addAddV2	mul_1:z:0-sequential_1/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:????????? 2
add?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCalladd:z:0sequential_3_878381sequential_3_878383sequential_3_878385sequential_3_878387sequential_3_878389sequential_3_878391sequential_3_878393sequential_3_878395sequential_3_878397sequential_3_878399*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_8779982&
$sequential_3/StatefulPartitionedCalle
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2-sequential_1/StatefulPartitionedCall:output:0-sequential_2/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@2
concat?
IdentityIdentityconcat:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity-sequential_3/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesu
s:?????????@@::::::::::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: 
?!
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_877998

inputs
dec_dense1_877971
dec_dense1_877973
dec_deconv1_877977
dec_deconv1_877979
dec_deconv2_877982
dec_deconv2_877984
dec_deconv3_877987
dec_deconv3_877989
dec_deconv4_877992
dec_deconv4_877994
identity??#dec_deconv1/StatefulPartitionedCall?#dec_deconv2/StatefulPartitionedCall?#dec_deconv3/StatefulPartitionedCall?#dec_deconv4/StatefulPartitionedCall?"dec_dense1/StatefulPartitionedCall?
"dec_dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdec_dense1_877971dec_dense1_877973*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_dec_dense1_layer_call_and_return_conditional_losses_8778762$
"dec_dense1/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall+dec_dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_8779062
reshape/PartitionedCall?
#dec_deconv1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0dec_deconv1_877977dec_deconv1_877979*
Tin
2*
Tout
2*B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dec_deconv1_layer_call_and_return_conditional_losses_8777042%
#dec_deconv1/StatefulPartitionedCall?
#dec_deconv2/StatefulPartitionedCallStatefulPartitionedCall,dec_deconv1/StatefulPartitionedCall:output:0dec_deconv2_877982dec_deconv2_877984*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dec_deconv2_layer_call_and_return_conditional_losses_8777532%
#dec_deconv2/StatefulPartitionedCall?
#dec_deconv3/StatefulPartitionedCallStatefulPartitionedCall,dec_deconv2/StatefulPartitionedCall:output:0dec_deconv3_877987dec_deconv3_877989*
Tin
2*
Tout
2*A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dec_deconv3_layer_call_and_return_conditional_losses_8778022%
#dec_deconv3/StatefulPartitionedCall?
#dec_deconv4/StatefulPartitionedCallStatefulPartitionedCall,dec_deconv3/StatefulPartitionedCall:output:0dec_deconv4_877992dec_deconv4_877994*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dec_deconv4_layer_call_and_return_conditional_losses_8778512%
#dec_deconv4/StatefulPartitionedCall?
IdentityIdentity,dec_deconv4/StatefulPartitionedCall:output:0$^dec_deconv1/StatefulPartitionedCall$^dec_deconv2/StatefulPartitionedCall$^dec_deconv3/StatefulPartitionedCall$^dec_deconv4/StatefulPartitionedCall#^dec_dense1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:????????? ::::::::::2J
#dec_deconv1/StatefulPartitionedCall#dec_deconv1/StatefulPartitionedCall2J
#dec_deconv2/StatefulPartitionedCall#dec_deconv2/StatefulPartitionedCall2J
#dec_deconv3/StatefulPartitionedCall#dec_deconv3/StatefulPartitionedCall2J
#dec_deconv4/StatefulPartitionedCall#dec_deconv4/StatefulPartitionedCall2H
"dec_dense1/StatefulPartitionedCall"dec_dense1/StatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?%
?
F__inference_sequential_layer_call_and_return_conditional_losses_879757

inputs,
(enc_conv1_conv2d_readvariableop_resource-
)enc_conv1_biasadd_readvariableop_resource,
(enc_conv2_conv2d_readvariableop_resource-
)enc_conv2_biasadd_readvariableop_resource,
(enc_conv3_conv2d_readvariableop_resource-
)enc_conv3_biasadd_readvariableop_resource,
(enc_conv4_conv2d_readvariableop_resource-
)enc_conv4_biasadd_readvariableop_resource
identity??
enc_conv1/Conv2D/ReadVariableOpReadVariableOp(enc_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
enc_conv1/Conv2D/ReadVariableOp?
enc_conv1/Conv2DConv2Dinputs'enc_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
enc_conv1/Conv2D?
 enc_conv1/BiasAdd/ReadVariableOpReadVariableOp)enc_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 enc_conv1/BiasAdd/ReadVariableOp?
enc_conv1/BiasAddBiasAddenc_conv1/Conv2D:output:0(enc_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
enc_conv1/BiasAdd~
enc_conv1/ReluReluenc_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
enc_conv1/Relu?
enc_conv2/Conv2D/ReadVariableOpReadVariableOp(enc_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
enc_conv2/Conv2D/ReadVariableOp?
enc_conv2/Conv2DConv2Denc_conv1/Relu:activations:0'enc_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
enc_conv2/Conv2D?
 enc_conv2/BiasAdd/ReadVariableOpReadVariableOp)enc_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 enc_conv2/BiasAdd/ReadVariableOp?
enc_conv2/BiasAddBiasAddenc_conv2/Conv2D:output:0(enc_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
enc_conv2/BiasAdd~
enc_conv2/ReluReluenc_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
enc_conv2/Relu?
enc_conv3/Conv2D/ReadVariableOpReadVariableOp(enc_conv3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
enc_conv3/Conv2D/ReadVariableOp?
enc_conv3/Conv2DConv2Denc_conv2/Relu:activations:0'enc_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
enc_conv3/Conv2D?
 enc_conv3/BiasAdd/ReadVariableOpReadVariableOp)enc_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 enc_conv3/BiasAdd/ReadVariableOp?
enc_conv3/BiasAddBiasAddenc_conv3/Conv2D:output:0(enc_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
enc_conv3/BiasAdd
enc_conv3/ReluReluenc_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
enc_conv3/Relu?
enc_conv4/Conv2D/ReadVariableOpReadVariableOp(enc_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
enc_conv4/Conv2D/ReadVariableOp?
enc_conv4/Conv2DConv2Denc_conv3/Relu:activations:0'enc_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
enc_conv4/Conv2D?
 enc_conv4/BiasAdd/ReadVariableOpReadVariableOp)enc_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 enc_conv4/BiasAdd/ReadVariableOp?
enc_conv4/BiasAddBiasAddenc_conv4/Conv2D:output:0(enc_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
enc_conv4/BiasAdd
enc_conv4/ReluReluenc_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
enc_conv4/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshapeenc_conv4/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshapem
IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????@@:::::::::W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_dec_deconv1_layer_call_fn_877714

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dec_deconv1_layer_call_and_return_conditional_losses_8777042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?

?
E__inference_enc_conv1_layer_call_and_return_conditional_losses_877278

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????:::i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
??
?
C__inference_gan_vae_layer_call_and_return_conditional_losses_879059

inputs7
3sequential_enc_conv1_conv2d_readvariableop_resource8
4sequential_enc_conv1_biasadd_readvariableop_resource7
3sequential_enc_conv2_conv2d_readvariableop_resource8
4sequential_enc_conv2_biasadd_readvariableop_resource7
3sequential_enc_conv3_conv2d_readvariableop_resource8
4sequential_enc_conv3_biasadd_readvariableop_resource7
3sequential_enc_conv4_conv2d_readvariableop_resource8
4sequential_enc_conv4_biasadd_readvariableop_resource9
5sequential_1_enc_fc_mu_matmul_readvariableop_resource:
6sequential_1_enc_fc_mu_biasadd_readvariableop_resource>
:sequential_2_enc_fc_log_var_matmul_readvariableop_resource?
;sequential_2_enc_fc_log_var_biasadd_readvariableop_resource:
6sequential_3_dec_dense1_matmul_readvariableop_resource;
7sequential_3_dec_dense1_biasadd_readvariableop_resourceE
Asequential_3_dec_deconv1_conv2d_transpose_readvariableop_resource<
8sequential_3_dec_deconv1_biasadd_readvariableop_resourceE
Asequential_3_dec_deconv2_conv2d_transpose_readvariableop_resource<
8sequential_3_dec_deconv2_biasadd_readvariableop_resourceE
Asequential_3_dec_deconv3_conv2d_transpose_readvariableop_resource<
8sequential_3_dec_deconv3_biasadd_readvariableop_resourceE
Asequential_3_dec_deconv4_conv2d_transpose_readvariableop_resource<
8sequential_3_dec_deconv4_biasadd_readvariableop_resource
identity

identity_1??
*sequential/enc_conv1/Conv2D/ReadVariableOpReadVariableOp3sequential_enc_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*sequential/enc_conv1/Conv2D/ReadVariableOp?
sequential/enc_conv1/Conv2DConv2Dinputs2sequential/enc_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
sequential/enc_conv1/Conv2D?
+sequential/enc_conv1/BiasAdd/ReadVariableOpReadVariableOp4sequential_enc_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential/enc_conv1/BiasAdd/ReadVariableOp?
sequential/enc_conv1/BiasAddBiasAdd$sequential/enc_conv1/Conv2D:output:03sequential/enc_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential/enc_conv1/BiasAdd?
sequential/enc_conv1/ReluRelu%sequential/enc_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential/enc_conv1/Relu?
*sequential/enc_conv2/Conv2D/ReadVariableOpReadVariableOp3sequential_enc_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*sequential/enc_conv2/Conv2D/ReadVariableOp?
sequential/enc_conv2/Conv2DConv2D'sequential/enc_conv1/Relu:activations:02sequential/enc_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
sequential/enc_conv2/Conv2D?
+sequential/enc_conv2/BiasAdd/ReadVariableOpReadVariableOp4sequential_enc_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential/enc_conv2/BiasAdd/ReadVariableOp?
sequential/enc_conv2/BiasAddBiasAdd$sequential/enc_conv2/Conv2D:output:03sequential/enc_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential/enc_conv2/BiasAdd?
sequential/enc_conv2/ReluRelu%sequential/enc_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential/enc_conv2/Relu?
*sequential/enc_conv3/Conv2D/ReadVariableOpReadVariableOp3sequential_enc_conv3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02,
*sequential/enc_conv3/Conv2D/ReadVariableOp?
sequential/enc_conv3/Conv2DConv2D'sequential/enc_conv2/Relu:activations:02sequential/enc_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential/enc_conv3/Conv2D?
+sequential/enc_conv3/BiasAdd/ReadVariableOpReadVariableOp4sequential_enc_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential/enc_conv3/BiasAdd/ReadVariableOp?
sequential/enc_conv3/BiasAddBiasAdd$sequential/enc_conv3/Conv2D:output:03sequential/enc_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential/enc_conv3/BiasAdd?
sequential/enc_conv3/ReluRelu%sequential/enc_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential/enc_conv3/Relu?
*sequential/enc_conv4/Conv2D/ReadVariableOpReadVariableOp3sequential_enc_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*sequential/enc_conv4/Conv2D/ReadVariableOp?
sequential/enc_conv4/Conv2DConv2D'sequential/enc_conv3/Relu:activations:02sequential/enc_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential/enc_conv4/Conv2D?
+sequential/enc_conv4/BiasAdd/ReadVariableOpReadVariableOp4sequential_enc_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential/enc_conv4/BiasAdd/ReadVariableOp?
sequential/enc_conv4/BiasAddBiasAdd$sequential/enc_conv4/Conv2D:output:03sequential/enc_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential/enc_conv4/BiasAdd?
sequential/enc_conv4/ReluRelu%sequential/enc_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential/enc_conv4/Relu?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape'sequential/enc_conv4/Relu:activations:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/flatten/Reshape?
,sequential_1/enc_fc_mu/MatMul/ReadVariableOpReadVariableOp5sequential_1_enc_fc_mu_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02.
,sequential_1/enc_fc_mu/MatMul/ReadVariableOp?
sequential_1/enc_fc_mu/MatMulMatMul#sequential/flatten/Reshape:output:04sequential_1/enc_fc_mu/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_1/enc_fc_mu/MatMul?
-sequential_1/enc_fc_mu/BiasAdd/ReadVariableOpReadVariableOp6sequential_1_enc_fc_mu_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_1/enc_fc_mu/BiasAdd/ReadVariableOp?
sequential_1/enc_fc_mu/BiasAddBiasAdd'sequential_1/enc_fc_mu/MatMul:product:05sequential_1/enc_fc_mu/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2 
sequential_1/enc_fc_mu/BiasAdd?
1sequential_2/enc_fc_log_var/MatMul/ReadVariableOpReadVariableOp:sequential_2_enc_fc_log_var_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype023
1sequential_2/enc_fc_log_var/MatMul/ReadVariableOp?
"sequential_2/enc_fc_log_var/MatMulMatMul#sequential/flatten/Reshape:output:09sequential_2/enc_fc_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2$
"sequential_2/enc_fc_log_var/MatMul?
2sequential_2/enc_fc_log_var/BiasAdd/ReadVariableOpReadVariableOp;sequential_2_enc_fc_log_var_biasadd_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_2/enc_fc_log_var/BiasAdd/ReadVariableOp?
#sequential_2/enc_fc_log_var/BiasAddBiasAdd,sequential_2/enc_fc_log_var/MatMul:product:0:sequential_2/enc_fc_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2%
#sequential_2/enc_fc_log_var/BiasAdde
ShapeShape'sequential_1/enc_fc_mu/BiasAdd:output:0*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:????????? *
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:????????? 2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:????????? 2
random_normalS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/y?
mulMul,sequential_2/enc_fc_log_var/BiasAdd:output:0mul/y:output:0*
T0*'
_output_shapes
:????????? 2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:????????? 2
Expc
mul_1Mulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:????????? 2
mul_1y
addAddV2	mul_1:z:0'sequential_1/enc_fc_mu/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
add?
-sequential_3/dec_dense1/MatMul/ReadVariableOpReadVariableOp6sequential_3_dec_dense1_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02/
-sequential_3/dec_dense1/MatMul/ReadVariableOp?
sequential_3/dec_dense1/MatMulMatMuladd:z:05sequential_3/dec_dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_3/dec_dense1/MatMul?
.sequential_3/dec_dense1/BiasAdd/ReadVariableOpReadVariableOp7sequential_3_dec_dense1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.sequential_3/dec_dense1/BiasAdd/ReadVariableOp?
sequential_3/dec_dense1/BiasAddBiasAdd(sequential_3/dec_dense1/MatMul:product:06sequential_3/dec_dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
sequential_3/dec_dense1/BiasAdd?
sequential_3/dec_dense1/ReluRelu(sequential_3/dec_dense1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_3/dec_dense1/Relu?
sequential_3/reshape/ShapeShape*sequential_3/dec_dense1/Relu:activations:0*
T0*
_output_shapes
:2
sequential_3/reshape/Shape?
(sequential_3/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_3/reshape/strided_slice/stack?
*sequential_3/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_3/reshape/strided_slice/stack_1?
*sequential_3/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_3/reshape/strided_slice/stack_2?
"sequential_3/reshape/strided_sliceStridedSlice#sequential_3/reshape/Shape:output:01sequential_3/reshape/strided_slice/stack:output:03sequential_3/reshape/strided_slice/stack_1:output:03sequential_3/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_3/reshape/strided_slice?
$sequential_3/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_3/reshape/Reshape/shape/1?
$sequential_3/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_3/reshape/Reshape/shape/2?
$sequential_3/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2&
$sequential_3/reshape/Reshape/shape/3?
"sequential_3/reshape/Reshape/shapePack+sequential_3/reshape/strided_slice:output:0-sequential_3/reshape/Reshape/shape/1:output:0-sequential_3/reshape/Reshape/shape/2:output:0-sequential_3/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"sequential_3/reshape/Reshape/shape?
sequential_3/reshape/ReshapeReshape*sequential_3/dec_dense1/Relu:activations:0+sequential_3/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
sequential_3/reshape/Reshape?
sequential_3/dec_deconv1/ShapeShape%sequential_3/reshape/Reshape:output:0*
T0*
_output_shapes
:2 
sequential_3/dec_deconv1/Shape?
,sequential_3/dec_deconv1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_3/dec_deconv1/strided_slice/stack?
.sequential_3/dec_deconv1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv1/strided_slice/stack_1?
.sequential_3/dec_deconv1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv1/strided_slice/stack_2?
&sequential_3/dec_deconv1/strided_sliceStridedSlice'sequential_3/dec_deconv1/Shape:output:05sequential_3/dec_deconv1/strided_slice/stack:output:07sequential_3/dec_deconv1/strided_slice/stack_1:output:07sequential_3/dec_deconv1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_3/dec_deconv1/strided_slice?
.sequential_3/dec_deconv1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv1/strided_slice_1/stack?
0sequential_3/dec_deconv1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_1/stack_1?
0sequential_3/dec_deconv1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_1/stack_2?
(sequential_3/dec_deconv1/strided_slice_1StridedSlice'sequential_3/dec_deconv1/Shape:output:07sequential_3/dec_deconv1/strided_slice_1/stack:output:09sequential_3/dec_deconv1/strided_slice_1/stack_1:output:09sequential_3/dec_deconv1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv1/strided_slice_1?
.sequential_3/dec_deconv1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv1/strided_slice_2/stack?
0sequential_3/dec_deconv1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_2/stack_1?
0sequential_3/dec_deconv1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_2/stack_2?
(sequential_3/dec_deconv1/strided_slice_2StridedSlice'sequential_3/dec_deconv1/Shape:output:07sequential_3/dec_deconv1/strided_slice_2/stack:output:09sequential_3/dec_deconv1/strided_slice_2/stack_1:output:09sequential_3/dec_deconv1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv1/strided_slice_2?
sequential_3/dec_deconv1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv1/mul/y?
sequential_3/dec_deconv1/mulMul1sequential_3/dec_deconv1/strided_slice_1:output:0'sequential_3/dec_deconv1/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv1/mul?
sequential_3/dec_deconv1/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv1/add/y?
sequential_3/dec_deconv1/addAddV2 sequential_3/dec_deconv1/mul:z:0'sequential_3/dec_deconv1/add/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv1/add?
 sequential_3/dec_deconv1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv1/mul_1/y?
sequential_3/dec_deconv1/mul_1Mul1sequential_3/dec_deconv1/strided_slice_2:output:0)sequential_3/dec_deconv1/mul_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv1/mul_1?
 sequential_3/dec_deconv1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv1/add_1/y?
sequential_3/dec_deconv1/add_1AddV2"sequential_3/dec_deconv1/mul_1:z:0)sequential_3/dec_deconv1/add_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv1/add_1?
 sequential_3/dec_deconv1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2"
 sequential_3/dec_deconv1/stack/3?
sequential_3/dec_deconv1/stackPack/sequential_3/dec_deconv1/strided_slice:output:0 sequential_3/dec_deconv1/add:z:0"sequential_3/dec_deconv1/add_1:z:0)sequential_3/dec_deconv1/stack/3:output:0*
N*
T0*
_output_shapes
:2 
sequential_3/dec_deconv1/stack?
.sequential_3/dec_deconv1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_3/dec_deconv1/strided_slice_3/stack?
0sequential_3/dec_deconv1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_3/stack_1?
0sequential_3/dec_deconv1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_3/stack_2?
(sequential_3/dec_deconv1/strided_slice_3StridedSlice'sequential_3/dec_deconv1/stack:output:07sequential_3/dec_deconv1/strided_slice_3/stack:output:09sequential_3/dec_deconv1/strided_slice_3/stack_1:output:09sequential_3/dec_deconv1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv1/strided_slice_3?
8sequential_3/dec_deconv1/conv2d_transpose/ReadVariableOpReadVariableOpAsequential_3_dec_deconv1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02:
8sequential_3/dec_deconv1/conv2d_transpose/ReadVariableOp?
)sequential_3/dec_deconv1/conv2d_transposeConv2DBackpropInput'sequential_3/dec_deconv1/stack:output:0@sequential_3/dec_deconv1/conv2d_transpose/ReadVariableOp:value:0%sequential_3/reshape/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2+
)sequential_3/dec_deconv1/conv2d_transpose?
/sequential_3/dec_deconv1/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dec_deconv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/sequential_3/dec_deconv1/BiasAdd/ReadVariableOp?
 sequential_3/dec_deconv1/BiasAddBiasAdd2sequential_3/dec_deconv1/conv2d_transpose:output:07sequential_3/dec_deconv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2"
 sequential_3/dec_deconv1/BiasAdd?
sequential_3/dec_deconv1/ReluRelu)sequential_3/dec_deconv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential_3/dec_deconv1/Relu?
sequential_3/dec_deconv2/ShapeShape+sequential_3/dec_deconv1/Relu:activations:0*
T0*
_output_shapes
:2 
sequential_3/dec_deconv2/Shape?
,sequential_3/dec_deconv2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_3/dec_deconv2/strided_slice/stack?
.sequential_3/dec_deconv2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv2/strided_slice/stack_1?
.sequential_3/dec_deconv2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv2/strided_slice/stack_2?
&sequential_3/dec_deconv2/strided_sliceStridedSlice'sequential_3/dec_deconv2/Shape:output:05sequential_3/dec_deconv2/strided_slice/stack:output:07sequential_3/dec_deconv2/strided_slice/stack_1:output:07sequential_3/dec_deconv2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_3/dec_deconv2/strided_slice?
.sequential_3/dec_deconv2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv2/strided_slice_1/stack?
0sequential_3/dec_deconv2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_1/stack_1?
0sequential_3/dec_deconv2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_1/stack_2?
(sequential_3/dec_deconv2/strided_slice_1StridedSlice'sequential_3/dec_deconv2/Shape:output:07sequential_3/dec_deconv2/strided_slice_1/stack:output:09sequential_3/dec_deconv2/strided_slice_1/stack_1:output:09sequential_3/dec_deconv2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv2/strided_slice_1?
.sequential_3/dec_deconv2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv2/strided_slice_2/stack?
0sequential_3/dec_deconv2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_2/stack_1?
0sequential_3/dec_deconv2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_2/stack_2?
(sequential_3/dec_deconv2/strided_slice_2StridedSlice'sequential_3/dec_deconv2/Shape:output:07sequential_3/dec_deconv2/strided_slice_2/stack:output:09sequential_3/dec_deconv2/strided_slice_2/stack_1:output:09sequential_3/dec_deconv2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv2/strided_slice_2?
sequential_3/dec_deconv2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv2/mul/y?
sequential_3/dec_deconv2/mulMul1sequential_3/dec_deconv2/strided_slice_1:output:0'sequential_3/dec_deconv2/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv2/mul?
sequential_3/dec_deconv2/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv2/add/y?
sequential_3/dec_deconv2/addAddV2 sequential_3/dec_deconv2/mul:z:0'sequential_3/dec_deconv2/add/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv2/add?
 sequential_3/dec_deconv2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv2/mul_1/y?
sequential_3/dec_deconv2/mul_1Mul1sequential_3/dec_deconv2/strided_slice_2:output:0)sequential_3/dec_deconv2/mul_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv2/mul_1?
 sequential_3/dec_deconv2/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv2/add_1/y?
sequential_3/dec_deconv2/add_1AddV2"sequential_3/dec_deconv2/mul_1:z:0)sequential_3/dec_deconv2/add_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv2/add_1?
 sequential_3/dec_deconv2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2"
 sequential_3/dec_deconv2/stack/3?
sequential_3/dec_deconv2/stackPack/sequential_3/dec_deconv2/strided_slice:output:0 sequential_3/dec_deconv2/add:z:0"sequential_3/dec_deconv2/add_1:z:0)sequential_3/dec_deconv2/stack/3:output:0*
N*
T0*
_output_shapes
:2 
sequential_3/dec_deconv2/stack?
.sequential_3/dec_deconv2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_3/dec_deconv2/strided_slice_3/stack?
0sequential_3/dec_deconv2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_3/stack_1?
0sequential_3/dec_deconv2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_3/stack_2?
(sequential_3/dec_deconv2/strided_slice_3StridedSlice'sequential_3/dec_deconv2/stack:output:07sequential_3/dec_deconv2/strided_slice_3/stack:output:09sequential_3/dec_deconv2/strided_slice_3/stack_1:output:09sequential_3/dec_deconv2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv2/strided_slice_3?
8sequential_3/dec_deconv2/conv2d_transpose/ReadVariableOpReadVariableOpAsequential_3_dec_deconv2_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02:
8sequential_3/dec_deconv2/conv2d_transpose/ReadVariableOp?
)sequential_3/dec_deconv2/conv2d_transposeConv2DBackpropInput'sequential_3/dec_deconv2/stack:output:0@sequential_3/dec_deconv2/conv2d_transpose/ReadVariableOp:value:0+sequential_3/dec_deconv1/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2+
)sequential_3/dec_deconv2/conv2d_transpose?
/sequential_3/dec_deconv2/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dec_deconv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/sequential_3/dec_deconv2/BiasAdd/ReadVariableOp?
 sequential_3/dec_deconv2/BiasAddBiasAdd2sequential_3/dec_deconv2/conv2d_transpose:output:07sequential_3/dec_deconv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2"
 sequential_3/dec_deconv2/BiasAdd?
sequential_3/dec_deconv2/ReluRelu)sequential_3/dec_deconv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_3/dec_deconv2/Relu?
sequential_3/dec_deconv3/ShapeShape+sequential_3/dec_deconv2/Relu:activations:0*
T0*
_output_shapes
:2 
sequential_3/dec_deconv3/Shape?
,sequential_3/dec_deconv3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_3/dec_deconv3/strided_slice/stack?
.sequential_3/dec_deconv3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv3/strided_slice/stack_1?
.sequential_3/dec_deconv3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv3/strided_slice/stack_2?
&sequential_3/dec_deconv3/strided_sliceStridedSlice'sequential_3/dec_deconv3/Shape:output:05sequential_3/dec_deconv3/strided_slice/stack:output:07sequential_3/dec_deconv3/strided_slice/stack_1:output:07sequential_3/dec_deconv3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_3/dec_deconv3/strided_slice?
.sequential_3/dec_deconv3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv3/strided_slice_1/stack?
0sequential_3/dec_deconv3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_1/stack_1?
0sequential_3/dec_deconv3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_1/stack_2?
(sequential_3/dec_deconv3/strided_slice_1StridedSlice'sequential_3/dec_deconv3/Shape:output:07sequential_3/dec_deconv3/strided_slice_1/stack:output:09sequential_3/dec_deconv3/strided_slice_1/stack_1:output:09sequential_3/dec_deconv3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv3/strided_slice_1?
.sequential_3/dec_deconv3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv3/strided_slice_2/stack?
0sequential_3/dec_deconv3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_2/stack_1?
0sequential_3/dec_deconv3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_2/stack_2?
(sequential_3/dec_deconv3/strided_slice_2StridedSlice'sequential_3/dec_deconv3/Shape:output:07sequential_3/dec_deconv3/strided_slice_2/stack:output:09sequential_3/dec_deconv3/strided_slice_2/stack_1:output:09sequential_3/dec_deconv3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv3/strided_slice_2?
sequential_3/dec_deconv3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv3/mul/y?
sequential_3/dec_deconv3/mulMul1sequential_3/dec_deconv3/strided_slice_1:output:0'sequential_3/dec_deconv3/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv3/mul?
sequential_3/dec_deconv3/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv3/add/y?
sequential_3/dec_deconv3/addAddV2 sequential_3/dec_deconv3/mul:z:0'sequential_3/dec_deconv3/add/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv3/add?
 sequential_3/dec_deconv3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv3/mul_1/y?
sequential_3/dec_deconv3/mul_1Mul1sequential_3/dec_deconv3/strided_slice_2:output:0)sequential_3/dec_deconv3/mul_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv3/mul_1?
 sequential_3/dec_deconv3/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv3/add_1/y?
sequential_3/dec_deconv3/add_1AddV2"sequential_3/dec_deconv3/mul_1:z:0)sequential_3/dec_deconv3/add_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv3/add_1?
 sequential_3/dec_deconv3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2"
 sequential_3/dec_deconv3/stack/3?
sequential_3/dec_deconv3/stackPack/sequential_3/dec_deconv3/strided_slice:output:0 sequential_3/dec_deconv3/add:z:0"sequential_3/dec_deconv3/add_1:z:0)sequential_3/dec_deconv3/stack/3:output:0*
N*
T0*
_output_shapes
:2 
sequential_3/dec_deconv3/stack?
.sequential_3/dec_deconv3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_3/dec_deconv3/strided_slice_3/stack?
0sequential_3/dec_deconv3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_3/stack_1?
0sequential_3/dec_deconv3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_3/stack_2?
(sequential_3/dec_deconv3/strided_slice_3StridedSlice'sequential_3/dec_deconv3/stack:output:07sequential_3/dec_deconv3/strided_slice_3/stack:output:09sequential_3/dec_deconv3/strided_slice_3/stack_1:output:09sequential_3/dec_deconv3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv3/strided_slice_3?
8sequential_3/dec_deconv3/conv2d_transpose/ReadVariableOpReadVariableOpAsequential_3_dec_deconv3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02:
8sequential_3/dec_deconv3/conv2d_transpose/ReadVariableOp?
)sequential_3/dec_deconv3/conv2d_transposeConv2DBackpropInput'sequential_3/dec_deconv3/stack:output:0@sequential_3/dec_deconv3/conv2d_transpose/ReadVariableOp:value:0+sequential_3/dec_deconv2/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2+
)sequential_3/dec_deconv3/conv2d_transpose?
/sequential_3/dec_deconv3/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dec_deconv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential_3/dec_deconv3/BiasAdd/ReadVariableOp?
 sequential_3/dec_deconv3/BiasAddBiasAdd2sequential_3/dec_deconv3/conv2d_transpose:output:07sequential_3/dec_deconv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2"
 sequential_3/dec_deconv3/BiasAdd?
sequential_3/dec_deconv3/ReluRelu)sequential_3/dec_deconv3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_3/dec_deconv3/Relu?
sequential_3/dec_deconv4/ShapeShape+sequential_3/dec_deconv3/Relu:activations:0*
T0*
_output_shapes
:2 
sequential_3/dec_deconv4/Shape?
,sequential_3/dec_deconv4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_3/dec_deconv4/strided_slice/stack?
.sequential_3/dec_deconv4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv4/strided_slice/stack_1?
.sequential_3/dec_deconv4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv4/strided_slice/stack_2?
&sequential_3/dec_deconv4/strided_sliceStridedSlice'sequential_3/dec_deconv4/Shape:output:05sequential_3/dec_deconv4/strided_slice/stack:output:07sequential_3/dec_deconv4/strided_slice/stack_1:output:07sequential_3/dec_deconv4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_3/dec_deconv4/strided_slice?
.sequential_3/dec_deconv4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv4/strided_slice_1/stack?
0sequential_3/dec_deconv4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_1/stack_1?
0sequential_3/dec_deconv4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_1/stack_2?
(sequential_3/dec_deconv4/strided_slice_1StridedSlice'sequential_3/dec_deconv4/Shape:output:07sequential_3/dec_deconv4/strided_slice_1/stack:output:09sequential_3/dec_deconv4/strided_slice_1/stack_1:output:09sequential_3/dec_deconv4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv4/strided_slice_1?
.sequential_3/dec_deconv4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv4/strided_slice_2/stack?
0sequential_3/dec_deconv4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_2/stack_1?
0sequential_3/dec_deconv4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_2/stack_2?
(sequential_3/dec_deconv4/strided_slice_2StridedSlice'sequential_3/dec_deconv4/Shape:output:07sequential_3/dec_deconv4/strided_slice_2/stack:output:09sequential_3/dec_deconv4/strided_slice_2/stack_1:output:09sequential_3/dec_deconv4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv4/strided_slice_2?
sequential_3/dec_deconv4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv4/mul/y?
sequential_3/dec_deconv4/mulMul1sequential_3/dec_deconv4/strided_slice_1:output:0'sequential_3/dec_deconv4/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv4/mul?
sequential_3/dec_deconv4/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv4/add/y?
sequential_3/dec_deconv4/addAddV2 sequential_3/dec_deconv4/mul:z:0'sequential_3/dec_deconv4/add/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv4/add?
 sequential_3/dec_deconv4/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv4/mul_1/y?
sequential_3/dec_deconv4/mul_1Mul1sequential_3/dec_deconv4/strided_slice_2:output:0)sequential_3/dec_deconv4/mul_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv4/mul_1?
 sequential_3/dec_deconv4/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv4/add_1/y?
sequential_3/dec_deconv4/add_1AddV2"sequential_3/dec_deconv4/mul_1:z:0)sequential_3/dec_deconv4/add_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv4/add_1?
 sequential_3/dec_deconv4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv4/stack/3?
sequential_3/dec_deconv4/stackPack/sequential_3/dec_deconv4/strided_slice:output:0 sequential_3/dec_deconv4/add:z:0"sequential_3/dec_deconv4/add_1:z:0)sequential_3/dec_deconv4/stack/3:output:0*
N*
T0*
_output_shapes
:2 
sequential_3/dec_deconv4/stack?
.sequential_3/dec_deconv4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_3/dec_deconv4/strided_slice_3/stack?
0sequential_3/dec_deconv4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_3/stack_1?
0sequential_3/dec_deconv4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_3/stack_2?
(sequential_3/dec_deconv4/strided_slice_3StridedSlice'sequential_3/dec_deconv4/stack:output:07sequential_3/dec_deconv4/strided_slice_3/stack:output:09sequential_3/dec_deconv4/strided_slice_3/stack_1:output:09sequential_3/dec_deconv4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv4/strided_slice_3?
8sequential_3/dec_deconv4/conv2d_transpose/ReadVariableOpReadVariableOpAsequential_3_dec_deconv4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02:
8sequential_3/dec_deconv4/conv2d_transpose/ReadVariableOp?
)sequential_3/dec_deconv4/conv2d_transposeConv2DBackpropInput'sequential_3/dec_deconv4/stack:output:0@sequential_3/dec_deconv4/conv2d_transpose/ReadVariableOp:value:0+sequential_3/dec_deconv3/Relu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingVALID*
strides
2+
)sequential_3/dec_deconv4/conv2d_transpose?
/sequential_3/dec_deconv4/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dec_deconv4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_3/dec_deconv4/BiasAdd/ReadVariableOp?
 sequential_3/dec_deconv4/BiasAddBiasAdd2sequential_3/dec_deconv4/conv2d_transpose:output:07sequential_3/dec_deconv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2"
 sequential_3/dec_deconv4/BiasAdd?
 sequential_3/dec_deconv4/SigmoidSigmoid)sequential_3/dec_deconv4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2"
 sequential_3/dec_deconv4/Sigmoide
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2'sequential_1/enc_fc_mu/BiasAdd:output:0,sequential_2/enc_fc_log_var/BiasAdd:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity$sequential_3/dec_deconv4/Sigmoid:y:0*
T0*/
_output_shapes
:?????????@@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesu
s:?????????@@:::::::::::::::::::::::W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: 
?

*__inference_enc_conv4_layer_call_fn_877354

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_conv4_layer_call_and_return_conditional_losses_8773442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
H__inference_sequential_2_layer_call_and_return_conditional_losses_877628
input_3
enc_fc_log_var_877622
enc_fc_log_var_877624
identity??&enc_fc_log_var/StatefulPartitionedCall?
&enc_fc_log_var/StatefulPartitionedCallStatefulPartitionedCallinput_3enc_fc_log_var_877622enc_fc_log_var_877624*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_enc_fc_log_var_layer_call_and_return_conditional_losses_8776022(
&enc_fc_log_var/StatefulPartitionedCall?
IdentityIdentity/enc_fc_log_var/StatefulPartitionedCall:output:0'^enc_fc_log_var/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2P
&enc_fc_log_var/StatefulPartitionedCall&enc_fc_log_var/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: 
?

*__inference_enc_fc_mu_layer_call_fn_880269

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_fc_mu_layer_call_and_return_conditional_losses_8775252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
J__inference_enc_fc_log_var_layer_call_and_return_conditional_losses_877602

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
??
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_880032

inputs-
)dec_dense1_matmul_readvariableop_resource.
*dec_dense1_biasadd_readvariableop_resource8
4dec_deconv1_conv2d_transpose_readvariableop_resource/
+dec_deconv1_biasadd_readvariableop_resource8
4dec_deconv2_conv2d_transpose_readvariableop_resource/
+dec_deconv2_biasadd_readvariableop_resource8
4dec_deconv3_conv2d_transpose_readvariableop_resource/
+dec_deconv3_biasadd_readvariableop_resource8
4dec_deconv4_conv2d_transpose_readvariableop_resource/
+dec_deconv4_biasadd_readvariableop_resource
identity??
 dec_dense1/MatMul/ReadVariableOpReadVariableOp)dec_dense1_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02"
 dec_dense1/MatMul/ReadVariableOp?
dec_dense1/MatMulMatMulinputs(dec_dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dec_dense1/MatMul?
!dec_dense1/BiasAdd/ReadVariableOpReadVariableOp*dec_dense1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!dec_dense1/BiasAdd/ReadVariableOp?
dec_dense1/BiasAddBiasAdddec_dense1/MatMul:product:0)dec_dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dec_dense1/BiasAddz
dec_dense1/ReluReludec_dense1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dec_dense1/Reluk
reshape/ShapeShapedec_dense1/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2u
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedec_dense1/Relu:activations:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape/Reshapen
dec_deconv1/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
dec_deconv1/Shape?
dec_deconv1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
dec_deconv1/strided_slice/stack?
!dec_deconv1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv1/strided_slice/stack_1?
!dec_deconv1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv1/strided_slice/stack_2?
dec_deconv1/strided_sliceStridedSlicedec_deconv1/Shape:output:0(dec_deconv1/strided_slice/stack:output:0*dec_deconv1/strided_slice/stack_1:output:0*dec_deconv1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv1/strided_slice?
!dec_deconv1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv1/strided_slice_1/stack?
#dec_deconv1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv1/strided_slice_1/stack_1?
#dec_deconv1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv1/strided_slice_1/stack_2?
dec_deconv1/strided_slice_1StridedSlicedec_deconv1/Shape:output:0*dec_deconv1/strided_slice_1/stack:output:0,dec_deconv1/strided_slice_1/stack_1:output:0,dec_deconv1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv1/strided_slice_1?
!dec_deconv1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv1/strided_slice_2/stack?
#dec_deconv1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv1/strided_slice_2/stack_1?
#dec_deconv1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv1/strided_slice_2/stack_2?
dec_deconv1/strided_slice_2StridedSlicedec_deconv1/Shape:output:0*dec_deconv1/strided_slice_2/stack:output:0,dec_deconv1/strided_slice_2/stack_1:output:0,dec_deconv1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv1/strided_slice_2h
dec_deconv1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv1/mul/y?
dec_deconv1/mulMul$dec_deconv1/strided_slice_1:output:0dec_deconv1/mul/y:output:0*
T0*
_output_shapes
: 2
dec_deconv1/mulh
dec_deconv1/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv1/add/y}
dec_deconv1/addAddV2dec_deconv1/mul:z:0dec_deconv1/add/y:output:0*
T0*
_output_shapes
: 2
dec_deconv1/addl
dec_deconv1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv1/mul_1/y?
dec_deconv1/mul_1Mul$dec_deconv1/strided_slice_2:output:0dec_deconv1/mul_1/y:output:0*
T0*
_output_shapes
: 2
dec_deconv1/mul_1l
dec_deconv1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv1/add_1/y?
dec_deconv1/add_1AddV2dec_deconv1/mul_1:z:0dec_deconv1/add_1/y:output:0*
T0*
_output_shapes
: 2
dec_deconv1/add_1m
dec_deconv1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
dec_deconv1/stack/3?
dec_deconv1/stackPack"dec_deconv1/strided_slice:output:0dec_deconv1/add:z:0dec_deconv1/add_1:z:0dec_deconv1/stack/3:output:0*
N*
T0*
_output_shapes
:2
dec_deconv1/stack?
!dec_deconv1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dec_deconv1/strided_slice_3/stack?
#dec_deconv1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv1/strided_slice_3/stack_1?
#dec_deconv1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv1/strided_slice_3/stack_2?
dec_deconv1/strided_slice_3StridedSlicedec_deconv1/stack:output:0*dec_deconv1/strided_slice_3/stack:output:0,dec_deconv1/strided_slice_3/stack_1:output:0,dec_deconv1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv1/strided_slice_3?
+dec_deconv1/conv2d_transpose/ReadVariableOpReadVariableOp4dec_deconv1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02-
+dec_deconv1/conv2d_transpose/ReadVariableOp?
dec_deconv1/conv2d_transposeConv2DBackpropInputdec_deconv1/stack:output:03dec_deconv1/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
dec_deconv1/conv2d_transpose?
"dec_deconv1/BiasAdd/ReadVariableOpReadVariableOp+dec_deconv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"dec_deconv1/BiasAdd/ReadVariableOp?
dec_deconv1/BiasAddBiasAdd%dec_deconv1/conv2d_transpose:output:0*dec_deconv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
dec_deconv1/BiasAdd?
dec_deconv1/ReluReludec_deconv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
dec_deconv1/Relut
dec_deconv2/ShapeShapedec_deconv1/Relu:activations:0*
T0*
_output_shapes
:2
dec_deconv2/Shape?
dec_deconv2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
dec_deconv2/strided_slice/stack?
!dec_deconv2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv2/strided_slice/stack_1?
!dec_deconv2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv2/strided_slice/stack_2?
dec_deconv2/strided_sliceStridedSlicedec_deconv2/Shape:output:0(dec_deconv2/strided_slice/stack:output:0*dec_deconv2/strided_slice/stack_1:output:0*dec_deconv2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv2/strided_slice?
!dec_deconv2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv2/strided_slice_1/stack?
#dec_deconv2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv2/strided_slice_1/stack_1?
#dec_deconv2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv2/strided_slice_1/stack_2?
dec_deconv2/strided_slice_1StridedSlicedec_deconv2/Shape:output:0*dec_deconv2/strided_slice_1/stack:output:0,dec_deconv2/strided_slice_1/stack_1:output:0,dec_deconv2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv2/strided_slice_1?
!dec_deconv2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv2/strided_slice_2/stack?
#dec_deconv2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv2/strided_slice_2/stack_1?
#dec_deconv2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv2/strided_slice_2/stack_2?
dec_deconv2/strided_slice_2StridedSlicedec_deconv2/Shape:output:0*dec_deconv2/strided_slice_2/stack:output:0,dec_deconv2/strided_slice_2/stack_1:output:0,dec_deconv2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv2/strided_slice_2h
dec_deconv2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv2/mul/y?
dec_deconv2/mulMul$dec_deconv2/strided_slice_1:output:0dec_deconv2/mul/y:output:0*
T0*
_output_shapes
: 2
dec_deconv2/mulh
dec_deconv2/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv2/add/y}
dec_deconv2/addAddV2dec_deconv2/mul:z:0dec_deconv2/add/y:output:0*
T0*
_output_shapes
: 2
dec_deconv2/addl
dec_deconv2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv2/mul_1/y?
dec_deconv2/mul_1Mul$dec_deconv2/strided_slice_2:output:0dec_deconv2/mul_1/y:output:0*
T0*
_output_shapes
: 2
dec_deconv2/mul_1l
dec_deconv2/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv2/add_1/y?
dec_deconv2/add_1AddV2dec_deconv2/mul_1:z:0dec_deconv2/add_1/y:output:0*
T0*
_output_shapes
: 2
dec_deconv2/add_1l
dec_deconv2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
dec_deconv2/stack/3?
dec_deconv2/stackPack"dec_deconv2/strided_slice:output:0dec_deconv2/add:z:0dec_deconv2/add_1:z:0dec_deconv2/stack/3:output:0*
N*
T0*
_output_shapes
:2
dec_deconv2/stack?
!dec_deconv2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dec_deconv2/strided_slice_3/stack?
#dec_deconv2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv2/strided_slice_3/stack_1?
#dec_deconv2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv2/strided_slice_3/stack_2?
dec_deconv2/strided_slice_3StridedSlicedec_deconv2/stack:output:0*dec_deconv2/strided_slice_3/stack:output:0,dec_deconv2/strided_slice_3/stack_1:output:0,dec_deconv2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv2/strided_slice_3?
+dec_deconv2/conv2d_transpose/ReadVariableOpReadVariableOp4dec_deconv2_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02-
+dec_deconv2/conv2d_transpose/ReadVariableOp?
dec_deconv2/conv2d_transposeConv2DBackpropInputdec_deconv2/stack:output:03dec_deconv2/conv2d_transpose/ReadVariableOp:value:0dec_deconv1/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
dec_deconv2/conv2d_transpose?
"dec_deconv2/BiasAdd/ReadVariableOpReadVariableOp+dec_deconv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"dec_deconv2/BiasAdd/ReadVariableOp?
dec_deconv2/BiasAddBiasAdd%dec_deconv2/conv2d_transpose:output:0*dec_deconv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
dec_deconv2/BiasAdd?
dec_deconv2/ReluReludec_deconv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
dec_deconv2/Relut
dec_deconv3/ShapeShapedec_deconv2/Relu:activations:0*
T0*
_output_shapes
:2
dec_deconv3/Shape?
dec_deconv3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
dec_deconv3/strided_slice/stack?
!dec_deconv3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv3/strided_slice/stack_1?
!dec_deconv3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv3/strided_slice/stack_2?
dec_deconv3/strided_sliceStridedSlicedec_deconv3/Shape:output:0(dec_deconv3/strided_slice/stack:output:0*dec_deconv3/strided_slice/stack_1:output:0*dec_deconv3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv3/strided_slice?
!dec_deconv3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv3/strided_slice_1/stack?
#dec_deconv3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv3/strided_slice_1/stack_1?
#dec_deconv3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv3/strided_slice_1/stack_2?
dec_deconv3/strided_slice_1StridedSlicedec_deconv3/Shape:output:0*dec_deconv3/strided_slice_1/stack:output:0,dec_deconv3/strided_slice_1/stack_1:output:0,dec_deconv3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv3/strided_slice_1?
!dec_deconv3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv3/strided_slice_2/stack?
#dec_deconv3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv3/strided_slice_2/stack_1?
#dec_deconv3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv3/strided_slice_2/stack_2?
dec_deconv3/strided_slice_2StridedSlicedec_deconv3/Shape:output:0*dec_deconv3/strided_slice_2/stack:output:0,dec_deconv3/strided_slice_2/stack_1:output:0,dec_deconv3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv3/strided_slice_2h
dec_deconv3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv3/mul/y?
dec_deconv3/mulMul$dec_deconv3/strided_slice_1:output:0dec_deconv3/mul/y:output:0*
T0*
_output_shapes
: 2
dec_deconv3/mulh
dec_deconv3/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv3/add/y}
dec_deconv3/addAddV2dec_deconv3/mul:z:0dec_deconv3/add/y:output:0*
T0*
_output_shapes
: 2
dec_deconv3/addl
dec_deconv3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv3/mul_1/y?
dec_deconv3/mul_1Mul$dec_deconv3/strided_slice_2:output:0dec_deconv3/mul_1/y:output:0*
T0*
_output_shapes
: 2
dec_deconv3/mul_1l
dec_deconv3/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv3/add_1/y?
dec_deconv3/add_1AddV2dec_deconv3/mul_1:z:0dec_deconv3/add_1/y:output:0*
T0*
_output_shapes
: 2
dec_deconv3/add_1l
dec_deconv3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
dec_deconv3/stack/3?
dec_deconv3/stackPack"dec_deconv3/strided_slice:output:0dec_deconv3/add:z:0dec_deconv3/add_1:z:0dec_deconv3/stack/3:output:0*
N*
T0*
_output_shapes
:2
dec_deconv3/stack?
!dec_deconv3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dec_deconv3/strided_slice_3/stack?
#dec_deconv3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv3/strided_slice_3/stack_1?
#dec_deconv3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv3/strided_slice_3/stack_2?
dec_deconv3/strided_slice_3StridedSlicedec_deconv3/stack:output:0*dec_deconv3/strided_slice_3/stack:output:0,dec_deconv3/strided_slice_3/stack_1:output:0,dec_deconv3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv3/strided_slice_3?
+dec_deconv3/conv2d_transpose/ReadVariableOpReadVariableOp4dec_deconv3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+dec_deconv3/conv2d_transpose/ReadVariableOp?
dec_deconv3/conv2d_transposeConv2DBackpropInputdec_deconv3/stack:output:03dec_deconv3/conv2d_transpose/ReadVariableOp:value:0dec_deconv2/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
dec_deconv3/conv2d_transpose?
"dec_deconv3/BiasAdd/ReadVariableOpReadVariableOp+dec_deconv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"dec_deconv3/BiasAdd/ReadVariableOp?
dec_deconv3/BiasAddBiasAdd%dec_deconv3/conv2d_transpose:output:0*dec_deconv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
dec_deconv3/BiasAdd?
dec_deconv3/ReluReludec_deconv3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
dec_deconv3/Relut
dec_deconv4/ShapeShapedec_deconv3/Relu:activations:0*
T0*
_output_shapes
:2
dec_deconv4/Shape?
dec_deconv4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
dec_deconv4/strided_slice/stack?
!dec_deconv4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv4/strided_slice/stack_1?
!dec_deconv4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv4/strided_slice/stack_2?
dec_deconv4/strided_sliceStridedSlicedec_deconv4/Shape:output:0(dec_deconv4/strided_slice/stack:output:0*dec_deconv4/strided_slice/stack_1:output:0*dec_deconv4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv4/strided_slice?
!dec_deconv4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv4/strided_slice_1/stack?
#dec_deconv4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv4/strided_slice_1/stack_1?
#dec_deconv4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv4/strided_slice_1/stack_2?
dec_deconv4/strided_slice_1StridedSlicedec_deconv4/Shape:output:0*dec_deconv4/strided_slice_1/stack:output:0,dec_deconv4/strided_slice_1/stack_1:output:0,dec_deconv4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv4/strided_slice_1?
!dec_deconv4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv4/strided_slice_2/stack?
#dec_deconv4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv4/strided_slice_2/stack_1?
#dec_deconv4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv4/strided_slice_2/stack_2?
dec_deconv4/strided_slice_2StridedSlicedec_deconv4/Shape:output:0*dec_deconv4/strided_slice_2/stack:output:0,dec_deconv4/strided_slice_2/stack_1:output:0,dec_deconv4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv4/strided_slice_2h
dec_deconv4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv4/mul/y?
dec_deconv4/mulMul$dec_deconv4/strided_slice_1:output:0dec_deconv4/mul/y:output:0*
T0*
_output_shapes
: 2
dec_deconv4/mulh
dec_deconv4/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv4/add/y}
dec_deconv4/addAddV2dec_deconv4/mul:z:0dec_deconv4/add/y:output:0*
T0*
_output_shapes
: 2
dec_deconv4/addl
dec_deconv4/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv4/mul_1/y?
dec_deconv4/mul_1Mul$dec_deconv4/strided_slice_2:output:0dec_deconv4/mul_1/y:output:0*
T0*
_output_shapes
: 2
dec_deconv4/mul_1l
dec_deconv4/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv4/add_1/y?
dec_deconv4/add_1AddV2dec_deconv4/mul_1:z:0dec_deconv4/add_1/y:output:0*
T0*
_output_shapes
: 2
dec_deconv4/add_1l
dec_deconv4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv4/stack/3?
dec_deconv4/stackPack"dec_deconv4/strided_slice:output:0dec_deconv4/add:z:0dec_deconv4/add_1:z:0dec_deconv4/stack/3:output:0*
N*
T0*
_output_shapes
:2
dec_deconv4/stack?
!dec_deconv4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dec_deconv4/strided_slice_3/stack?
#dec_deconv4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv4/strided_slice_3/stack_1?
#dec_deconv4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv4/strided_slice_3/stack_2?
dec_deconv4/strided_slice_3StridedSlicedec_deconv4/stack:output:0*dec_deconv4/strided_slice_3/stack:output:0,dec_deconv4/strided_slice_3/stack_1:output:0,dec_deconv4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv4/strided_slice_3?
+dec_deconv4/conv2d_transpose/ReadVariableOpReadVariableOp4dec_deconv4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02-
+dec_deconv4/conv2d_transpose/ReadVariableOp?
dec_deconv4/conv2d_transposeConv2DBackpropInputdec_deconv4/stack:output:03dec_deconv4/conv2d_transpose/ReadVariableOp:value:0dec_deconv3/Relu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingVALID*
strides
2
dec_deconv4/conv2d_transpose?
"dec_deconv4/BiasAdd/ReadVariableOpReadVariableOp+dec_deconv4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"dec_deconv4/BiasAdd/ReadVariableOp?
dec_deconv4/BiasAddBiasAdd%dec_deconv4/conv2d_transpose:output:0*dec_deconv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
dec_deconv4/BiasAdd?
dec_deconv4/SigmoidSigmoiddec_deconv4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
dec_deconv4/Sigmoids
IdentityIdentitydec_deconv4/Sigmoid:y:0*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:????????? :::::::::::O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
?
/__inference_enc_fc_log_var_layer_call_fn_880288

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_enc_fc_log_var_layer_call_and_return_conditional_losses_8776022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
$__inference_signature_wrapper_878633
input_1
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

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*B
_output_shapes0
.:?????????@:?????????@@*8
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8**
f%R#
!__inference__wrapped_model_8772662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesu
s:?????????@@::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: 
?
?
-__inference_sequential_2_layer_call_fn_879866

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_8776402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_877581

inputs
enc_fc_mu_877575
enc_fc_mu_877577
identity??!enc_fc_mu/StatefulPartitionedCall?
!enc_fc_mu/StatefulPartitionedCallStatefulPartitionedCallinputsenc_fc_mu_877575enc_fc_mu_877577*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_fc_mu_layer_call_and_return_conditional_losses_8775252#
!enc_fc_mu/StatefulPartitionedCall?
IdentityIdentity*enc_fc_mu/StatefulPartitionedCall:output:0"^enc_fc_mu/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2F
!enc_fc_mu/StatefulPartitionedCall!enc_fc_mu/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_877563

inputs
enc_fc_mu_877557
enc_fc_mu_877559
identity??!enc_fc_mu/StatefulPartitionedCall?
!enc_fc_mu/StatefulPartitionedCallStatefulPartitionedCallinputsenc_fc_mu_877557enc_fc_mu_877559*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_fc_mu_layer_call_and_return_conditional_losses_8775252#
!enc_fc_mu/StatefulPartitionedCall?
IdentityIdentity*enc_fc_mu/StatefulPartitionedCall:output:0"^enc_fc_mu/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2F
!enc_fc_mu/StatefulPartitionedCall!enc_fc_mu/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?!
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_877935
input_4
dec_dense1_877887
dec_dense1_877889
dec_deconv1_877914
dec_deconv1_877916
dec_deconv2_877919
dec_deconv2_877921
dec_deconv3_877924
dec_deconv3_877926
dec_deconv4_877929
dec_deconv4_877931
identity??#dec_deconv1/StatefulPartitionedCall?#dec_deconv2/StatefulPartitionedCall?#dec_deconv3/StatefulPartitionedCall?#dec_deconv4/StatefulPartitionedCall?"dec_dense1/StatefulPartitionedCall?
"dec_dense1/StatefulPartitionedCallStatefulPartitionedCallinput_4dec_dense1_877887dec_dense1_877889*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_dec_dense1_layer_call_and_return_conditional_losses_8778762$
"dec_dense1/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall+dec_dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_8779062
reshape/PartitionedCall?
#dec_deconv1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0dec_deconv1_877914dec_deconv1_877916*
Tin
2*
Tout
2*B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dec_deconv1_layer_call_and_return_conditional_losses_8777042%
#dec_deconv1/StatefulPartitionedCall?
#dec_deconv2/StatefulPartitionedCallStatefulPartitionedCall,dec_deconv1/StatefulPartitionedCall:output:0dec_deconv2_877919dec_deconv2_877921*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dec_deconv2_layer_call_and_return_conditional_losses_8777532%
#dec_deconv2/StatefulPartitionedCall?
#dec_deconv3/StatefulPartitionedCallStatefulPartitionedCall,dec_deconv2/StatefulPartitionedCall:output:0dec_deconv3_877924dec_deconv3_877926*
Tin
2*
Tout
2*A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dec_deconv3_layer_call_and_return_conditional_losses_8778022%
#dec_deconv3/StatefulPartitionedCall?
#dec_deconv4/StatefulPartitionedCallStatefulPartitionedCall,dec_deconv3/StatefulPartitionedCall:output:0dec_deconv4_877929dec_deconv4_877931*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dec_deconv4_layer_call_and_return_conditional_losses_8778512%
#dec_deconv4/StatefulPartitionedCall?
IdentityIdentity,dec_deconv4/StatefulPartitionedCall:output:0$^dec_deconv1/StatefulPartitionedCall$^dec_deconv2/StatefulPartitionedCall$^dec_deconv3/StatefulPartitionedCall$^dec_deconv4/StatefulPartitionedCall#^dec_dense1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:????????? ::::::::::2J
#dec_deconv1/StatefulPartitionedCall#dec_deconv1/StatefulPartitionedCall2J
#dec_deconv2/StatefulPartitionedCall#dec_deconv2/StatefulPartitionedCall2J
#dec_deconv3/StatefulPartitionedCall#dec_deconv3/StatefulPartitionedCall2J
#dec_deconv4/StatefulPartitionedCall#dec_deconv4/StatefulPartitionedCall2H
"dec_dense1/StatefulPartitionedCall"dec_dense1/StatefulPartitionedCall:P L
'
_output_shapes
:????????? 
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
?
-__inference_sequential_3_layer_call_fn_878076
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_8780532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:????????? ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:????????? 
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?%
?
F__inference_sequential_layer_call_and_return_conditional_losses_879723

inputs,
(enc_conv1_conv2d_readvariableop_resource-
)enc_conv1_biasadd_readvariableop_resource,
(enc_conv2_conv2d_readvariableop_resource-
)enc_conv2_biasadd_readvariableop_resource,
(enc_conv3_conv2d_readvariableop_resource-
)enc_conv3_biasadd_readvariableop_resource,
(enc_conv4_conv2d_readvariableop_resource-
)enc_conv4_biasadd_readvariableop_resource
identity??
enc_conv1/Conv2D/ReadVariableOpReadVariableOp(enc_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
enc_conv1/Conv2D/ReadVariableOp?
enc_conv1/Conv2DConv2Dinputs'enc_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
enc_conv1/Conv2D?
 enc_conv1/BiasAdd/ReadVariableOpReadVariableOp)enc_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 enc_conv1/BiasAdd/ReadVariableOp?
enc_conv1/BiasAddBiasAddenc_conv1/Conv2D:output:0(enc_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
enc_conv1/BiasAdd~
enc_conv1/ReluReluenc_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
enc_conv1/Relu?
enc_conv2/Conv2D/ReadVariableOpReadVariableOp(enc_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
enc_conv2/Conv2D/ReadVariableOp?
enc_conv2/Conv2DConv2Denc_conv1/Relu:activations:0'enc_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
enc_conv2/Conv2D?
 enc_conv2/BiasAdd/ReadVariableOpReadVariableOp)enc_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 enc_conv2/BiasAdd/ReadVariableOp?
enc_conv2/BiasAddBiasAddenc_conv2/Conv2D:output:0(enc_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
enc_conv2/BiasAdd~
enc_conv2/ReluReluenc_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
enc_conv2/Relu?
enc_conv3/Conv2D/ReadVariableOpReadVariableOp(enc_conv3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
enc_conv3/Conv2D/ReadVariableOp?
enc_conv3/Conv2DConv2Denc_conv2/Relu:activations:0'enc_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
enc_conv3/Conv2D?
 enc_conv3/BiasAdd/ReadVariableOpReadVariableOp)enc_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 enc_conv3/BiasAdd/ReadVariableOp?
enc_conv3/BiasAddBiasAddenc_conv3/Conv2D:output:0(enc_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
enc_conv3/BiasAdd
enc_conv3/ReluReluenc_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
enc_conv3/Relu?
enc_conv4/Conv2D/ReadVariableOpReadVariableOp(enc_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
enc_conv4/Conv2D/ReadVariableOp?
enc_conv4/Conv2DConv2Denc_conv3/Relu:activations:0'enc_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
enc_conv4/Conv2D?
 enc_conv4/BiasAdd/ReadVariableOpReadVariableOp)enc_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 enc_conv4/BiasAdd/ReadVariableOp?
enc_conv4/BiasAddBiasAddenc_conv4/Conv2D:output:0(enc_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
enc_conv4/BiasAdd
enc_conv4/ReluReluenc_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
enc_conv4/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshapeenc_conv4/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshapem
IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????@@:::::::::W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ھ
?
!__inference__wrapped_model_877266
input_17
3sequential_enc_conv1_conv2d_readvariableop_resource8
4sequential_enc_conv1_biasadd_readvariableop_resource7
3sequential_enc_conv2_conv2d_readvariableop_resource8
4sequential_enc_conv2_biasadd_readvariableop_resource7
3sequential_enc_conv3_conv2d_readvariableop_resource8
4sequential_enc_conv3_biasadd_readvariableop_resource7
3sequential_enc_conv4_conv2d_readvariableop_resource8
4sequential_enc_conv4_biasadd_readvariableop_resource9
5sequential_1_enc_fc_mu_matmul_readvariableop_resource:
6sequential_1_enc_fc_mu_biasadd_readvariableop_resource>
:sequential_2_enc_fc_log_var_matmul_readvariableop_resource?
;sequential_2_enc_fc_log_var_biasadd_readvariableop_resource:
6sequential_3_dec_dense1_matmul_readvariableop_resource;
7sequential_3_dec_dense1_biasadd_readvariableop_resourceE
Asequential_3_dec_deconv1_conv2d_transpose_readvariableop_resource<
8sequential_3_dec_deconv1_biasadd_readvariableop_resourceE
Asequential_3_dec_deconv2_conv2d_transpose_readvariableop_resource<
8sequential_3_dec_deconv2_biasadd_readvariableop_resourceE
Asequential_3_dec_deconv3_conv2d_transpose_readvariableop_resource<
8sequential_3_dec_deconv3_biasadd_readvariableop_resourceE
Asequential_3_dec_deconv4_conv2d_transpose_readvariableop_resource<
8sequential_3_dec_deconv4_biasadd_readvariableop_resource
identity

identity_1??
*sequential/enc_conv1/Conv2D/ReadVariableOpReadVariableOp3sequential_enc_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*sequential/enc_conv1/Conv2D/ReadVariableOp?
sequential/enc_conv1/Conv2DConv2Dinput_12sequential/enc_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
sequential/enc_conv1/Conv2D?
+sequential/enc_conv1/BiasAdd/ReadVariableOpReadVariableOp4sequential_enc_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential/enc_conv1/BiasAdd/ReadVariableOp?
sequential/enc_conv1/BiasAddBiasAdd$sequential/enc_conv1/Conv2D:output:03sequential/enc_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential/enc_conv1/BiasAdd?
sequential/enc_conv1/ReluRelu%sequential/enc_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential/enc_conv1/Relu?
*sequential/enc_conv2/Conv2D/ReadVariableOpReadVariableOp3sequential_enc_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*sequential/enc_conv2/Conv2D/ReadVariableOp?
sequential/enc_conv2/Conv2DConv2D'sequential/enc_conv1/Relu:activations:02sequential/enc_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
sequential/enc_conv2/Conv2D?
+sequential/enc_conv2/BiasAdd/ReadVariableOpReadVariableOp4sequential_enc_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential/enc_conv2/BiasAdd/ReadVariableOp?
sequential/enc_conv2/BiasAddBiasAdd$sequential/enc_conv2/Conv2D:output:03sequential/enc_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential/enc_conv2/BiasAdd?
sequential/enc_conv2/ReluRelu%sequential/enc_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential/enc_conv2/Relu?
*sequential/enc_conv3/Conv2D/ReadVariableOpReadVariableOp3sequential_enc_conv3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02,
*sequential/enc_conv3/Conv2D/ReadVariableOp?
sequential/enc_conv3/Conv2DConv2D'sequential/enc_conv2/Relu:activations:02sequential/enc_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential/enc_conv3/Conv2D?
+sequential/enc_conv3/BiasAdd/ReadVariableOpReadVariableOp4sequential_enc_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential/enc_conv3/BiasAdd/ReadVariableOp?
sequential/enc_conv3/BiasAddBiasAdd$sequential/enc_conv3/Conv2D:output:03sequential/enc_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential/enc_conv3/BiasAdd?
sequential/enc_conv3/ReluRelu%sequential/enc_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential/enc_conv3/Relu?
*sequential/enc_conv4/Conv2D/ReadVariableOpReadVariableOp3sequential_enc_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*sequential/enc_conv4/Conv2D/ReadVariableOp?
sequential/enc_conv4/Conv2DConv2D'sequential/enc_conv3/Relu:activations:02sequential/enc_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential/enc_conv4/Conv2D?
+sequential/enc_conv4/BiasAdd/ReadVariableOpReadVariableOp4sequential_enc_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential/enc_conv4/BiasAdd/ReadVariableOp?
sequential/enc_conv4/BiasAddBiasAdd$sequential/enc_conv4/Conv2D:output:03sequential/enc_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential/enc_conv4/BiasAdd?
sequential/enc_conv4/ReluRelu%sequential/enc_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential/enc_conv4/Relu?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape'sequential/enc_conv4/Relu:activations:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/flatten/Reshape?
,sequential_1/enc_fc_mu/MatMul/ReadVariableOpReadVariableOp5sequential_1_enc_fc_mu_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02.
,sequential_1/enc_fc_mu/MatMul/ReadVariableOp?
sequential_1/enc_fc_mu/MatMulMatMul#sequential/flatten/Reshape:output:04sequential_1/enc_fc_mu/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_1/enc_fc_mu/MatMul?
-sequential_1/enc_fc_mu/BiasAdd/ReadVariableOpReadVariableOp6sequential_1_enc_fc_mu_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_1/enc_fc_mu/BiasAdd/ReadVariableOp?
sequential_1/enc_fc_mu/BiasAddBiasAdd'sequential_1/enc_fc_mu/MatMul:product:05sequential_1/enc_fc_mu/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2 
sequential_1/enc_fc_mu/BiasAdd?
1sequential_2/enc_fc_log_var/MatMul/ReadVariableOpReadVariableOp:sequential_2_enc_fc_log_var_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype023
1sequential_2/enc_fc_log_var/MatMul/ReadVariableOp?
"sequential_2/enc_fc_log_var/MatMulMatMul#sequential/flatten/Reshape:output:09sequential_2/enc_fc_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2$
"sequential_2/enc_fc_log_var/MatMul?
2sequential_2/enc_fc_log_var/BiasAdd/ReadVariableOpReadVariableOp;sequential_2_enc_fc_log_var_biasadd_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_2/enc_fc_log_var/BiasAdd/ReadVariableOp?
#sequential_2/enc_fc_log_var/BiasAddBiasAdd,sequential_2/enc_fc_log_var/MatMul:product:0:sequential_2/enc_fc_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2%
#sequential_2/enc_fc_log_var/BiasAdde
ShapeShape'sequential_1/enc_fc_mu/BiasAdd:output:0*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:????????? *
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:????????? 2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:????????? 2
random_normalS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/y?
mulMul,sequential_2/enc_fc_log_var/BiasAdd:output:0mul/y:output:0*
T0*'
_output_shapes
:????????? 2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:????????? 2
Expc
mul_1Mulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:????????? 2
mul_1y
addAddV2	mul_1:z:0'sequential_1/enc_fc_mu/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
add?
-sequential_3/dec_dense1/MatMul/ReadVariableOpReadVariableOp6sequential_3_dec_dense1_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02/
-sequential_3/dec_dense1/MatMul/ReadVariableOp?
sequential_3/dec_dense1/MatMulMatMuladd:z:05sequential_3/dec_dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_3/dec_dense1/MatMul?
.sequential_3/dec_dense1/BiasAdd/ReadVariableOpReadVariableOp7sequential_3_dec_dense1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.sequential_3/dec_dense1/BiasAdd/ReadVariableOp?
sequential_3/dec_dense1/BiasAddBiasAdd(sequential_3/dec_dense1/MatMul:product:06sequential_3/dec_dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
sequential_3/dec_dense1/BiasAdd?
sequential_3/dec_dense1/ReluRelu(sequential_3/dec_dense1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_3/dec_dense1/Relu?
sequential_3/reshape/ShapeShape*sequential_3/dec_dense1/Relu:activations:0*
T0*
_output_shapes
:2
sequential_3/reshape/Shape?
(sequential_3/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_3/reshape/strided_slice/stack?
*sequential_3/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_3/reshape/strided_slice/stack_1?
*sequential_3/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_3/reshape/strided_slice/stack_2?
"sequential_3/reshape/strided_sliceStridedSlice#sequential_3/reshape/Shape:output:01sequential_3/reshape/strided_slice/stack:output:03sequential_3/reshape/strided_slice/stack_1:output:03sequential_3/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_3/reshape/strided_slice?
$sequential_3/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_3/reshape/Reshape/shape/1?
$sequential_3/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_3/reshape/Reshape/shape/2?
$sequential_3/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2&
$sequential_3/reshape/Reshape/shape/3?
"sequential_3/reshape/Reshape/shapePack+sequential_3/reshape/strided_slice:output:0-sequential_3/reshape/Reshape/shape/1:output:0-sequential_3/reshape/Reshape/shape/2:output:0-sequential_3/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"sequential_3/reshape/Reshape/shape?
sequential_3/reshape/ReshapeReshape*sequential_3/dec_dense1/Relu:activations:0+sequential_3/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
sequential_3/reshape/Reshape?
sequential_3/dec_deconv1/ShapeShape%sequential_3/reshape/Reshape:output:0*
T0*
_output_shapes
:2 
sequential_3/dec_deconv1/Shape?
,sequential_3/dec_deconv1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_3/dec_deconv1/strided_slice/stack?
.sequential_3/dec_deconv1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv1/strided_slice/stack_1?
.sequential_3/dec_deconv1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv1/strided_slice/stack_2?
&sequential_3/dec_deconv1/strided_sliceStridedSlice'sequential_3/dec_deconv1/Shape:output:05sequential_3/dec_deconv1/strided_slice/stack:output:07sequential_3/dec_deconv1/strided_slice/stack_1:output:07sequential_3/dec_deconv1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_3/dec_deconv1/strided_slice?
.sequential_3/dec_deconv1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv1/strided_slice_1/stack?
0sequential_3/dec_deconv1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_1/stack_1?
0sequential_3/dec_deconv1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_1/stack_2?
(sequential_3/dec_deconv1/strided_slice_1StridedSlice'sequential_3/dec_deconv1/Shape:output:07sequential_3/dec_deconv1/strided_slice_1/stack:output:09sequential_3/dec_deconv1/strided_slice_1/stack_1:output:09sequential_3/dec_deconv1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv1/strided_slice_1?
.sequential_3/dec_deconv1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv1/strided_slice_2/stack?
0sequential_3/dec_deconv1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_2/stack_1?
0sequential_3/dec_deconv1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_2/stack_2?
(sequential_3/dec_deconv1/strided_slice_2StridedSlice'sequential_3/dec_deconv1/Shape:output:07sequential_3/dec_deconv1/strided_slice_2/stack:output:09sequential_3/dec_deconv1/strided_slice_2/stack_1:output:09sequential_3/dec_deconv1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv1/strided_slice_2?
sequential_3/dec_deconv1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv1/mul/y?
sequential_3/dec_deconv1/mulMul1sequential_3/dec_deconv1/strided_slice_1:output:0'sequential_3/dec_deconv1/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv1/mul?
sequential_3/dec_deconv1/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv1/add/y?
sequential_3/dec_deconv1/addAddV2 sequential_3/dec_deconv1/mul:z:0'sequential_3/dec_deconv1/add/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv1/add?
 sequential_3/dec_deconv1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv1/mul_1/y?
sequential_3/dec_deconv1/mul_1Mul1sequential_3/dec_deconv1/strided_slice_2:output:0)sequential_3/dec_deconv1/mul_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv1/mul_1?
 sequential_3/dec_deconv1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv1/add_1/y?
sequential_3/dec_deconv1/add_1AddV2"sequential_3/dec_deconv1/mul_1:z:0)sequential_3/dec_deconv1/add_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv1/add_1?
 sequential_3/dec_deconv1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2"
 sequential_3/dec_deconv1/stack/3?
sequential_3/dec_deconv1/stackPack/sequential_3/dec_deconv1/strided_slice:output:0 sequential_3/dec_deconv1/add:z:0"sequential_3/dec_deconv1/add_1:z:0)sequential_3/dec_deconv1/stack/3:output:0*
N*
T0*
_output_shapes
:2 
sequential_3/dec_deconv1/stack?
.sequential_3/dec_deconv1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_3/dec_deconv1/strided_slice_3/stack?
0sequential_3/dec_deconv1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_3/stack_1?
0sequential_3/dec_deconv1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_3/stack_2?
(sequential_3/dec_deconv1/strided_slice_3StridedSlice'sequential_3/dec_deconv1/stack:output:07sequential_3/dec_deconv1/strided_slice_3/stack:output:09sequential_3/dec_deconv1/strided_slice_3/stack_1:output:09sequential_3/dec_deconv1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv1/strided_slice_3?
8sequential_3/dec_deconv1/conv2d_transpose/ReadVariableOpReadVariableOpAsequential_3_dec_deconv1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02:
8sequential_3/dec_deconv1/conv2d_transpose/ReadVariableOp?
)sequential_3/dec_deconv1/conv2d_transposeConv2DBackpropInput'sequential_3/dec_deconv1/stack:output:0@sequential_3/dec_deconv1/conv2d_transpose/ReadVariableOp:value:0%sequential_3/reshape/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2+
)sequential_3/dec_deconv1/conv2d_transpose?
/sequential_3/dec_deconv1/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dec_deconv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/sequential_3/dec_deconv1/BiasAdd/ReadVariableOp?
 sequential_3/dec_deconv1/BiasAddBiasAdd2sequential_3/dec_deconv1/conv2d_transpose:output:07sequential_3/dec_deconv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2"
 sequential_3/dec_deconv1/BiasAdd?
sequential_3/dec_deconv1/ReluRelu)sequential_3/dec_deconv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential_3/dec_deconv1/Relu?
sequential_3/dec_deconv2/ShapeShape+sequential_3/dec_deconv1/Relu:activations:0*
T0*
_output_shapes
:2 
sequential_3/dec_deconv2/Shape?
,sequential_3/dec_deconv2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_3/dec_deconv2/strided_slice/stack?
.sequential_3/dec_deconv2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv2/strided_slice/stack_1?
.sequential_3/dec_deconv2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv2/strided_slice/stack_2?
&sequential_3/dec_deconv2/strided_sliceStridedSlice'sequential_3/dec_deconv2/Shape:output:05sequential_3/dec_deconv2/strided_slice/stack:output:07sequential_3/dec_deconv2/strided_slice/stack_1:output:07sequential_3/dec_deconv2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_3/dec_deconv2/strided_slice?
.sequential_3/dec_deconv2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv2/strided_slice_1/stack?
0sequential_3/dec_deconv2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_1/stack_1?
0sequential_3/dec_deconv2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_1/stack_2?
(sequential_3/dec_deconv2/strided_slice_1StridedSlice'sequential_3/dec_deconv2/Shape:output:07sequential_3/dec_deconv2/strided_slice_1/stack:output:09sequential_3/dec_deconv2/strided_slice_1/stack_1:output:09sequential_3/dec_deconv2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv2/strided_slice_1?
.sequential_3/dec_deconv2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv2/strided_slice_2/stack?
0sequential_3/dec_deconv2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_2/stack_1?
0sequential_3/dec_deconv2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_2/stack_2?
(sequential_3/dec_deconv2/strided_slice_2StridedSlice'sequential_3/dec_deconv2/Shape:output:07sequential_3/dec_deconv2/strided_slice_2/stack:output:09sequential_3/dec_deconv2/strided_slice_2/stack_1:output:09sequential_3/dec_deconv2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv2/strided_slice_2?
sequential_3/dec_deconv2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv2/mul/y?
sequential_3/dec_deconv2/mulMul1sequential_3/dec_deconv2/strided_slice_1:output:0'sequential_3/dec_deconv2/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv2/mul?
sequential_3/dec_deconv2/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv2/add/y?
sequential_3/dec_deconv2/addAddV2 sequential_3/dec_deconv2/mul:z:0'sequential_3/dec_deconv2/add/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv2/add?
 sequential_3/dec_deconv2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv2/mul_1/y?
sequential_3/dec_deconv2/mul_1Mul1sequential_3/dec_deconv2/strided_slice_2:output:0)sequential_3/dec_deconv2/mul_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv2/mul_1?
 sequential_3/dec_deconv2/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv2/add_1/y?
sequential_3/dec_deconv2/add_1AddV2"sequential_3/dec_deconv2/mul_1:z:0)sequential_3/dec_deconv2/add_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv2/add_1?
 sequential_3/dec_deconv2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2"
 sequential_3/dec_deconv2/stack/3?
sequential_3/dec_deconv2/stackPack/sequential_3/dec_deconv2/strided_slice:output:0 sequential_3/dec_deconv2/add:z:0"sequential_3/dec_deconv2/add_1:z:0)sequential_3/dec_deconv2/stack/3:output:0*
N*
T0*
_output_shapes
:2 
sequential_3/dec_deconv2/stack?
.sequential_3/dec_deconv2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_3/dec_deconv2/strided_slice_3/stack?
0sequential_3/dec_deconv2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_3/stack_1?
0sequential_3/dec_deconv2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_3/stack_2?
(sequential_3/dec_deconv2/strided_slice_3StridedSlice'sequential_3/dec_deconv2/stack:output:07sequential_3/dec_deconv2/strided_slice_3/stack:output:09sequential_3/dec_deconv2/strided_slice_3/stack_1:output:09sequential_3/dec_deconv2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv2/strided_slice_3?
8sequential_3/dec_deconv2/conv2d_transpose/ReadVariableOpReadVariableOpAsequential_3_dec_deconv2_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02:
8sequential_3/dec_deconv2/conv2d_transpose/ReadVariableOp?
)sequential_3/dec_deconv2/conv2d_transposeConv2DBackpropInput'sequential_3/dec_deconv2/stack:output:0@sequential_3/dec_deconv2/conv2d_transpose/ReadVariableOp:value:0+sequential_3/dec_deconv1/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2+
)sequential_3/dec_deconv2/conv2d_transpose?
/sequential_3/dec_deconv2/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dec_deconv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/sequential_3/dec_deconv2/BiasAdd/ReadVariableOp?
 sequential_3/dec_deconv2/BiasAddBiasAdd2sequential_3/dec_deconv2/conv2d_transpose:output:07sequential_3/dec_deconv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2"
 sequential_3/dec_deconv2/BiasAdd?
sequential_3/dec_deconv2/ReluRelu)sequential_3/dec_deconv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_3/dec_deconv2/Relu?
sequential_3/dec_deconv3/ShapeShape+sequential_3/dec_deconv2/Relu:activations:0*
T0*
_output_shapes
:2 
sequential_3/dec_deconv3/Shape?
,sequential_3/dec_deconv3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_3/dec_deconv3/strided_slice/stack?
.sequential_3/dec_deconv3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv3/strided_slice/stack_1?
.sequential_3/dec_deconv3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv3/strided_slice/stack_2?
&sequential_3/dec_deconv3/strided_sliceStridedSlice'sequential_3/dec_deconv3/Shape:output:05sequential_3/dec_deconv3/strided_slice/stack:output:07sequential_3/dec_deconv3/strided_slice/stack_1:output:07sequential_3/dec_deconv3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_3/dec_deconv3/strided_slice?
.sequential_3/dec_deconv3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv3/strided_slice_1/stack?
0sequential_3/dec_deconv3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_1/stack_1?
0sequential_3/dec_deconv3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_1/stack_2?
(sequential_3/dec_deconv3/strided_slice_1StridedSlice'sequential_3/dec_deconv3/Shape:output:07sequential_3/dec_deconv3/strided_slice_1/stack:output:09sequential_3/dec_deconv3/strided_slice_1/stack_1:output:09sequential_3/dec_deconv3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv3/strided_slice_1?
.sequential_3/dec_deconv3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv3/strided_slice_2/stack?
0sequential_3/dec_deconv3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_2/stack_1?
0sequential_3/dec_deconv3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_2/stack_2?
(sequential_3/dec_deconv3/strided_slice_2StridedSlice'sequential_3/dec_deconv3/Shape:output:07sequential_3/dec_deconv3/strided_slice_2/stack:output:09sequential_3/dec_deconv3/strided_slice_2/stack_1:output:09sequential_3/dec_deconv3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv3/strided_slice_2?
sequential_3/dec_deconv3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv3/mul/y?
sequential_3/dec_deconv3/mulMul1sequential_3/dec_deconv3/strided_slice_1:output:0'sequential_3/dec_deconv3/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv3/mul?
sequential_3/dec_deconv3/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv3/add/y?
sequential_3/dec_deconv3/addAddV2 sequential_3/dec_deconv3/mul:z:0'sequential_3/dec_deconv3/add/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv3/add?
 sequential_3/dec_deconv3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv3/mul_1/y?
sequential_3/dec_deconv3/mul_1Mul1sequential_3/dec_deconv3/strided_slice_2:output:0)sequential_3/dec_deconv3/mul_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv3/mul_1?
 sequential_3/dec_deconv3/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv3/add_1/y?
sequential_3/dec_deconv3/add_1AddV2"sequential_3/dec_deconv3/mul_1:z:0)sequential_3/dec_deconv3/add_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv3/add_1?
 sequential_3/dec_deconv3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2"
 sequential_3/dec_deconv3/stack/3?
sequential_3/dec_deconv3/stackPack/sequential_3/dec_deconv3/strided_slice:output:0 sequential_3/dec_deconv3/add:z:0"sequential_3/dec_deconv3/add_1:z:0)sequential_3/dec_deconv3/stack/3:output:0*
N*
T0*
_output_shapes
:2 
sequential_3/dec_deconv3/stack?
.sequential_3/dec_deconv3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_3/dec_deconv3/strided_slice_3/stack?
0sequential_3/dec_deconv3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_3/stack_1?
0sequential_3/dec_deconv3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_3/stack_2?
(sequential_3/dec_deconv3/strided_slice_3StridedSlice'sequential_3/dec_deconv3/stack:output:07sequential_3/dec_deconv3/strided_slice_3/stack:output:09sequential_3/dec_deconv3/strided_slice_3/stack_1:output:09sequential_3/dec_deconv3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv3/strided_slice_3?
8sequential_3/dec_deconv3/conv2d_transpose/ReadVariableOpReadVariableOpAsequential_3_dec_deconv3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02:
8sequential_3/dec_deconv3/conv2d_transpose/ReadVariableOp?
)sequential_3/dec_deconv3/conv2d_transposeConv2DBackpropInput'sequential_3/dec_deconv3/stack:output:0@sequential_3/dec_deconv3/conv2d_transpose/ReadVariableOp:value:0+sequential_3/dec_deconv2/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2+
)sequential_3/dec_deconv3/conv2d_transpose?
/sequential_3/dec_deconv3/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dec_deconv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential_3/dec_deconv3/BiasAdd/ReadVariableOp?
 sequential_3/dec_deconv3/BiasAddBiasAdd2sequential_3/dec_deconv3/conv2d_transpose:output:07sequential_3/dec_deconv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2"
 sequential_3/dec_deconv3/BiasAdd?
sequential_3/dec_deconv3/ReluRelu)sequential_3/dec_deconv3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_3/dec_deconv3/Relu?
sequential_3/dec_deconv4/ShapeShape+sequential_3/dec_deconv3/Relu:activations:0*
T0*
_output_shapes
:2 
sequential_3/dec_deconv4/Shape?
,sequential_3/dec_deconv4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_3/dec_deconv4/strided_slice/stack?
.sequential_3/dec_deconv4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv4/strided_slice/stack_1?
.sequential_3/dec_deconv4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv4/strided_slice/stack_2?
&sequential_3/dec_deconv4/strided_sliceStridedSlice'sequential_3/dec_deconv4/Shape:output:05sequential_3/dec_deconv4/strided_slice/stack:output:07sequential_3/dec_deconv4/strided_slice/stack_1:output:07sequential_3/dec_deconv4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_3/dec_deconv4/strided_slice?
.sequential_3/dec_deconv4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv4/strided_slice_1/stack?
0sequential_3/dec_deconv4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_1/stack_1?
0sequential_3/dec_deconv4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_1/stack_2?
(sequential_3/dec_deconv4/strided_slice_1StridedSlice'sequential_3/dec_deconv4/Shape:output:07sequential_3/dec_deconv4/strided_slice_1/stack:output:09sequential_3/dec_deconv4/strided_slice_1/stack_1:output:09sequential_3/dec_deconv4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv4/strided_slice_1?
.sequential_3/dec_deconv4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv4/strided_slice_2/stack?
0sequential_3/dec_deconv4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_2/stack_1?
0sequential_3/dec_deconv4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_2/stack_2?
(sequential_3/dec_deconv4/strided_slice_2StridedSlice'sequential_3/dec_deconv4/Shape:output:07sequential_3/dec_deconv4/strided_slice_2/stack:output:09sequential_3/dec_deconv4/strided_slice_2/stack_1:output:09sequential_3/dec_deconv4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv4/strided_slice_2?
sequential_3/dec_deconv4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv4/mul/y?
sequential_3/dec_deconv4/mulMul1sequential_3/dec_deconv4/strided_slice_1:output:0'sequential_3/dec_deconv4/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv4/mul?
sequential_3/dec_deconv4/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv4/add/y?
sequential_3/dec_deconv4/addAddV2 sequential_3/dec_deconv4/mul:z:0'sequential_3/dec_deconv4/add/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv4/add?
 sequential_3/dec_deconv4/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv4/mul_1/y?
sequential_3/dec_deconv4/mul_1Mul1sequential_3/dec_deconv4/strided_slice_2:output:0)sequential_3/dec_deconv4/mul_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv4/mul_1?
 sequential_3/dec_deconv4/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv4/add_1/y?
sequential_3/dec_deconv4/add_1AddV2"sequential_3/dec_deconv4/mul_1:z:0)sequential_3/dec_deconv4/add_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv4/add_1?
 sequential_3/dec_deconv4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv4/stack/3?
sequential_3/dec_deconv4/stackPack/sequential_3/dec_deconv4/strided_slice:output:0 sequential_3/dec_deconv4/add:z:0"sequential_3/dec_deconv4/add_1:z:0)sequential_3/dec_deconv4/stack/3:output:0*
N*
T0*
_output_shapes
:2 
sequential_3/dec_deconv4/stack?
.sequential_3/dec_deconv4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_3/dec_deconv4/strided_slice_3/stack?
0sequential_3/dec_deconv4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_3/stack_1?
0sequential_3/dec_deconv4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_3/stack_2?
(sequential_3/dec_deconv4/strided_slice_3StridedSlice'sequential_3/dec_deconv4/stack:output:07sequential_3/dec_deconv4/strided_slice_3/stack:output:09sequential_3/dec_deconv4/strided_slice_3/stack_1:output:09sequential_3/dec_deconv4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv4/strided_slice_3?
8sequential_3/dec_deconv4/conv2d_transpose/ReadVariableOpReadVariableOpAsequential_3_dec_deconv4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02:
8sequential_3/dec_deconv4/conv2d_transpose/ReadVariableOp?
)sequential_3/dec_deconv4/conv2d_transposeConv2DBackpropInput'sequential_3/dec_deconv4/stack:output:0@sequential_3/dec_deconv4/conv2d_transpose/ReadVariableOp:value:0+sequential_3/dec_deconv3/Relu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingVALID*
strides
2+
)sequential_3/dec_deconv4/conv2d_transpose?
/sequential_3/dec_deconv4/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dec_deconv4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_3/dec_deconv4/BiasAdd/ReadVariableOp?
 sequential_3/dec_deconv4/BiasAddBiasAdd2sequential_3/dec_deconv4/conv2d_transpose:output:07sequential_3/dec_deconv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2"
 sequential_3/dec_deconv4/BiasAdd?
 sequential_3/dec_deconv4/SigmoidSigmoid)sequential_3/dec_deconv4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2"
 sequential_3/dec_deconv4/Sigmoide
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2'sequential_1/enc_fc_mu/BiasAdd:output:0,sequential_2/enc_fc_log_var/BiasAdd:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity$sequential_3/dec_deconv4/Sigmoid:y:0*
T0*/
_output_shapes
:?????????@@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesu
s:?????????@@:::::::::::::::::::::::X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: 
?
?
H__inference_sequential_2_layer_call_and_return_conditional_losses_877619
input_3
enc_fc_log_var_877613
enc_fc_log_var_877615
identity??&enc_fc_log_var/StatefulPartitionedCall?
&enc_fc_log_var/StatefulPartitionedCallStatefulPartitionedCallinput_3enc_fc_log_var_877613enc_fc_log_var_877615*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_enc_fc_log_var_layer_call_and_return_conditional_losses_8776022(
&enc_fc_log_var/StatefulPartitionedCall?
IdentityIdentity/enc_fc_log_var/StatefulPartitionedCall:output:0'^enc_fc_log_var/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2P
&enc_fc_log_var/StatefulPartitionedCall&enc_fc_log_var/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: 
?
?
-__inference_sequential_2_layer_call_fn_879875

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_8776582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?

?
H__inference_sequential_2_layer_call_and_return_conditional_losses_879857

inputs1
-enc_fc_log_var_matmul_readvariableop_resource2
.enc_fc_log_var_biasadd_readvariableop_resource
identity??
$enc_fc_log_var/MatMul/ReadVariableOpReadVariableOp-enc_fc_log_var_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02&
$enc_fc_log_var/MatMul/ReadVariableOp?
enc_fc_log_var/MatMulMatMulinputs,enc_fc_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
enc_fc_log_var/MatMul?
%enc_fc_log_var/BiasAdd/ReadVariableOpReadVariableOp.enc_fc_log_var_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%enc_fc_log_var/BiasAdd/ReadVariableOp?
enc_fc_log_var/BiasAddBiasAddenc_fc_log_var/MatMul:product:0-enc_fc_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
enc_fc_log_var/BiasAdds
IdentityIdentityenc_fc_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_877393
input_1
enc_conv1_877358
enc_conv1_877360
enc_conv2_877363
enc_conv2_877365
enc_conv3_877368
enc_conv3_877370
enc_conv4_877373
enc_conv4_877375
identity??!enc_conv1/StatefulPartitionedCall?!enc_conv2/StatefulPartitionedCall?!enc_conv3/StatefulPartitionedCall?!enc_conv4/StatefulPartitionedCall?
!enc_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1enc_conv1_877358enc_conv1_877360*
Tin
2*
Tout
2*/
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_conv1_layer_call_and_return_conditional_losses_8772782#
!enc_conv1/StatefulPartitionedCall?
!enc_conv2/StatefulPartitionedCallStatefulPartitionedCall*enc_conv1/StatefulPartitionedCall:output:0enc_conv2_877363enc_conv2_877365*
Tin
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_conv2_layer_call_and_return_conditional_losses_8773002#
!enc_conv2/StatefulPartitionedCall?
!enc_conv3/StatefulPartitionedCallStatefulPartitionedCall*enc_conv2/StatefulPartitionedCall:output:0enc_conv3_877368enc_conv3_877370*
Tin
2*
Tout
2*0
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_conv3_layer_call_and_return_conditional_losses_8773222#
!enc_conv3/StatefulPartitionedCall?
!enc_conv4/StatefulPartitionedCallStatefulPartitionedCall*enc_conv3/StatefulPartitionedCall:output:0enc_conv4_877373enc_conv4_877375*
Tin
2*
Tout
2*0
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_conv4_layer_call_and_return_conditional_losses_8773442#
!enc_conv4/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall*enc_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_8773842
flatten/PartitionedCall?
IdentityIdentity flatten/PartitionedCall:output:0"^enc_conv1/StatefulPartitionedCall"^enc_conv2/StatefulPartitionedCall"^enc_conv3/StatefulPartitionedCall"^enc_conv4/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????@@::::::::2F
!enc_conv1/StatefulPartitionedCall!enc_conv1/StatefulPartitionedCall2F
!enc_conv2/StatefulPartitionedCall!enc_conv2/StatefulPartitionedCall2F
!enc_conv3/StatefulPartitionedCall!enc_conv3/StatefulPartitionedCall2F
!enc_conv4/StatefulPartitionedCall!enc_conv4/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
-__inference_sequential_1_layer_call_fn_879828

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_8775632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?

?
E__inference_enc_conv3_layer_call_and_return_conditional_losses_877322

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@:::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?	
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_879809

inputs,
(enc_fc_mu_matmul_readvariableop_resource-
)enc_fc_mu_biasadd_readvariableop_resource
identity??
enc_fc_mu/MatMul/ReadVariableOpReadVariableOp(enc_fc_mu_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02!
enc_fc_mu/MatMul/ReadVariableOp?
enc_fc_mu/MatMulMatMulinputs'enc_fc_mu/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
enc_fc_mu/MatMul?
 enc_fc_mu/BiasAdd/ReadVariableOpReadVariableOp)enc_fc_mu_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 enc_fc_mu/BiasAdd/ReadVariableOp?
enc_fc_mu/BiasAddBiasAddenc_fc_mu/MatMul:product:0(enc_fc_mu/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
enc_fc_mu/BiasAddn
IdentityIdentityenc_fc_mu/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
E__inference_enc_fc_mu_layer_call_and_return_conditional_losses_880260

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?!
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_878053

inputs
dec_dense1_878026
dec_dense1_878028
dec_deconv1_878032
dec_deconv1_878034
dec_deconv2_878037
dec_deconv2_878039
dec_deconv3_878042
dec_deconv3_878044
dec_deconv4_878047
dec_deconv4_878049
identity??#dec_deconv1/StatefulPartitionedCall?#dec_deconv2/StatefulPartitionedCall?#dec_deconv3/StatefulPartitionedCall?#dec_deconv4/StatefulPartitionedCall?"dec_dense1/StatefulPartitionedCall?
"dec_dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdec_dense1_878026dec_dense1_878028*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_dec_dense1_layer_call_and_return_conditional_losses_8778762$
"dec_dense1/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall+dec_dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_8779062
reshape/PartitionedCall?
#dec_deconv1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0dec_deconv1_878032dec_deconv1_878034*
Tin
2*
Tout
2*B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dec_deconv1_layer_call_and_return_conditional_losses_8777042%
#dec_deconv1/StatefulPartitionedCall?
#dec_deconv2/StatefulPartitionedCallStatefulPartitionedCall,dec_deconv1/StatefulPartitionedCall:output:0dec_deconv2_878037dec_deconv2_878039*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dec_deconv2_layer_call_and_return_conditional_losses_8777532%
#dec_deconv2/StatefulPartitionedCall?
#dec_deconv3/StatefulPartitionedCallStatefulPartitionedCall,dec_deconv2/StatefulPartitionedCall:output:0dec_deconv3_878042dec_deconv3_878044*
Tin
2*
Tout
2*A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dec_deconv3_layer_call_and_return_conditional_losses_8778022%
#dec_deconv3/StatefulPartitionedCall?
#dec_deconv4/StatefulPartitionedCallStatefulPartitionedCall,dec_deconv3/StatefulPartitionedCall:output:0dec_deconv4_878047dec_deconv4_878049*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dec_deconv4_layer_call_and_return_conditional_losses_8778512%
#dec_deconv4/StatefulPartitionedCall?
IdentityIdentity,dec_deconv4/StatefulPartitionedCall:output:0$^dec_deconv1/StatefulPartitionedCall$^dec_deconv2/StatefulPartitionedCall$^dec_deconv3/StatefulPartitionedCall$^dec_deconv4/StatefulPartitionedCall#^dec_dense1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:????????? ::::::::::2J
#dec_deconv1/StatefulPartitionedCall#dec_deconv1/StatefulPartitionedCall2J
#dec_deconv2/StatefulPartitionedCall#dec_deconv2/StatefulPartitionedCall2J
#dec_deconv3/StatefulPartitionedCall#dec_deconv3/StatefulPartitionedCall2J
#dec_deconv4/StatefulPartitionedCall#dec_deconv4/StatefulPartitionedCall2H
"dec_dense1/StatefulPartitionedCall"dec_dense1/StatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
?
(__inference_gan_vae_layer_call_fn_879689
input_1
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

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*T
_output_shapesB
@:?????????@:+???????????????????????????*8
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_gan_vae_layer_call_and_return_conditional_losses_8785232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesu
s:?????????@@::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: 
?
?
H__inference_sequential_2_layer_call_and_return_conditional_losses_877658

inputs
enc_fc_log_var_877652
enc_fc_log_var_877654
identity??&enc_fc_log_var/StatefulPartitionedCall?
&enc_fc_log_var/StatefulPartitionedCallStatefulPartitionedCallinputsenc_fc_log_var_877652enc_fc_log_var_877654*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_enc_fc_log_var_layer_call_and_return_conditional_losses_8776022(
&enc_fc_log_var/StatefulPartitionedCall?
IdentityIdentity/enc_fc_log_var/StatefulPartitionedCall:output:0'^enc_fc_log_var/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2P
&enc_fc_log_var/StatefulPartitionedCall&enc_fc_log_var/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_880245

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
? 
__inference__traced_save_880586
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop/
+savev2_enc_conv1_kernel_read_readvariableop-
)savev2_enc_conv1_bias_read_readvariableop/
+savev2_enc_conv2_kernel_read_readvariableop-
)savev2_enc_conv2_bias_read_readvariableop/
+savev2_enc_conv3_kernel_read_readvariableop-
)savev2_enc_conv3_bias_read_readvariableop/
+savev2_enc_conv4_kernel_read_readvariableop-
)savev2_enc_conv4_bias_read_readvariableop/
+savev2_enc_fc_mu_kernel_read_readvariableop-
)savev2_enc_fc_mu_bias_read_readvariableop4
0savev2_enc_fc_log_var_kernel_read_readvariableop2
.savev2_enc_fc_log_var_bias_read_readvariableop0
,savev2_dec_dense1_kernel_read_readvariableop.
*savev2_dec_dense1_bias_read_readvariableop1
-savev2_dec_deconv1_kernel_read_readvariableop/
+savev2_dec_deconv1_bias_read_readvariableop1
-savev2_dec_deconv2_kernel_read_readvariableop/
+savev2_dec_deconv2_bias_read_readvariableop1
-savev2_dec_deconv3_kernel_read_readvariableop/
+savev2_dec_deconv3_bias_read_readvariableop1
-savev2_dec_deconv4_kernel_read_readvariableop/
+savev2_dec_deconv4_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop6
2savev2_adam_enc_conv1_kernel_m_read_readvariableop4
0savev2_adam_enc_conv1_bias_m_read_readvariableop6
2savev2_adam_enc_conv2_kernel_m_read_readvariableop4
0savev2_adam_enc_conv2_bias_m_read_readvariableop6
2savev2_adam_enc_conv3_kernel_m_read_readvariableop4
0savev2_adam_enc_conv3_bias_m_read_readvariableop6
2savev2_adam_enc_conv4_kernel_m_read_readvariableop4
0savev2_adam_enc_conv4_bias_m_read_readvariableop6
2savev2_adam_enc_fc_mu_kernel_m_read_readvariableop4
0savev2_adam_enc_fc_mu_bias_m_read_readvariableop;
7savev2_adam_enc_fc_log_var_kernel_m_read_readvariableop9
5savev2_adam_enc_fc_log_var_bias_m_read_readvariableop7
3savev2_adam_dec_dense1_kernel_m_read_readvariableop5
1savev2_adam_dec_dense1_bias_m_read_readvariableop8
4savev2_adam_dec_deconv1_kernel_m_read_readvariableop6
2savev2_adam_dec_deconv1_bias_m_read_readvariableop8
4savev2_adam_dec_deconv2_kernel_m_read_readvariableop6
2savev2_adam_dec_deconv2_bias_m_read_readvariableop8
4savev2_adam_dec_deconv3_kernel_m_read_readvariableop6
2savev2_adam_dec_deconv3_bias_m_read_readvariableop8
4savev2_adam_dec_deconv4_kernel_m_read_readvariableop6
2savev2_adam_dec_deconv4_bias_m_read_readvariableop6
2savev2_adam_enc_conv1_kernel_v_read_readvariableop4
0savev2_adam_enc_conv1_bias_v_read_readvariableop6
2savev2_adam_enc_conv2_kernel_v_read_readvariableop4
0savev2_adam_enc_conv2_bias_v_read_readvariableop6
2savev2_adam_enc_conv3_kernel_v_read_readvariableop4
0savev2_adam_enc_conv3_bias_v_read_readvariableop6
2savev2_adam_enc_conv4_kernel_v_read_readvariableop4
0savev2_adam_enc_conv4_bias_v_read_readvariableop6
2savev2_adam_enc_fc_mu_kernel_v_read_readvariableop4
0savev2_adam_enc_fc_mu_bias_v_read_readvariableop;
7savev2_adam_enc_fc_log_var_kernel_v_read_readvariableop9
5savev2_adam_enc_fc_log_var_bias_v_read_readvariableop7
3savev2_adam_dec_dense1_kernel_v_read_readvariableop5
1savev2_adam_dec_dense1_bias_v_read_readvariableop8
4savev2_adam_dec_deconv1_kernel_v_read_readvariableop6
2savev2_adam_dec_deconv1_bias_v_read_readvariableop8
4savev2_adam_dec_deconv2_kernel_v_read_readvariableop6
2savev2_adam_dec_deconv2_bias_v_read_readvariableop8
4savev2_adam_dec_deconv3_kernel_v_read_readvariableop6
2savev2_adam_dec_deconv3_bias_v_read_readvariableop8
4savev2_adam_dec_deconv4_kernel_v_read_readvariableop6
2savev2_adam_dec_deconv4_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
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
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_8f64928a2e7542c8ad13c35d0aac072f/part2	
Const_1?
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
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?)
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:M*
dtype0*?(
value?(B?(MB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:M*
dtype0*?
value?B?MB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop+savev2_enc_conv1_kernel_read_readvariableop)savev2_enc_conv1_bias_read_readvariableop+savev2_enc_conv2_kernel_read_readvariableop)savev2_enc_conv2_bias_read_readvariableop+savev2_enc_conv3_kernel_read_readvariableop)savev2_enc_conv3_bias_read_readvariableop+savev2_enc_conv4_kernel_read_readvariableop)savev2_enc_conv4_bias_read_readvariableop+savev2_enc_fc_mu_kernel_read_readvariableop)savev2_enc_fc_mu_bias_read_readvariableop0savev2_enc_fc_log_var_kernel_read_readvariableop.savev2_enc_fc_log_var_bias_read_readvariableop,savev2_dec_dense1_kernel_read_readvariableop*savev2_dec_dense1_bias_read_readvariableop-savev2_dec_deconv1_kernel_read_readvariableop+savev2_dec_deconv1_bias_read_readvariableop-savev2_dec_deconv2_kernel_read_readvariableop+savev2_dec_deconv2_bias_read_readvariableop-savev2_dec_deconv3_kernel_read_readvariableop+savev2_dec_deconv3_bias_read_readvariableop-savev2_dec_deconv4_kernel_read_readvariableop+savev2_dec_deconv4_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop2savev2_adam_enc_conv1_kernel_m_read_readvariableop0savev2_adam_enc_conv1_bias_m_read_readvariableop2savev2_adam_enc_conv2_kernel_m_read_readvariableop0savev2_adam_enc_conv2_bias_m_read_readvariableop2savev2_adam_enc_conv3_kernel_m_read_readvariableop0savev2_adam_enc_conv3_bias_m_read_readvariableop2savev2_adam_enc_conv4_kernel_m_read_readvariableop0savev2_adam_enc_conv4_bias_m_read_readvariableop2savev2_adam_enc_fc_mu_kernel_m_read_readvariableop0savev2_adam_enc_fc_mu_bias_m_read_readvariableop7savev2_adam_enc_fc_log_var_kernel_m_read_readvariableop5savev2_adam_enc_fc_log_var_bias_m_read_readvariableop3savev2_adam_dec_dense1_kernel_m_read_readvariableop1savev2_adam_dec_dense1_bias_m_read_readvariableop4savev2_adam_dec_deconv1_kernel_m_read_readvariableop2savev2_adam_dec_deconv1_bias_m_read_readvariableop4savev2_adam_dec_deconv2_kernel_m_read_readvariableop2savev2_adam_dec_deconv2_bias_m_read_readvariableop4savev2_adam_dec_deconv3_kernel_m_read_readvariableop2savev2_adam_dec_deconv3_bias_m_read_readvariableop4savev2_adam_dec_deconv4_kernel_m_read_readvariableop2savev2_adam_dec_deconv4_bias_m_read_readvariableop2savev2_adam_enc_conv1_kernel_v_read_readvariableop0savev2_adam_enc_conv1_bias_v_read_readvariableop2savev2_adam_enc_conv2_kernel_v_read_readvariableop0savev2_adam_enc_conv2_bias_v_read_readvariableop2savev2_adam_enc_conv3_kernel_v_read_readvariableop0savev2_adam_enc_conv3_bias_v_read_readvariableop2savev2_adam_enc_conv4_kernel_v_read_readvariableop0savev2_adam_enc_conv4_bias_v_read_readvariableop2savev2_adam_enc_fc_mu_kernel_v_read_readvariableop0savev2_adam_enc_fc_mu_bias_v_read_readvariableop7savev2_adam_enc_fc_log_var_kernel_v_read_readvariableop5savev2_adam_enc_fc_log_var_bias_v_read_readvariableop3savev2_adam_dec_dense1_kernel_v_read_readvariableop1savev2_adam_dec_dense1_bias_v_read_readvariableop4savev2_adam_dec_deconv1_kernel_v_read_readvariableop2savev2_adam_dec_deconv1_bias_v_read_readvariableop4savev2_adam_dec_deconv2_kernel_v_read_readvariableop2savev2_adam_dec_deconv2_bias_v_read_readvariableop4savev2_adam_dec_deconv3_kernel_v_read_readvariableop2savev2_adam_dec_deconv3_bias_v_read_readvariableop4savev2_adam_dec_deconv4_kernel_v_read_readvariableop2savev2_adam_dec_deconv4_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *[
dtypesQ
O2M	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : : : @:@:@?:?:??:?:	? : :	? : :	 ?:?:??:?:@?:@: @: : :: : : : : : : : : @:@:@?:?:??:?:	? : :	? : :	 ?:?:??:?:@?:@: @: : :: : : @:@:@?:?:??:?:	? : :	? : :	 ?:?:??:?:@?:@: @: : :: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 	

_output_shapes
:@:-
)
'
_output_shapes
:@?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:%!

_output_shapes
:	? : 

_output_shapes
: :%!

_output_shapes
:	? : 

_output_shapes
: :%!

_output_shapes
:	 ?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:-)
'
_output_shapes
:@?: 

_output_shapes
:@:,(
&
_output_shapes
: @: 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :,"(
&
_output_shapes
: : #

_output_shapes
: :,$(
&
_output_shapes
: @: %

_output_shapes
:@:-&)
'
_output_shapes
:@?:!'

_output_shapes	
:?:.(*
(
_output_shapes
:??:!)

_output_shapes	
:?:%*!

_output_shapes
:	? : +

_output_shapes
: :%,!

_output_shapes
:	? : -

_output_shapes
: :%.!

_output_shapes
:	 ?:!/

_output_shapes	
:?:.0*
(
_output_shapes
:??:!1

_output_shapes	
:?:-2)
'
_output_shapes
:@?: 3

_output_shapes
:@:,4(
&
_output_shapes
: @: 5

_output_shapes
: :,6(
&
_output_shapes
: : 7

_output_shapes
::,8(
&
_output_shapes
: : 9

_output_shapes
: :,:(
&
_output_shapes
: @: ;

_output_shapes
:@:-<)
'
_output_shapes
:@?:!=

_output_shapes	
:?:.>*
(
_output_shapes
:??:!?

_output_shapes	
:?:%@!

_output_shapes
:	? : A

_output_shapes
: :%B!

_output_shapes
:	? : C

_output_shapes
: :%D!

_output_shapes
:	 ?:!E

_output_shapes	
:?:.F*
(
_output_shapes
:??:!G

_output_shapes	
:?:-H)
'
_output_shapes
:@?: I

_output_shapes
:@:,J(
&
_output_shapes
: @: K

_output_shapes
: :,L(
&
_output_shapes
: : M

_output_shapes
::N

_output_shapes
: 
?
D
(__inference_flatten_layer_call_fn_880250

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_8773842
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_1_layer_call_fn_877570
input_2
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_8775632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: 
?

?
+__inference_sequential_layer_call_fn_877465
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*(
_output_shapes
:??????????**
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_8774462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????@@::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?1
?
C__inference_gan_vae_layer_call_and_return_conditional_losses_878523

inputs
sequential_878460
sequential_878462
sequential_878464
sequential_878466
sequential_878468
sequential_878470
sequential_878472
sequential_878474
sequential_1_878477
sequential_1_878479
sequential_2_878482
sequential_2_878484
sequential_3_878498
sequential_3_878500
sequential_3_878502
sequential_3_878504
sequential_3_878506
sequential_3_878508
sequential_3_878510
sequential_3_878512
sequential_3_878514
sequential_3_878516
identity

identity_1??"sequential/StatefulPartitionedCall?$sequential_1/StatefulPartitionedCall?$sequential_2/StatefulPartitionedCall?$sequential_3/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_878460sequential_878462sequential_878464sequential_878466sequential_878468sequential_878470sequential_878472sequential_878474*
Tin
2	*
Tout
2*(
_output_shapes
:??????????**
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_8774922$
"sequential/StatefulPartitionedCall?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_878477sequential_1_878479*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_8775812&
$sequential_1/StatefulPartitionedCall?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_2_878482sequential_2_878484*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_8776582&
$sequential_2/StatefulPartitionedCallk
ShapeShape-sequential_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:????????? *
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:????????? 2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:????????? 2
random_normalS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/y?
mulMul-sequential_2/StatefulPartitionedCall:output:0mul/y:output:0*
T0*'
_output_shapes
:????????? 2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:????????? 2
Expc
mul_1Mulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:????????? 2
mul_1
addAddV2	mul_1:z:0-sequential_1/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:????????? 2
add?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCalladd:z:0sequential_3_878498sequential_3_878500sequential_3_878502sequential_3_878504sequential_3_878506sequential_3_878508sequential_3_878510sequential_3_878512sequential_3_878514sequential_3_878516*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_8780532&
$sequential_3/StatefulPartitionedCalle
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2-sequential_1/StatefulPartitionedCall:output:0-sequential_2/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@2
concat?
IdentityIdentityconcat:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity-sequential_3/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesu
s:?????????@@::::::::::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: 
??
?
C__inference_gan_vae_layer_call_and_return_conditional_losses_878846

inputs7
3sequential_enc_conv1_conv2d_readvariableop_resource8
4sequential_enc_conv1_biasadd_readvariableop_resource7
3sequential_enc_conv2_conv2d_readvariableop_resource8
4sequential_enc_conv2_biasadd_readvariableop_resource7
3sequential_enc_conv3_conv2d_readvariableop_resource8
4sequential_enc_conv3_biasadd_readvariableop_resource7
3sequential_enc_conv4_conv2d_readvariableop_resource8
4sequential_enc_conv4_biasadd_readvariableop_resource9
5sequential_1_enc_fc_mu_matmul_readvariableop_resource:
6sequential_1_enc_fc_mu_biasadd_readvariableop_resource>
:sequential_2_enc_fc_log_var_matmul_readvariableop_resource?
;sequential_2_enc_fc_log_var_biasadd_readvariableop_resource:
6sequential_3_dec_dense1_matmul_readvariableop_resource;
7sequential_3_dec_dense1_biasadd_readvariableop_resourceE
Asequential_3_dec_deconv1_conv2d_transpose_readvariableop_resource<
8sequential_3_dec_deconv1_biasadd_readvariableop_resourceE
Asequential_3_dec_deconv2_conv2d_transpose_readvariableop_resource<
8sequential_3_dec_deconv2_biasadd_readvariableop_resourceE
Asequential_3_dec_deconv3_conv2d_transpose_readvariableop_resource<
8sequential_3_dec_deconv3_biasadd_readvariableop_resourceE
Asequential_3_dec_deconv4_conv2d_transpose_readvariableop_resource<
8sequential_3_dec_deconv4_biasadd_readvariableop_resource
identity

identity_1??
*sequential/enc_conv1/Conv2D/ReadVariableOpReadVariableOp3sequential_enc_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*sequential/enc_conv1/Conv2D/ReadVariableOp?
sequential/enc_conv1/Conv2DConv2Dinputs2sequential/enc_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
sequential/enc_conv1/Conv2D?
+sequential/enc_conv1/BiasAdd/ReadVariableOpReadVariableOp4sequential_enc_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential/enc_conv1/BiasAdd/ReadVariableOp?
sequential/enc_conv1/BiasAddBiasAdd$sequential/enc_conv1/Conv2D:output:03sequential/enc_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential/enc_conv1/BiasAdd?
sequential/enc_conv1/ReluRelu%sequential/enc_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential/enc_conv1/Relu?
*sequential/enc_conv2/Conv2D/ReadVariableOpReadVariableOp3sequential_enc_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*sequential/enc_conv2/Conv2D/ReadVariableOp?
sequential/enc_conv2/Conv2DConv2D'sequential/enc_conv1/Relu:activations:02sequential/enc_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
sequential/enc_conv2/Conv2D?
+sequential/enc_conv2/BiasAdd/ReadVariableOpReadVariableOp4sequential_enc_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential/enc_conv2/BiasAdd/ReadVariableOp?
sequential/enc_conv2/BiasAddBiasAdd$sequential/enc_conv2/Conv2D:output:03sequential/enc_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential/enc_conv2/BiasAdd?
sequential/enc_conv2/ReluRelu%sequential/enc_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential/enc_conv2/Relu?
*sequential/enc_conv3/Conv2D/ReadVariableOpReadVariableOp3sequential_enc_conv3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02,
*sequential/enc_conv3/Conv2D/ReadVariableOp?
sequential/enc_conv3/Conv2DConv2D'sequential/enc_conv2/Relu:activations:02sequential/enc_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential/enc_conv3/Conv2D?
+sequential/enc_conv3/BiasAdd/ReadVariableOpReadVariableOp4sequential_enc_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential/enc_conv3/BiasAdd/ReadVariableOp?
sequential/enc_conv3/BiasAddBiasAdd$sequential/enc_conv3/Conv2D:output:03sequential/enc_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential/enc_conv3/BiasAdd?
sequential/enc_conv3/ReluRelu%sequential/enc_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential/enc_conv3/Relu?
*sequential/enc_conv4/Conv2D/ReadVariableOpReadVariableOp3sequential_enc_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*sequential/enc_conv4/Conv2D/ReadVariableOp?
sequential/enc_conv4/Conv2DConv2D'sequential/enc_conv3/Relu:activations:02sequential/enc_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential/enc_conv4/Conv2D?
+sequential/enc_conv4/BiasAdd/ReadVariableOpReadVariableOp4sequential_enc_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential/enc_conv4/BiasAdd/ReadVariableOp?
sequential/enc_conv4/BiasAddBiasAdd$sequential/enc_conv4/Conv2D:output:03sequential/enc_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential/enc_conv4/BiasAdd?
sequential/enc_conv4/ReluRelu%sequential/enc_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential/enc_conv4/Relu?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape'sequential/enc_conv4/Relu:activations:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/flatten/Reshape?
,sequential_1/enc_fc_mu/MatMul/ReadVariableOpReadVariableOp5sequential_1_enc_fc_mu_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02.
,sequential_1/enc_fc_mu/MatMul/ReadVariableOp?
sequential_1/enc_fc_mu/MatMulMatMul#sequential/flatten/Reshape:output:04sequential_1/enc_fc_mu/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_1/enc_fc_mu/MatMul?
-sequential_1/enc_fc_mu/BiasAdd/ReadVariableOpReadVariableOp6sequential_1_enc_fc_mu_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_1/enc_fc_mu/BiasAdd/ReadVariableOp?
sequential_1/enc_fc_mu/BiasAddBiasAdd'sequential_1/enc_fc_mu/MatMul:product:05sequential_1/enc_fc_mu/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2 
sequential_1/enc_fc_mu/BiasAdd?
1sequential_2/enc_fc_log_var/MatMul/ReadVariableOpReadVariableOp:sequential_2_enc_fc_log_var_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype023
1sequential_2/enc_fc_log_var/MatMul/ReadVariableOp?
"sequential_2/enc_fc_log_var/MatMulMatMul#sequential/flatten/Reshape:output:09sequential_2/enc_fc_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2$
"sequential_2/enc_fc_log_var/MatMul?
2sequential_2/enc_fc_log_var/BiasAdd/ReadVariableOpReadVariableOp;sequential_2_enc_fc_log_var_biasadd_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_2/enc_fc_log_var/BiasAdd/ReadVariableOp?
#sequential_2/enc_fc_log_var/BiasAddBiasAdd,sequential_2/enc_fc_log_var/MatMul:product:0:sequential_2/enc_fc_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2%
#sequential_2/enc_fc_log_var/BiasAdde
ShapeShape'sequential_1/enc_fc_mu/BiasAdd:output:0*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:????????? *
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:????????? 2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:????????? 2
random_normalS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/y?
mulMul,sequential_2/enc_fc_log_var/BiasAdd:output:0mul/y:output:0*
T0*'
_output_shapes
:????????? 2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:????????? 2
Expc
mul_1Mulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:????????? 2
mul_1y
addAddV2	mul_1:z:0'sequential_1/enc_fc_mu/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
add?
-sequential_3/dec_dense1/MatMul/ReadVariableOpReadVariableOp6sequential_3_dec_dense1_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02/
-sequential_3/dec_dense1/MatMul/ReadVariableOp?
sequential_3/dec_dense1/MatMulMatMuladd:z:05sequential_3/dec_dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_3/dec_dense1/MatMul?
.sequential_3/dec_dense1/BiasAdd/ReadVariableOpReadVariableOp7sequential_3_dec_dense1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.sequential_3/dec_dense1/BiasAdd/ReadVariableOp?
sequential_3/dec_dense1/BiasAddBiasAdd(sequential_3/dec_dense1/MatMul:product:06sequential_3/dec_dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
sequential_3/dec_dense1/BiasAdd?
sequential_3/dec_dense1/ReluRelu(sequential_3/dec_dense1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_3/dec_dense1/Relu?
sequential_3/reshape/ShapeShape*sequential_3/dec_dense1/Relu:activations:0*
T0*
_output_shapes
:2
sequential_3/reshape/Shape?
(sequential_3/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_3/reshape/strided_slice/stack?
*sequential_3/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_3/reshape/strided_slice/stack_1?
*sequential_3/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_3/reshape/strided_slice/stack_2?
"sequential_3/reshape/strided_sliceStridedSlice#sequential_3/reshape/Shape:output:01sequential_3/reshape/strided_slice/stack:output:03sequential_3/reshape/strided_slice/stack_1:output:03sequential_3/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_3/reshape/strided_slice?
$sequential_3/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_3/reshape/Reshape/shape/1?
$sequential_3/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_3/reshape/Reshape/shape/2?
$sequential_3/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2&
$sequential_3/reshape/Reshape/shape/3?
"sequential_3/reshape/Reshape/shapePack+sequential_3/reshape/strided_slice:output:0-sequential_3/reshape/Reshape/shape/1:output:0-sequential_3/reshape/Reshape/shape/2:output:0-sequential_3/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"sequential_3/reshape/Reshape/shape?
sequential_3/reshape/ReshapeReshape*sequential_3/dec_dense1/Relu:activations:0+sequential_3/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
sequential_3/reshape/Reshape?
sequential_3/dec_deconv1/ShapeShape%sequential_3/reshape/Reshape:output:0*
T0*
_output_shapes
:2 
sequential_3/dec_deconv1/Shape?
,sequential_3/dec_deconv1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_3/dec_deconv1/strided_slice/stack?
.sequential_3/dec_deconv1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv1/strided_slice/stack_1?
.sequential_3/dec_deconv1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv1/strided_slice/stack_2?
&sequential_3/dec_deconv1/strided_sliceStridedSlice'sequential_3/dec_deconv1/Shape:output:05sequential_3/dec_deconv1/strided_slice/stack:output:07sequential_3/dec_deconv1/strided_slice/stack_1:output:07sequential_3/dec_deconv1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_3/dec_deconv1/strided_slice?
.sequential_3/dec_deconv1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv1/strided_slice_1/stack?
0sequential_3/dec_deconv1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_1/stack_1?
0sequential_3/dec_deconv1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_1/stack_2?
(sequential_3/dec_deconv1/strided_slice_1StridedSlice'sequential_3/dec_deconv1/Shape:output:07sequential_3/dec_deconv1/strided_slice_1/stack:output:09sequential_3/dec_deconv1/strided_slice_1/stack_1:output:09sequential_3/dec_deconv1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv1/strided_slice_1?
.sequential_3/dec_deconv1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv1/strided_slice_2/stack?
0sequential_3/dec_deconv1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_2/stack_1?
0sequential_3/dec_deconv1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_2/stack_2?
(sequential_3/dec_deconv1/strided_slice_2StridedSlice'sequential_3/dec_deconv1/Shape:output:07sequential_3/dec_deconv1/strided_slice_2/stack:output:09sequential_3/dec_deconv1/strided_slice_2/stack_1:output:09sequential_3/dec_deconv1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv1/strided_slice_2?
sequential_3/dec_deconv1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv1/mul/y?
sequential_3/dec_deconv1/mulMul1sequential_3/dec_deconv1/strided_slice_1:output:0'sequential_3/dec_deconv1/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv1/mul?
sequential_3/dec_deconv1/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv1/add/y?
sequential_3/dec_deconv1/addAddV2 sequential_3/dec_deconv1/mul:z:0'sequential_3/dec_deconv1/add/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv1/add?
 sequential_3/dec_deconv1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv1/mul_1/y?
sequential_3/dec_deconv1/mul_1Mul1sequential_3/dec_deconv1/strided_slice_2:output:0)sequential_3/dec_deconv1/mul_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv1/mul_1?
 sequential_3/dec_deconv1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv1/add_1/y?
sequential_3/dec_deconv1/add_1AddV2"sequential_3/dec_deconv1/mul_1:z:0)sequential_3/dec_deconv1/add_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv1/add_1?
 sequential_3/dec_deconv1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2"
 sequential_3/dec_deconv1/stack/3?
sequential_3/dec_deconv1/stackPack/sequential_3/dec_deconv1/strided_slice:output:0 sequential_3/dec_deconv1/add:z:0"sequential_3/dec_deconv1/add_1:z:0)sequential_3/dec_deconv1/stack/3:output:0*
N*
T0*
_output_shapes
:2 
sequential_3/dec_deconv1/stack?
.sequential_3/dec_deconv1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_3/dec_deconv1/strided_slice_3/stack?
0sequential_3/dec_deconv1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_3/stack_1?
0sequential_3/dec_deconv1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_3/stack_2?
(sequential_3/dec_deconv1/strided_slice_3StridedSlice'sequential_3/dec_deconv1/stack:output:07sequential_3/dec_deconv1/strided_slice_3/stack:output:09sequential_3/dec_deconv1/strided_slice_3/stack_1:output:09sequential_3/dec_deconv1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv1/strided_slice_3?
8sequential_3/dec_deconv1/conv2d_transpose/ReadVariableOpReadVariableOpAsequential_3_dec_deconv1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02:
8sequential_3/dec_deconv1/conv2d_transpose/ReadVariableOp?
)sequential_3/dec_deconv1/conv2d_transposeConv2DBackpropInput'sequential_3/dec_deconv1/stack:output:0@sequential_3/dec_deconv1/conv2d_transpose/ReadVariableOp:value:0%sequential_3/reshape/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2+
)sequential_3/dec_deconv1/conv2d_transpose?
/sequential_3/dec_deconv1/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dec_deconv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/sequential_3/dec_deconv1/BiasAdd/ReadVariableOp?
 sequential_3/dec_deconv1/BiasAddBiasAdd2sequential_3/dec_deconv1/conv2d_transpose:output:07sequential_3/dec_deconv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2"
 sequential_3/dec_deconv1/BiasAdd?
sequential_3/dec_deconv1/ReluRelu)sequential_3/dec_deconv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential_3/dec_deconv1/Relu?
sequential_3/dec_deconv2/ShapeShape+sequential_3/dec_deconv1/Relu:activations:0*
T0*
_output_shapes
:2 
sequential_3/dec_deconv2/Shape?
,sequential_3/dec_deconv2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_3/dec_deconv2/strided_slice/stack?
.sequential_3/dec_deconv2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv2/strided_slice/stack_1?
.sequential_3/dec_deconv2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv2/strided_slice/stack_2?
&sequential_3/dec_deconv2/strided_sliceStridedSlice'sequential_3/dec_deconv2/Shape:output:05sequential_3/dec_deconv2/strided_slice/stack:output:07sequential_3/dec_deconv2/strided_slice/stack_1:output:07sequential_3/dec_deconv2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_3/dec_deconv2/strided_slice?
.sequential_3/dec_deconv2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv2/strided_slice_1/stack?
0sequential_3/dec_deconv2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_1/stack_1?
0sequential_3/dec_deconv2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_1/stack_2?
(sequential_3/dec_deconv2/strided_slice_1StridedSlice'sequential_3/dec_deconv2/Shape:output:07sequential_3/dec_deconv2/strided_slice_1/stack:output:09sequential_3/dec_deconv2/strided_slice_1/stack_1:output:09sequential_3/dec_deconv2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv2/strided_slice_1?
.sequential_3/dec_deconv2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv2/strided_slice_2/stack?
0sequential_3/dec_deconv2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_2/stack_1?
0sequential_3/dec_deconv2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_2/stack_2?
(sequential_3/dec_deconv2/strided_slice_2StridedSlice'sequential_3/dec_deconv2/Shape:output:07sequential_3/dec_deconv2/strided_slice_2/stack:output:09sequential_3/dec_deconv2/strided_slice_2/stack_1:output:09sequential_3/dec_deconv2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv2/strided_slice_2?
sequential_3/dec_deconv2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv2/mul/y?
sequential_3/dec_deconv2/mulMul1sequential_3/dec_deconv2/strided_slice_1:output:0'sequential_3/dec_deconv2/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv2/mul?
sequential_3/dec_deconv2/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv2/add/y?
sequential_3/dec_deconv2/addAddV2 sequential_3/dec_deconv2/mul:z:0'sequential_3/dec_deconv2/add/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv2/add?
 sequential_3/dec_deconv2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv2/mul_1/y?
sequential_3/dec_deconv2/mul_1Mul1sequential_3/dec_deconv2/strided_slice_2:output:0)sequential_3/dec_deconv2/mul_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv2/mul_1?
 sequential_3/dec_deconv2/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv2/add_1/y?
sequential_3/dec_deconv2/add_1AddV2"sequential_3/dec_deconv2/mul_1:z:0)sequential_3/dec_deconv2/add_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv2/add_1?
 sequential_3/dec_deconv2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2"
 sequential_3/dec_deconv2/stack/3?
sequential_3/dec_deconv2/stackPack/sequential_3/dec_deconv2/strided_slice:output:0 sequential_3/dec_deconv2/add:z:0"sequential_3/dec_deconv2/add_1:z:0)sequential_3/dec_deconv2/stack/3:output:0*
N*
T0*
_output_shapes
:2 
sequential_3/dec_deconv2/stack?
.sequential_3/dec_deconv2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_3/dec_deconv2/strided_slice_3/stack?
0sequential_3/dec_deconv2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_3/stack_1?
0sequential_3/dec_deconv2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_3/stack_2?
(sequential_3/dec_deconv2/strided_slice_3StridedSlice'sequential_3/dec_deconv2/stack:output:07sequential_3/dec_deconv2/strided_slice_3/stack:output:09sequential_3/dec_deconv2/strided_slice_3/stack_1:output:09sequential_3/dec_deconv2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv2/strided_slice_3?
8sequential_3/dec_deconv2/conv2d_transpose/ReadVariableOpReadVariableOpAsequential_3_dec_deconv2_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02:
8sequential_3/dec_deconv2/conv2d_transpose/ReadVariableOp?
)sequential_3/dec_deconv2/conv2d_transposeConv2DBackpropInput'sequential_3/dec_deconv2/stack:output:0@sequential_3/dec_deconv2/conv2d_transpose/ReadVariableOp:value:0+sequential_3/dec_deconv1/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2+
)sequential_3/dec_deconv2/conv2d_transpose?
/sequential_3/dec_deconv2/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dec_deconv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/sequential_3/dec_deconv2/BiasAdd/ReadVariableOp?
 sequential_3/dec_deconv2/BiasAddBiasAdd2sequential_3/dec_deconv2/conv2d_transpose:output:07sequential_3/dec_deconv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2"
 sequential_3/dec_deconv2/BiasAdd?
sequential_3/dec_deconv2/ReluRelu)sequential_3/dec_deconv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_3/dec_deconv2/Relu?
sequential_3/dec_deconv3/ShapeShape+sequential_3/dec_deconv2/Relu:activations:0*
T0*
_output_shapes
:2 
sequential_3/dec_deconv3/Shape?
,sequential_3/dec_deconv3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_3/dec_deconv3/strided_slice/stack?
.sequential_3/dec_deconv3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv3/strided_slice/stack_1?
.sequential_3/dec_deconv3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv3/strided_slice/stack_2?
&sequential_3/dec_deconv3/strided_sliceStridedSlice'sequential_3/dec_deconv3/Shape:output:05sequential_3/dec_deconv3/strided_slice/stack:output:07sequential_3/dec_deconv3/strided_slice/stack_1:output:07sequential_3/dec_deconv3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_3/dec_deconv3/strided_slice?
.sequential_3/dec_deconv3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv3/strided_slice_1/stack?
0sequential_3/dec_deconv3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_1/stack_1?
0sequential_3/dec_deconv3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_1/stack_2?
(sequential_3/dec_deconv3/strided_slice_1StridedSlice'sequential_3/dec_deconv3/Shape:output:07sequential_3/dec_deconv3/strided_slice_1/stack:output:09sequential_3/dec_deconv3/strided_slice_1/stack_1:output:09sequential_3/dec_deconv3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv3/strided_slice_1?
.sequential_3/dec_deconv3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv3/strided_slice_2/stack?
0sequential_3/dec_deconv3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_2/stack_1?
0sequential_3/dec_deconv3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_2/stack_2?
(sequential_3/dec_deconv3/strided_slice_2StridedSlice'sequential_3/dec_deconv3/Shape:output:07sequential_3/dec_deconv3/strided_slice_2/stack:output:09sequential_3/dec_deconv3/strided_slice_2/stack_1:output:09sequential_3/dec_deconv3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv3/strided_slice_2?
sequential_3/dec_deconv3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv3/mul/y?
sequential_3/dec_deconv3/mulMul1sequential_3/dec_deconv3/strided_slice_1:output:0'sequential_3/dec_deconv3/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv3/mul?
sequential_3/dec_deconv3/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv3/add/y?
sequential_3/dec_deconv3/addAddV2 sequential_3/dec_deconv3/mul:z:0'sequential_3/dec_deconv3/add/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv3/add?
 sequential_3/dec_deconv3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv3/mul_1/y?
sequential_3/dec_deconv3/mul_1Mul1sequential_3/dec_deconv3/strided_slice_2:output:0)sequential_3/dec_deconv3/mul_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv3/mul_1?
 sequential_3/dec_deconv3/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv3/add_1/y?
sequential_3/dec_deconv3/add_1AddV2"sequential_3/dec_deconv3/mul_1:z:0)sequential_3/dec_deconv3/add_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv3/add_1?
 sequential_3/dec_deconv3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2"
 sequential_3/dec_deconv3/stack/3?
sequential_3/dec_deconv3/stackPack/sequential_3/dec_deconv3/strided_slice:output:0 sequential_3/dec_deconv3/add:z:0"sequential_3/dec_deconv3/add_1:z:0)sequential_3/dec_deconv3/stack/3:output:0*
N*
T0*
_output_shapes
:2 
sequential_3/dec_deconv3/stack?
.sequential_3/dec_deconv3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_3/dec_deconv3/strided_slice_3/stack?
0sequential_3/dec_deconv3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_3/stack_1?
0sequential_3/dec_deconv3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_3/stack_2?
(sequential_3/dec_deconv3/strided_slice_3StridedSlice'sequential_3/dec_deconv3/stack:output:07sequential_3/dec_deconv3/strided_slice_3/stack:output:09sequential_3/dec_deconv3/strided_slice_3/stack_1:output:09sequential_3/dec_deconv3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv3/strided_slice_3?
8sequential_3/dec_deconv3/conv2d_transpose/ReadVariableOpReadVariableOpAsequential_3_dec_deconv3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02:
8sequential_3/dec_deconv3/conv2d_transpose/ReadVariableOp?
)sequential_3/dec_deconv3/conv2d_transposeConv2DBackpropInput'sequential_3/dec_deconv3/stack:output:0@sequential_3/dec_deconv3/conv2d_transpose/ReadVariableOp:value:0+sequential_3/dec_deconv2/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2+
)sequential_3/dec_deconv3/conv2d_transpose?
/sequential_3/dec_deconv3/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dec_deconv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential_3/dec_deconv3/BiasAdd/ReadVariableOp?
 sequential_3/dec_deconv3/BiasAddBiasAdd2sequential_3/dec_deconv3/conv2d_transpose:output:07sequential_3/dec_deconv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2"
 sequential_3/dec_deconv3/BiasAdd?
sequential_3/dec_deconv3/ReluRelu)sequential_3/dec_deconv3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_3/dec_deconv3/Relu?
sequential_3/dec_deconv4/ShapeShape+sequential_3/dec_deconv3/Relu:activations:0*
T0*
_output_shapes
:2 
sequential_3/dec_deconv4/Shape?
,sequential_3/dec_deconv4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_3/dec_deconv4/strided_slice/stack?
.sequential_3/dec_deconv4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv4/strided_slice/stack_1?
.sequential_3/dec_deconv4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv4/strided_slice/stack_2?
&sequential_3/dec_deconv4/strided_sliceStridedSlice'sequential_3/dec_deconv4/Shape:output:05sequential_3/dec_deconv4/strided_slice/stack:output:07sequential_3/dec_deconv4/strided_slice/stack_1:output:07sequential_3/dec_deconv4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_3/dec_deconv4/strided_slice?
.sequential_3/dec_deconv4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv4/strided_slice_1/stack?
0sequential_3/dec_deconv4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_1/stack_1?
0sequential_3/dec_deconv4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_1/stack_2?
(sequential_3/dec_deconv4/strided_slice_1StridedSlice'sequential_3/dec_deconv4/Shape:output:07sequential_3/dec_deconv4/strided_slice_1/stack:output:09sequential_3/dec_deconv4/strided_slice_1/stack_1:output:09sequential_3/dec_deconv4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv4/strided_slice_1?
.sequential_3/dec_deconv4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv4/strided_slice_2/stack?
0sequential_3/dec_deconv4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_2/stack_1?
0sequential_3/dec_deconv4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_2/stack_2?
(sequential_3/dec_deconv4/strided_slice_2StridedSlice'sequential_3/dec_deconv4/Shape:output:07sequential_3/dec_deconv4/strided_slice_2/stack:output:09sequential_3/dec_deconv4/strided_slice_2/stack_1:output:09sequential_3/dec_deconv4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv4/strided_slice_2?
sequential_3/dec_deconv4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv4/mul/y?
sequential_3/dec_deconv4/mulMul1sequential_3/dec_deconv4/strided_slice_1:output:0'sequential_3/dec_deconv4/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv4/mul?
sequential_3/dec_deconv4/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv4/add/y?
sequential_3/dec_deconv4/addAddV2 sequential_3/dec_deconv4/mul:z:0'sequential_3/dec_deconv4/add/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv4/add?
 sequential_3/dec_deconv4/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv4/mul_1/y?
sequential_3/dec_deconv4/mul_1Mul1sequential_3/dec_deconv4/strided_slice_2:output:0)sequential_3/dec_deconv4/mul_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv4/mul_1?
 sequential_3/dec_deconv4/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv4/add_1/y?
sequential_3/dec_deconv4/add_1AddV2"sequential_3/dec_deconv4/mul_1:z:0)sequential_3/dec_deconv4/add_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv4/add_1?
 sequential_3/dec_deconv4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv4/stack/3?
sequential_3/dec_deconv4/stackPack/sequential_3/dec_deconv4/strided_slice:output:0 sequential_3/dec_deconv4/add:z:0"sequential_3/dec_deconv4/add_1:z:0)sequential_3/dec_deconv4/stack/3:output:0*
N*
T0*
_output_shapes
:2 
sequential_3/dec_deconv4/stack?
.sequential_3/dec_deconv4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_3/dec_deconv4/strided_slice_3/stack?
0sequential_3/dec_deconv4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_3/stack_1?
0sequential_3/dec_deconv4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_3/stack_2?
(sequential_3/dec_deconv4/strided_slice_3StridedSlice'sequential_3/dec_deconv4/stack:output:07sequential_3/dec_deconv4/strided_slice_3/stack:output:09sequential_3/dec_deconv4/strided_slice_3/stack_1:output:09sequential_3/dec_deconv4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv4/strided_slice_3?
8sequential_3/dec_deconv4/conv2d_transpose/ReadVariableOpReadVariableOpAsequential_3_dec_deconv4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02:
8sequential_3/dec_deconv4/conv2d_transpose/ReadVariableOp?
)sequential_3/dec_deconv4/conv2d_transposeConv2DBackpropInput'sequential_3/dec_deconv4/stack:output:0@sequential_3/dec_deconv4/conv2d_transpose/ReadVariableOp:value:0+sequential_3/dec_deconv3/Relu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingVALID*
strides
2+
)sequential_3/dec_deconv4/conv2d_transpose?
/sequential_3/dec_deconv4/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dec_deconv4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_3/dec_deconv4/BiasAdd/ReadVariableOp?
 sequential_3/dec_deconv4/BiasAddBiasAdd2sequential_3/dec_deconv4/conv2d_transpose:output:07sequential_3/dec_deconv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2"
 sequential_3/dec_deconv4/BiasAdd?
 sequential_3/dec_deconv4/SigmoidSigmoid)sequential_3/dec_deconv4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2"
 sequential_3/dec_deconv4/Sigmoide
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2'sequential_1/enc_fc_mu/BiasAdd:output:0,sequential_2/enc_fc_log_var/BiasAdd:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity$sequential_3/dec_deconv4/Sigmoid:y:0*
T0*/
_output_shapes
:?????????@@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesu
s:?????????@@:::::::::::::::::::::::W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: 
?

*__inference_enc_conv2_layer_call_fn_877310

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_conv2_layer_call_and_return_conditional_losses_8773002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?

?
H__inference_sequential_2_layer_call_and_return_conditional_losses_879847

inputs1
-enc_fc_log_var_matmul_readvariableop_resource2
.enc_fc_log_var_biasadd_readvariableop_resource
identity??
$enc_fc_log_var/MatMul/ReadVariableOpReadVariableOp-enc_fc_log_var_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02&
$enc_fc_log_var/MatMul/ReadVariableOp?
enc_fc_log_var/MatMulMatMulinputs,enc_fc_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
enc_fc_log_var/MatMul?
%enc_fc_log_var/BiasAdd/ReadVariableOpReadVariableOp.enc_fc_log_var_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%enc_fc_log_var/BiasAdd/ReadVariableOp?
enc_fc_log_var/BiasAddBiasAddenc_fc_log_var/MatMul:product:0-enc_fc_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
enc_fc_log_var/BiasAdds
IdentityIdentityenc_fc_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
-__inference_sequential_3_layer_call_fn_878021
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_8779982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:????????? ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:????????? 
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?

?
E__inference_enc_conv2_layer_call_and_return_conditional_losses_877300

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? :::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?

?
+__inference_sequential_layer_call_fn_879799

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*(
_output_shapes
:??????????**
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_8774922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????@@::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_dec_dense1_layer_call_fn_880308

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_dec_dense1_layer_call_and_return_conditional_losses_8778762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?

?
+__inference_sequential_layer_call_fn_879778

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*(
_output_shapes
:??????????**
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_8774462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????@@::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_dec_dense1_layer_call_and_return_conditional_losses_880299

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :::O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_877418
input_1
enc_conv1_877396
enc_conv1_877398
enc_conv2_877401
enc_conv2_877403
enc_conv3_877406
enc_conv3_877408
enc_conv4_877411
enc_conv4_877413
identity??!enc_conv1/StatefulPartitionedCall?!enc_conv2/StatefulPartitionedCall?!enc_conv3/StatefulPartitionedCall?!enc_conv4/StatefulPartitionedCall?
!enc_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1enc_conv1_877396enc_conv1_877398*
Tin
2*
Tout
2*/
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_conv1_layer_call_and_return_conditional_losses_8772782#
!enc_conv1/StatefulPartitionedCall?
!enc_conv2/StatefulPartitionedCallStatefulPartitionedCall*enc_conv1/StatefulPartitionedCall:output:0enc_conv2_877401enc_conv2_877403*
Tin
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_conv2_layer_call_and_return_conditional_losses_8773002#
!enc_conv2/StatefulPartitionedCall?
!enc_conv3/StatefulPartitionedCallStatefulPartitionedCall*enc_conv2/StatefulPartitionedCall:output:0enc_conv3_877406enc_conv3_877408*
Tin
2*
Tout
2*0
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_conv3_layer_call_and_return_conditional_losses_8773222#
!enc_conv3/StatefulPartitionedCall?
!enc_conv4/StatefulPartitionedCallStatefulPartitionedCall*enc_conv3/StatefulPartitionedCall:output:0enc_conv4_877411enc_conv4_877413*
Tin
2*
Tout
2*0
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_conv4_layer_call_and_return_conditional_losses_8773442#
!enc_conv4/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall*enc_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_8773842
flatten/PartitionedCall?
IdentityIdentity flatten/PartitionedCall:output:0"^enc_conv1/StatefulPartitionedCall"^enc_conv2/StatefulPartitionedCall"^enc_conv3/StatefulPartitionedCall"^enc_conv4/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????@@::::::::2F
!enc_conv1/StatefulPartitionedCall!enc_conv1/StatefulPartitionedCall2F
!enc_conv2/StatefulPartitionedCall!enc_conv2/StatefulPartitionedCall2F
!enc_conv3/StatefulPartitionedCall!enc_conv3/StatefulPartitionedCall2F
!enc_conv4/StatefulPartitionedCall!enc_conv4/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
C__inference_gan_vae_layer_call_and_return_conditional_losses_879374
input_17
3sequential_enc_conv1_conv2d_readvariableop_resource8
4sequential_enc_conv1_biasadd_readvariableop_resource7
3sequential_enc_conv2_conv2d_readvariableop_resource8
4sequential_enc_conv2_biasadd_readvariableop_resource7
3sequential_enc_conv3_conv2d_readvariableop_resource8
4sequential_enc_conv3_biasadd_readvariableop_resource7
3sequential_enc_conv4_conv2d_readvariableop_resource8
4sequential_enc_conv4_biasadd_readvariableop_resource9
5sequential_1_enc_fc_mu_matmul_readvariableop_resource:
6sequential_1_enc_fc_mu_biasadd_readvariableop_resource>
:sequential_2_enc_fc_log_var_matmul_readvariableop_resource?
;sequential_2_enc_fc_log_var_biasadd_readvariableop_resource:
6sequential_3_dec_dense1_matmul_readvariableop_resource;
7sequential_3_dec_dense1_biasadd_readvariableop_resourceE
Asequential_3_dec_deconv1_conv2d_transpose_readvariableop_resource<
8sequential_3_dec_deconv1_biasadd_readvariableop_resourceE
Asequential_3_dec_deconv2_conv2d_transpose_readvariableop_resource<
8sequential_3_dec_deconv2_biasadd_readvariableop_resourceE
Asequential_3_dec_deconv3_conv2d_transpose_readvariableop_resource<
8sequential_3_dec_deconv3_biasadd_readvariableop_resourceE
Asequential_3_dec_deconv4_conv2d_transpose_readvariableop_resource<
8sequential_3_dec_deconv4_biasadd_readvariableop_resource
identity

identity_1??
*sequential/enc_conv1/Conv2D/ReadVariableOpReadVariableOp3sequential_enc_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*sequential/enc_conv1/Conv2D/ReadVariableOp?
sequential/enc_conv1/Conv2DConv2Dinput_12sequential/enc_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
sequential/enc_conv1/Conv2D?
+sequential/enc_conv1/BiasAdd/ReadVariableOpReadVariableOp4sequential_enc_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential/enc_conv1/BiasAdd/ReadVariableOp?
sequential/enc_conv1/BiasAddBiasAdd$sequential/enc_conv1/Conv2D:output:03sequential/enc_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential/enc_conv1/BiasAdd?
sequential/enc_conv1/ReluRelu%sequential/enc_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential/enc_conv1/Relu?
*sequential/enc_conv2/Conv2D/ReadVariableOpReadVariableOp3sequential_enc_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*sequential/enc_conv2/Conv2D/ReadVariableOp?
sequential/enc_conv2/Conv2DConv2D'sequential/enc_conv1/Relu:activations:02sequential/enc_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
sequential/enc_conv2/Conv2D?
+sequential/enc_conv2/BiasAdd/ReadVariableOpReadVariableOp4sequential_enc_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential/enc_conv2/BiasAdd/ReadVariableOp?
sequential/enc_conv2/BiasAddBiasAdd$sequential/enc_conv2/Conv2D:output:03sequential/enc_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential/enc_conv2/BiasAdd?
sequential/enc_conv2/ReluRelu%sequential/enc_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential/enc_conv2/Relu?
*sequential/enc_conv3/Conv2D/ReadVariableOpReadVariableOp3sequential_enc_conv3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02,
*sequential/enc_conv3/Conv2D/ReadVariableOp?
sequential/enc_conv3/Conv2DConv2D'sequential/enc_conv2/Relu:activations:02sequential/enc_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential/enc_conv3/Conv2D?
+sequential/enc_conv3/BiasAdd/ReadVariableOpReadVariableOp4sequential_enc_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential/enc_conv3/BiasAdd/ReadVariableOp?
sequential/enc_conv3/BiasAddBiasAdd$sequential/enc_conv3/Conv2D:output:03sequential/enc_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential/enc_conv3/BiasAdd?
sequential/enc_conv3/ReluRelu%sequential/enc_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential/enc_conv3/Relu?
*sequential/enc_conv4/Conv2D/ReadVariableOpReadVariableOp3sequential_enc_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*sequential/enc_conv4/Conv2D/ReadVariableOp?
sequential/enc_conv4/Conv2DConv2D'sequential/enc_conv3/Relu:activations:02sequential/enc_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential/enc_conv4/Conv2D?
+sequential/enc_conv4/BiasAdd/ReadVariableOpReadVariableOp4sequential_enc_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential/enc_conv4/BiasAdd/ReadVariableOp?
sequential/enc_conv4/BiasAddBiasAdd$sequential/enc_conv4/Conv2D:output:03sequential/enc_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential/enc_conv4/BiasAdd?
sequential/enc_conv4/ReluRelu%sequential/enc_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential/enc_conv4/Relu?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape'sequential/enc_conv4/Relu:activations:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/flatten/Reshape?
,sequential_1/enc_fc_mu/MatMul/ReadVariableOpReadVariableOp5sequential_1_enc_fc_mu_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02.
,sequential_1/enc_fc_mu/MatMul/ReadVariableOp?
sequential_1/enc_fc_mu/MatMulMatMul#sequential/flatten/Reshape:output:04sequential_1/enc_fc_mu/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_1/enc_fc_mu/MatMul?
-sequential_1/enc_fc_mu/BiasAdd/ReadVariableOpReadVariableOp6sequential_1_enc_fc_mu_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_1/enc_fc_mu/BiasAdd/ReadVariableOp?
sequential_1/enc_fc_mu/BiasAddBiasAdd'sequential_1/enc_fc_mu/MatMul:product:05sequential_1/enc_fc_mu/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2 
sequential_1/enc_fc_mu/BiasAdd?
1sequential_2/enc_fc_log_var/MatMul/ReadVariableOpReadVariableOp:sequential_2_enc_fc_log_var_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype023
1sequential_2/enc_fc_log_var/MatMul/ReadVariableOp?
"sequential_2/enc_fc_log_var/MatMulMatMul#sequential/flatten/Reshape:output:09sequential_2/enc_fc_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2$
"sequential_2/enc_fc_log_var/MatMul?
2sequential_2/enc_fc_log_var/BiasAdd/ReadVariableOpReadVariableOp;sequential_2_enc_fc_log_var_biasadd_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_2/enc_fc_log_var/BiasAdd/ReadVariableOp?
#sequential_2/enc_fc_log_var/BiasAddBiasAdd,sequential_2/enc_fc_log_var/MatMul:product:0:sequential_2/enc_fc_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2%
#sequential_2/enc_fc_log_var/BiasAdde
ShapeShape'sequential_1/enc_fc_mu/BiasAdd:output:0*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:????????? *
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:????????? 2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:????????? 2
random_normalS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/y?
mulMul,sequential_2/enc_fc_log_var/BiasAdd:output:0mul/y:output:0*
T0*'
_output_shapes
:????????? 2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:????????? 2
Expc
mul_1Mulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:????????? 2
mul_1y
addAddV2	mul_1:z:0'sequential_1/enc_fc_mu/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
add?
-sequential_3/dec_dense1/MatMul/ReadVariableOpReadVariableOp6sequential_3_dec_dense1_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02/
-sequential_3/dec_dense1/MatMul/ReadVariableOp?
sequential_3/dec_dense1/MatMulMatMuladd:z:05sequential_3/dec_dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_3/dec_dense1/MatMul?
.sequential_3/dec_dense1/BiasAdd/ReadVariableOpReadVariableOp7sequential_3_dec_dense1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.sequential_3/dec_dense1/BiasAdd/ReadVariableOp?
sequential_3/dec_dense1/BiasAddBiasAdd(sequential_3/dec_dense1/MatMul:product:06sequential_3/dec_dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
sequential_3/dec_dense1/BiasAdd?
sequential_3/dec_dense1/ReluRelu(sequential_3/dec_dense1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_3/dec_dense1/Relu?
sequential_3/reshape/ShapeShape*sequential_3/dec_dense1/Relu:activations:0*
T0*
_output_shapes
:2
sequential_3/reshape/Shape?
(sequential_3/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_3/reshape/strided_slice/stack?
*sequential_3/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_3/reshape/strided_slice/stack_1?
*sequential_3/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_3/reshape/strided_slice/stack_2?
"sequential_3/reshape/strided_sliceStridedSlice#sequential_3/reshape/Shape:output:01sequential_3/reshape/strided_slice/stack:output:03sequential_3/reshape/strided_slice/stack_1:output:03sequential_3/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_3/reshape/strided_slice?
$sequential_3/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_3/reshape/Reshape/shape/1?
$sequential_3/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_3/reshape/Reshape/shape/2?
$sequential_3/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2&
$sequential_3/reshape/Reshape/shape/3?
"sequential_3/reshape/Reshape/shapePack+sequential_3/reshape/strided_slice:output:0-sequential_3/reshape/Reshape/shape/1:output:0-sequential_3/reshape/Reshape/shape/2:output:0-sequential_3/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"sequential_3/reshape/Reshape/shape?
sequential_3/reshape/ReshapeReshape*sequential_3/dec_dense1/Relu:activations:0+sequential_3/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
sequential_3/reshape/Reshape?
sequential_3/dec_deconv1/ShapeShape%sequential_3/reshape/Reshape:output:0*
T0*
_output_shapes
:2 
sequential_3/dec_deconv1/Shape?
,sequential_3/dec_deconv1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_3/dec_deconv1/strided_slice/stack?
.sequential_3/dec_deconv1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv1/strided_slice/stack_1?
.sequential_3/dec_deconv1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv1/strided_slice/stack_2?
&sequential_3/dec_deconv1/strided_sliceStridedSlice'sequential_3/dec_deconv1/Shape:output:05sequential_3/dec_deconv1/strided_slice/stack:output:07sequential_3/dec_deconv1/strided_slice/stack_1:output:07sequential_3/dec_deconv1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_3/dec_deconv1/strided_slice?
.sequential_3/dec_deconv1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv1/strided_slice_1/stack?
0sequential_3/dec_deconv1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_1/stack_1?
0sequential_3/dec_deconv1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_1/stack_2?
(sequential_3/dec_deconv1/strided_slice_1StridedSlice'sequential_3/dec_deconv1/Shape:output:07sequential_3/dec_deconv1/strided_slice_1/stack:output:09sequential_3/dec_deconv1/strided_slice_1/stack_1:output:09sequential_3/dec_deconv1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv1/strided_slice_1?
.sequential_3/dec_deconv1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv1/strided_slice_2/stack?
0sequential_3/dec_deconv1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_2/stack_1?
0sequential_3/dec_deconv1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_2/stack_2?
(sequential_3/dec_deconv1/strided_slice_2StridedSlice'sequential_3/dec_deconv1/Shape:output:07sequential_3/dec_deconv1/strided_slice_2/stack:output:09sequential_3/dec_deconv1/strided_slice_2/stack_1:output:09sequential_3/dec_deconv1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv1/strided_slice_2?
sequential_3/dec_deconv1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv1/mul/y?
sequential_3/dec_deconv1/mulMul1sequential_3/dec_deconv1/strided_slice_1:output:0'sequential_3/dec_deconv1/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv1/mul?
sequential_3/dec_deconv1/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv1/add/y?
sequential_3/dec_deconv1/addAddV2 sequential_3/dec_deconv1/mul:z:0'sequential_3/dec_deconv1/add/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv1/add?
 sequential_3/dec_deconv1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv1/mul_1/y?
sequential_3/dec_deconv1/mul_1Mul1sequential_3/dec_deconv1/strided_slice_2:output:0)sequential_3/dec_deconv1/mul_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv1/mul_1?
 sequential_3/dec_deconv1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv1/add_1/y?
sequential_3/dec_deconv1/add_1AddV2"sequential_3/dec_deconv1/mul_1:z:0)sequential_3/dec_deconv1/add_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv1/add_1?
 sequential_3/dec_deconv1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2"
 sequential_3/dec_deconv1/stack/3?
sequential_3/dec_deconv1/stackPack/sequential_3/dec_deconv1/strided_slice:output:0 sequential_3/dec_deconv1/add:z:0"sequential_3/dec_deconv1/add_1:z:0)sequential_3/dec_deconv1/stack/3:output:0*
N*
T0*
_output_shapes
:2 
sequential_3/dec_deconv1/stack?
.sequential_3/dec_deconv1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_3/dec_deconv1/strided_slice_3/stack?
0sequential_3/dec_deconv1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_3/stack_1?
0sequential_3/dec_deconv1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv1/strided_slice_3/stack_2?
(sequential_3/dec_deconv1/strided_slice_3StridedSlice'sequential_3/dec_deconv1/stack:output:07sequential_3/dec_deconv1/strided_slice_3/stack:output:09sequential_3/dec_deconv1/strided_slice_3/stack_1:output:09sequential_3/dec_deconv1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv1/strided_slice_3?
8sequential_3/dec_deconv1/conv2d_transpose/ReadVariableOpReadVariableOpAsequential_3_dec_deconv1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02:
8sequential_3/dec_deconv1/conv2d_transpose/ReadVariableOp?
)sequential_3/dec_deconv1/conv2d_transposeConv2DBackpropInput'sequential_3/dec_deconv1/stack:output:0@sequential_3/dec_deconv1/conv2d_transpose/ReadVariableOp:value:0%sequential_3/reshape/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2+
)sequential_3/dec_deconv1/conv2d_transpose?
/sequential_3/dec_deconv1/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dec_deconv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/sequential_3/dec_deconv1/BiasAdd/ReadVariableOp?
 sequential_3/dec_deconv1/BiasAddBiasAdd2sequential_3/dec_deconv1/conv2d_transpose:output:07sequential_3/dec_deconv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2"
 sequential_3/dec_deconv1/BiasAdd?
sequential_3/dec_deconv1/ReluRelu)sequential_3/dec_deconv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential_3/dec_deconv1/Relu?
sequential_3/dec_deconv2/ShapeShape+sequential_3/dec_deconv1/Relu:activations:0*
T0*
_output_shapes
:2 
sequential_3/dec_deconv2/Shape?
,sequential_3/dec_deconv2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_3/dec_deconv2/strided_slice/stack?
.sequential_3/dec_deconv2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv2/strided_slice/stack_1?
.sequential_3/dec_deconv2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv2/strided_slice/stack_2?
&sequential_3/dec_deconv2/strided_sliceStridedSlice'sequential_3/dec_deconv2/Shape:output:05sequential_3/dec_deconv2/strided_slice/stack:output:07sequential_3/dec_deconv2/strided_slice/stack_1:output:07sequential_3/dec_deconv2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_3/dec_deconv2/strided_slice?
.sequential_3/dec_deconv2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv2/strided_slice_1/stack?
0sequential_3/dec_deconv2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_1/stack_1?
0sequential_3/dec_deconv2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_1/stack_2?
(sequential_3/dec_deconv2/strided_slice_1StridedSlice'sequential_3/dec_deconv2/Shape:output:07sequential_3/dec_deconv2/strided_slice_1/stack:output:09sequential_3/dec_deconv2/strided_slice_1/stack_1:output:09sequential_3/dec_deconv2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv2/strided_slice_1?
.sequential_3/dec_deconv2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv2/strided_slice_2/stack?
0sequential_3/dec_deconv2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_2/stack_1?
0sequential_3/dec_deconv2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_2/stack_2?
(sequential_3/dec_deconv2/strided_slice_2StridedSlice'sequential_3/dec_deconv2/Shape:output:07sequential_3/dec_deconv2/strided_slice_2/stack:output:09sequential_3/dec_deconv2/strided_slice_2/stack_1:output:09sequential_3/dec_deconv2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv2/strided_slice_2?
sequential_3/dec_deconv2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv2/mul/y?
sequential_3/dec_deconv2/mulMul1sequential_3/dec_deconv2/strided_slice_1:output:0'sequential_3/dec_deconv2/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv2/mul?
sequential_3/dec_deconv2/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv2/add/y?
sequential_3/dec_deconv2/addAddV2 sequential_3/dec_deconv2/mul:z:0'sequential_3/dec_deconv2/add/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv2/add?
 sequential_3/dec_deconv2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv2/mul_1/y?
sequential_3/dec_deconv2/mul_1Mul1sequential_3/dec_deconv2/strided_slice_2:output:0)sequential_3/dec_deconv2/mul_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv2/mul_1?
 sequential_3/dec_deconv2/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv2/add_1/y?
sequential_3/dec_deconv2/add_1AddV2"sequential_3/dec_deconv2/mul_1:z:0)sequential_3/dec_deconv2/add_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv2/add_1?
 sequential_3/dec_deconv2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2"
 sequential_3/dec_deconv2/stack/3?
sequential_3/dec_deconv2/stackPack/sequential_3/dec_deconv2/strided_slice:output:0 sequential_3/dec_deconv2/add:z:0"sequential_3/dec_deconv2/add_1:z:0)sequential_3/dec_deconv2/stack/3:output:0*
N*
T0*
_output_shapes
:2 
sequential_3/dec_deconv2/stack?
.sequential_3/dec_deconv2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_3/dec_deconv2/strided_slice_3/stack?
0sequential_3/dec_deconv2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_3/stack_1?
0sequential_3/dec_deconv2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv2/strided_slice_3/stack_2?
(sequential_3/dec_deconv2/strided_slice_3StridedSlice'sequential_3/dec_deconv2/stack:output:07sequential_3/dec_deconv2/strided_slice_3/stack:output:09sequential_3/dec_deconv2/strided_slice_3/stack_1:output:09sequential_3/dec_deconv2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv2/strided_slice_3?
8sequential_3/dec_deconv2/conv2d_transpose/ReadVariableOpReadVariableOpAsequential_3_dec_deconv2_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02:
8sequential_3/dec_deconv2/conv2d_transpose/ReadVariableOp?
)sequential_3/dec_deconv2/conv2d_transposeConv2DBackpropInput'sequential_3/dec_deconv2/stack:output:0@sequential_3/dec_deconv2/conv2d_transpose/ReadVariableOp:value:0+sequential_3/dec_deconv1/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2+
)sequential_3/dec_deconv2/conv2d_transpose?
/sequential_3/dec_deconv2/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dec_deconv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/sequential_3/dec_deconv2/BiasAdd/ReadVariableOp?
 sequential_3/dec_deconv2/BiasAddBiasAdd2sequential_3/dec_deconv2/conv2d_transpose:output:07sequential_3/dec_deconv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2"
 sequential_3/dec_deconv2/BiasAdd?
sequential_3/dec_deconv2/ReluRelu)sequential_3/dec_deconv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_3/dec_deconv2/Relu?
sequential_3/dec_deconv3/ShapeShape+sequential_3/dec_deconv2/Relu:activations:0*
T0*
_output_shapes
:2 
sequential_3/dec_deconv3/Shape?
,sequential_3/dec_deconv3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_3/dec_deconv3/strided_slice/stack?
.sequential_3/dec_deconv3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv3/strided_slice/stack_1?
.sequential_3/dec_deconv3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv3/strided_slice/stack_2?
&sequential_3/dec_deconv3/strided_sliceStridedSlice'sequential_3/dec_deconv3/Shape:output:05sequential_3/dec_deconv3/strided_slice/stack:output:07sequential_3/dec_deconv3/strided_slice/stack_1:output:07sequential_3/dec_deconv3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_3/dec_deconv3/strided_slice?
.sequential_3/dec_deconv3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv3/strided_slice_1/stack?
0sequential_3/dec_deconv3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_1/stack_1?
0sequential_3/dec_deconv3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_1/stack_2?
(sequential_3/dec_deconv3/strided_slice_1StridedSlice'sequential_3/dec_deconv3/Shape:output:07sequential_3/dec_deconv3/strided_slice_1/stack:output:09sequential_3/dec_deconv3/strided_slice_1/stack_1:output:09sequential_3/dec_deconv3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv3/strided_slice_1?
.sequential_3/dec_deconv3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv3/strided_slice_2/stack?
0sequential_3/dec_deconv3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_2/stack_1?
0sequential_3/dec_deconv3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_2/stack_2?
(sequential_3/dec_deconv3/strided_slice_2StridedSlice'sequential_3/dec_deconv3/Shape:output:07sequential_3/dec_deconv3/strided_slice_2/stack:output:09sequential_3/dec_deconv3/strided_slice_2/stack_1:output:09sequential_3/dec_deconv3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv3/strided_slice_2?
sequential_3/dec_deconv3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv3/mul/y?
sequential_3/dec_deconv3/mulMul1sequential_3/dec_deconv3/strided_slice_1:output:0'sequential_3/dec_deconv3/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv3/mul?
sequential_3/dec_deconv3/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv3/add/y?
sequential_3/dec_deconv3/addAddV2 sequential_3/dec_deconv3/mul:z:0'sequential_3/dec_deconv3/add/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv3/add?
 sequential_3/dec_deconv3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv3/mul_1/y?
sequential_3/dec_deconv3/mul_1Mul1sequential_3/dec_deconv3/strided_slice_2:output:0)sequential_3/dec_deconv3/mul_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv3/mul_1?
 sequential_3/dec_deconv3/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv3/add_1/y?
sequential_3/dec_deconv3/add_1AddV2"sequential_3/dec_deconv3/mul_1:z:0)sequential_3/dec_deconv3/add_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv3/add_1?
 sequential_3/dec_deconv3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2"
 sequential_3/dec_deconv3/stack/3?
sequential_3/dec_deconv3/stackPack/sequential_3/dec_deconv3/strided_slice:output:0 sequential_3/dec_deconv3/add:z:0"sequential_3/dec_deconv3/add_1:z:0)sequential_3/dec_deconv3/stack/3:output:0*
N*
T0*
_output_shapes
:2 
sequential_3/dec_deconv3/stack?
.sequential_3/dec_deconv3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_3/dec_deconv3/strided_slice_3/stack?
0sequential_3/dec_deconv3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_3/stack_1?
0sequential_3/dec_deconv3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv3/strided_slice_3/stack_2?
(sequential_3/dec_deconv3/strided_slice_3StridedSlice'sequential_3/dec_deconv3/stack:output:07sequential_3/dec_deconv3/strided_slice_3/stack:output:09sequential_3/dec_deconv3/strided_slice_3/stack_1:output:09sequential_3/dec_deconv3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv3/strided_slice_3?
8sequential_3/dec_deconv3/conv2d_transpose/ReadVariableOpReadVariableOpAsequential_3_dec_deconv3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02:
8sequential_3/dec_deconv3/conv2d_transpose/ReadVariableOp?
)sequential_3/dec_deconv3/conv2d_transposeConv2DBackpropInput'sequential_3/dec_deconv3/stack:output:0@sequential_3/dec_deconv3/conv2d_transpose/ReadVariableOp:value:0+sequential_3/dec_deconv2/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2+
)sequential_3/dec_deconv3/conv2d_transpose?
/sequential_3/dec_deconv3/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dec_deconv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential_3/dec_deconv3/BiasAdd/ReadVariableOp?
 sequential_3/dec_deconv3/BiasAddBiasAdd2sequential_3/dec_deconv3/conv2d_transpose:output:07sequential_3/dec_deconv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2"
 sequential_3/dec_deconv3/BiasAdd?
sequential_3/dec_deconv3/ReluRelu)sequential_3/dec_deconv3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_3/dec_deconv3/Relu?
sequential_3/dec_deconv4/ShapeShape+sequential_3/dec_deconv3/Relu:activations:0*
T0*
_output_shapes
:2 
sequential_3/dec_deconv4/Shape?
,sequential_3/dec_deconv4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_3/dec_deconv4/strided_slice/stack?
.sequential_3/dec_deconv4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv4/strided_slice/stack_1?
.sequential_3/dec_deconv4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv4/strided_slice/stack_2?
&sequential_3/dec_deconv4/strided_sliceStridedSlice'sequential_3/dec_deconv4/Shape:output:05sequential_3/dec_deconv4/strided_slice/stack:output:07sequential_3/dec_deconv4/strided_slice/stack_1:output:07sequential_3/dec_deconv4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_3/dec_deconv4/strided_slice?
.sequential_3/dec_deconv4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv4/strided_slice_1/stack?
0sequential_3/dec_deconv4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_1/stack_1?
0sequential_3/dec_deconv4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_1/stack_2?
(sequential_3/dec_deconv4/strided_slice_1StridedSlice'sequential_3/dec_deconv4/Shape:output:07sequential_3/dec_deconv4/strided_slice_1/stack:output:09sequential_3/dec_deconv4/strided_slice_1/stack_1:output:09sequential_3/dec_deconv4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv4/strided_slice_1?
.sequential_3/dec_deconv4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_3/dec_deconv4/strided_slice_2/stack?
0sequential_3/dec_deconv4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_2/stack_1?
0sequential_3/dec_deconv4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_2/stack_2?
(sequential_3/dec_deconv4/strided_slice_2StridedSlice'sequential_3/dec_deconv4/Shape:output:07sequential_3/dec_deconv4/strided_slice_2/stack:output:09sequential_3/dec_deconv4/strided_slice_2/stack_1:output:09sequential_3/dec_deconv4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv4/strided_slice_2?
sequential_3/dec_deconv4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv4/mul/y?
sequential_3/dec_deconv4/mulMul1sequential_3/dec_deconv4/strided_slice_1:output:0'sequential_3/dec_deconv4/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv4/mul?
sequential_3/dec_deconv4/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_3/dec_deconv4/add/y?
sequential_3/dec_deconv4/addAddV2 sequential_3/dec_deconv4/mul:z:0'sequential_3/dec_deconv4/add/y:output:0*
T0*
_output_shapes
: 2
sequential_3/dec_deconv4/add?
 sequential_3/dec_deconv4/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv4/mul_1/y?
sequential_3/dec_deconv4/mul_1Mul1sequential_3/dec_deconv4/strided_slice_2:output:0)sequential_3/dec_deconv4/mul_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv4/mul_1?
 sequential_3/dec_deconv4/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv4/add_1/y?
sequential_3/dec_deconv4/add_1AddV2"sequential_3/dec_deconv4/mul_1:z:0)sequential_3/dec_deconv4/add_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_3/dec_deconv4/add_1?
 sequential_3/dec_deconv4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_3/dec_deconv4/stack/3?
sequential_3/dec_deconv4/stackPack/sequential_3/dec_deconv4/strided_slice:output:0 sequential_3/dec_deconv4/add:z:0"sequential_3/dec_deconv4/add_1:z:0)sequential_3/dec_deconv4/stack/3:output:0*
N*
T0*
_output_shapes
:2 
sequential_3/dec_deconv4/stack?
.sequential_3/dec_deconv4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_3/dec_deconv4/strided_slice_3/stack?
0sequential_3/dec_deconv4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_3/stack_1?
0sequential_3/dec_deconv4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_3/dec_deconv4/strided_slice_3/stack_2?
(sequential_3/dec_deconv4/strided_slice_3StridedSlice'sequential_3/dec_deconv4/stack:output:07sequential_3/dec_deconv4/strided_slice_3/stack:output:09sequential_3/dec_deconv4/strided_slice_3/stack_1:output:09sequential_3/dec_deconv4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_3/dec_deconv4/strided_slice_3?
8sequential_3/dec_deconv4/conv2d_transpose/ReadVariableOpReadVariableOpAsequential_3_dec_deconv4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02:
8sequential_3/dec_deconv4/conv2d_transpose/ReadVariableOp?
)sequential_3/dec_deconv4/conv2d_transposeConv2DBackpropInput'sequential_3/dec_deconv4/stack:output:0@sequential_3/dec_deconv4/conv2d_transpose/ReadVariableOp:value:0+sequential_3/dec_deconv3/Relu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingVALID*
strides
2+
)sequential_3/dec_deconv4/conv2d_transpose?
/sequential_3/dec_deconv4/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_dec_deconv4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_3/dec_deconv4/BiasAdd/ReadVariableOp?
 sequential_3/dec_deconv4/BiasAddBiasAdd2sequential_3/dec_deconv4/conv2d_transpose:output:07sequential_3/dec_deconv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2"
 sequential_3/dec_deconv4/BiasAdd?
 sequential_3/dec_deconv4/SigmoidSigmoid)sequential_3/dec_deconv4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2"
 sequential_3/dec_deconv4/Sigmoide
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2'sequential_1/enc_fc_mu/BiasAdd:output:0,sequential_2/enc_fc_log_var/BiasAdd:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity$sequential_3/dec_deconv4/Sigmoid:y:0*
T0*/
_output_shapes
:?????????@@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesu
s:?????????@@:::::::::::::::::::::::X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: 
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_877492

inputs
enc_conv1_877470
enc_conv1_877472
enc_conv2_877475
enc_conv2_877477
enc_conv3_877480
enc_conv3_877482
enc_conv4_877485
enc_conv4_877487
identity??!enc_conv1/StatefulPartitionedCall?!enc_conv2/StatefulPartitionedCall?!enc_conv3/StatefulPartitionedCall?!enc_conv4/StatefulPartitionedCall?
!enc_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsenc_conv1_877470enc_conv1_877472*
Tin
2*
Tout
2*/
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_conv1_layer_call_and_return_conditional_losses_8772782#
!enc_conv1/StatefulPartitionedCall?
!enc_conv2/StatefulPartitionedCallStatefulPartitionedCall*enc_conv1/StatefulPartitionedCall:output:0enc_conv2_877475enc_conv2_877477*
Tin
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_conv2_layer_call_and_return_conditional_losses_8773002#
!enc_conv2/StatefulPartitionedCall?
!enc_conv3/StatefulPartitionedCallStatefulPartitionedCall*enc_conv2/StatefulPartitionedCall:output:0enc_conv3_877480enc_conv3_877482*
Tin
2*
Tout
2*0
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_conv3_layer_call_and_return_conditional_losses_8773222#
!enc_conv3/StatefulPartitionedCall?
!enc_conv4/StatefulPartitionedCallStatefulPartitionedCall*enc_conv3/StatefulPartitionedCall:output:0enc_conv4_877485enc_conv4_877487*
Tin
2*
Tout
2*0
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_conv4_layer_call_and_return_conditional_losses_8773442#
!enc_conv4/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall*enc_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_8773842
flatten/PartitionedCall?
IdentityIdentity flatten/PartitionedCall:output:0"^enc_conv1/StatefulPartitionedCall"^enc_conv2/StatefulPartitionedCall"^enc_conv3/StatefulPartitionedCall"^enc_conv4/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????@@::::::::2F
!enc_conv1/StatefulPartitionedCall!enc_conv1/StatefulPartitionedCall2F
!enc_conv2/StatefulPartitionedCall!enc_conv2/StatefulPartitionedCall2F
!enc_conv3/StatefulPartitionedCall!enc_conv3/StatefulPartitionedCall2F
!enc_conv4/StatefulPartitionedCall!enc_conv4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_gan_vae_layer_call_fn_879110

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

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*T
_output_shapesB
@:?????????@:+???????????????????????????*8
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_gan_vae_layer_call_and_return_conditional_losses_8784062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesu
s:?????????@@::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: 
?
?
-__inference_sequential_2_layer_call_fn_877665
input_3
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_8776582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: 
?
?
-__inference_sequential_3_layer_call_fn_880214

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
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_8779982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:????????? ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?

*__inference_enc_conv1_layer_call_fn_877288

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_enc_conv1_layer_call_and_return_conditional_losses_8772782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?

?
+__inference_sequential_layer_call_fn_877511
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*(
_output_shapes
:??????????**
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_8774922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????@@::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_gan_vae_layer_call_fn_879161

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

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*T
_output_shapesB
@:?????????@:+???????????????????????????*8
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_gan_vae_layer_call_and_return_conditional_losses_8785232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesu
s:?????????@@::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: 
?%
?
G__inference_dec_deconv3_layer_call_and_return_conditional_losses_877802

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingVALID*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@:::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
??
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_880189

inputs-
)dec_dense1_matmul_readvariableop_resource.
*dec_dense1_biasadd_readvariableop_resource8
4dec_deconv1_conv2d_transpose_readvariableop_resource/
+dec_deconv1_biasadd_readvariableop_resource8
4dec_deconv2_conv2d_transpose_readvariableop_resource/
+dec_deconv2_biasadd_readvariableop_resource8
4dec_deconv3_conv2d_transpose_readvariableop_resource/
+dec_deconv3_biasadd_readvariableop_resource8
4dec_deconv4_conv2d_transpose_readvariableop_resource/
+dec_deconv4_biasadd_readvariableop_resource
identity??
 dec_dense1/MatMul/ReadVariableOpReadVariableOp)dec_dense1_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02"
 dec_dense1/MatMul/ReadVariableOp?
dec_dense1/MatMulMatMulinputs(dec_dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dec_dense1/MatMul?
!dec_dense1/BiasAdd/ReadVariableOpReadVariableOp*dec_dense1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!dec_dense1/BiasAdd/ReadVariableOp?
dec_dense1/BiasAddBiasAdddec_dense1/MatMul:product:0)dec_dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dec_dense1/BiasAddz
dec_dense1/ReluReludec_dense1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dec_dense1/Reluk
reshape/ShapeShapedec_dense1/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2u
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedec_dense1/Relu:activations:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape/Reshapen
dec_deconv1/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
dec_deconv1/Shape?
dec_deconv1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
dec_deconv1/strided_slice/stack?
!dec_deconv1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv1/strided_slice/stack_1?
!dec_deconv1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv1/strided_slice/stack_2?
dec_deconv1/strided_sliceStridedSlicedec_deconv1/Shape:output:0(dec_deconv1/strided_slice/stack:output:0*dec_deconv1/strided_slice/stack_1:output:0*dec_deconv1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv1/strided_slice?
!dec_deconv1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv1/strided_slice_1/stack?
#dec_deconv1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv1/strided_slice_1/stack_1?
#dec_deconv1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv1/strided_slice_1/stack_2?
dec_deconv1/strided_slice_1StridedSlicedec_deconv1/Shape:output:0*dec_deconv1/strided_slice_1/stack:output:0,dec_deconv1/strided_slice_1/stack_1:output:0,dec_deconv1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv1/strided_slice_1?
!dec_deconv1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv1/strided_slice_2/stack?
#dec_deconv1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv1/strided_slice_2/stack_1?
#dec_deconv1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv1/strided_slice_2/stack_2?
dec_deconv1/strided_slice_2StridedSlicedec_deconv1/Shape:output:0*dec_deconv1/strided_slice_2/stack:output:0,dec_deconv1/strided_slice_2/stack_1:output:0,dec_deconv1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv1/strided_slice_2h
dec_deconv1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv1/mul/y?
dec_deconv1/mulMul$dec_deconv1/strided_slice_1:output:0dec_deconv1/mul/y:output:0*
T0*
_output_shapes
: 2
dec_deconv1/mulh
dec_deconv1/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv1/add/y}
dec_deconv1/addAddV2dec_deconv1/mul:z:0dec_deconv1/add/y:output:0*
T0*
_output_shapes
: 2
dec_deconv1/addl
dec_deconv1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv1/mul_1/y?
dec_deconv1/mul_1Mul$dec_deconv1/strided_slice_2:output:0dec_deconv1/mul_1/y:output:0*
T0*
_output_shapes
: 2
dec_deconv1/mul_1l
dec_deconv1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv1/add_1/y?
dec_deconv1/add_1AddV2dec_deconv1/mul_1:z:0dec_deconv1/add_1/y:output:0*
T0*
_output_shapes
: 2
dec_deconv1/add_1m
dec_deconv1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
dec_deconv1/stack/3?
dec_deconv1/stackPack"dec_deconv1/strided_slice:output:0dec_deconv1/add:z:0dec_deconv1/add_1:z:0dec_deconv1/stack/3:output:0*
N*
T0*
_output_shapes
:2
dec_deconv1/stack?
!dec_deconv1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dec_deconv1/strided_slice_3/stack?
#dec_deconv1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv1/strided_slice_3/stack_1?
#dec_deconv1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv1/strided_slice_3/stack_2?
dec_deconv1/strided_slice_3StridedSlicedec_deconv1/stack:output:0*dec_deconv1/strided_slice_3/stack:output:0,dec_deconv1/strided_slice_3/stack_1:output:0,dec_deconv1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv1/strided_slice_3?
+dec_deconv1/conv2d_transpose/ReadVariableOpReadVariableOp4dec_deconv1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02-
+dec_deconv1/conv2d_transpose/ReadVariableOp?
dec_deconv1/conv2d_transposeConv2DBackpropInputdec_deconv1/stack:output:03dec_deconv1/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
dec_deconv1/conv2d_transpose?
"dec_deconv1/BiasAdd/ReadVariableOpReadVariableOp+dec_deconv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"dec_deconv1/BiasAdd/ReadVariableOp?
dec_deconv1/BiasAddBiasAdd%dec_deconv1/conv2d_transpose:output:0*dec_deconv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
dec_deconv1/BiasAdd?
dec_deconv1/ReluReludec_deconv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
dec_deconv1/Relut
dec_deconv2/ShapeShapedec_deconv1/Relu:activations:0*
T0*
_output_shapes
:2
dec_deconv2/Shape?
dec_deconv2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
dec_deconv2/strided_slice/stack?
!dec_deconv2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv2/strided_slice/stack_1?
!dec_deconv2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv2/strided_slice/stack_2?
dec_deconv2/strided_sliceStridedSlicedec_deconv2/Shape:output:0(dec_deconv2/strided_slice/stack:output:0*dec_deconv2/strided_slice/stack_1:output:0*dec_deconv2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv2/strided_slice?
!dec_deconv2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv2/strided_slice_1/stack?
#dec_deconv2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv2/strided_slice_1/stack_1?
#dec_deconv2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv2/strided_slice_1/stack_2?
dec_deconv2/strided_slice_1StridedSlicedec_deconv2/Shape:output:0*dec_deconv2/strided_slice_1/stack:output:0,dec_deconv2/strided_slice_1/stack_1:output:0,dec_deconv2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv2/strided_slice_1?
!dec_deconv2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv2/strided_slice_2/stack?
#dec_deconv2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv2/strided_slice_2/stack_1?
#dec_deconv2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv2/strided_slice_2/stack_2?
dec_deconv2/strided_slice_2StridedSlicedec_deconv2/Shape:output:0*dec_deconv2/strided_slice_2/stack:output:0,dec_deconv2/strided_slice_2/stack_1:output:0,dec_deconv2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv2/strided_slice_2h
dec_deconv2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv2/mul/y?
dec_deconv2/mulMul$dec_deconv2/strided_slice_1:output:0dec_deconv2/mul/y:output:0*
T0*
_output_shapes
: 2
dec_deconv2/mulh
dec_deconv2/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv2/add/y}
dec_deconv2/addAddV2dec_deconv2/mul:z:0dec_deconv2/add/y:output:0*
T0*
_output_shapes
: 2
dec_deconv2/addl
dec_deconv2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv2/mul_1/y?
dec_deconv2/mul_1Mul$dec_deconv2/strided_slice_2:output:0dec_deconv2/mul_1/y:output:0*
T0*
_output_shapes
: 2
dec_deconv2/mul_1l
dec_deconv2/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv2/add_1/y?
dec_deconv2/add_1AddV2dec_deconv2/mul_1:z:0dec_deconv2/add_1/y:output:0*
T0*
_output_shapes
: 2
dec_deconv2/add_1l
dec_deconv2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
dec_deconv2/stack/3?
dec_deconv2/stackPack"dec_deconv2/strided_slice:output:0dec_deconv2/add:z:0dec_deconv2/add_1:z:0dec_deconv2/stack/3:output:0*
N*
T0*
_output_shapes
:2
dec_deconv2/stack?
!dec_deconv2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dec_deconv2/strided_slice_3/stack?
#dec_deconv2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv2/strided_slice_3/stack_1?
#dec_deconv2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv2/strided_slice_3/stack_2?
dec_deconv2/strided_slice_3StridedSlicedec_deconv2/stack:output:0*dec_deconv2/strided_slice_3/stack:output:0,dec_deconv2/strided_slice_3/stack_1:output:0,dec_deconv2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv2/strided_slice_3?
+dec_deconv2/conv2d_transpose/ReadVariableOpReadVariableOp4dec_deconv2_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02-
+dec_deconv2/conv2d_transpose/ReadVariableOp?
dec_deconv2/conv2d_transposeConv2DBackpropInputdec_deconv2/stack:output:03dec_deconv2/conv2d_transpose/ReadVariableOp:value:0dec_deconv1/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
dec_deconv2/conv2d_transpose?
"dec_deconv2/BiasAdd/ReadVariableOpReadVariableOp+dec_deconv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"dec_deconv2/BiasAdd/ReadVariableOp?
dec_deconv2/BiasAddBiasAdd%dec_deconv2/conv2d_transpose:output:0*dec_deconv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
dec_deconv2/BiasAdd?
dec_deconv2/ReluReludec_deconv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
dec_deconv2/Relut
dec_deconv3/ShapeShapedec_deconv2/Relu:activations:0*
T0*
_output_shapes
:2
dec_deconv3/Shape?
dec_deconv3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
dec_deconv3/strided_slice/stack?
!dec_deconv3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv3/strided_slice/stack_1?
!dec_deconv3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv3/strided_slice/stack_2?
dec_deconv3/strided_sliceStridedSlicedec_deconv3/Shape:output:0(dec_deconv3/strided_slice/stack:output:0*dec_deconv3/strided_slice/stack_1:output:0*dec_deconv3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv3/strided_slice?
!dec_deconv3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv3/strided_slice_1/stack?
#dec_deconv3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv3/strided_slice_1/stack_1?
#dec_deconv3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv3/strided_slice_1/stack_2?
dec_deconv3/strided_slice_1StridedSlicedec_deconv3/Shape:output:0*dec_deconv3/strided_slice_1/stack:output:0,dec_deconv3/strided_slice_1/stack_1:output:0,dec_deconv3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv3/strided_slice_1?
!dec_deconv3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv3/strided_slice_2/stack?
#dec_deconv3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv3/strided_slice_2/stack_1?
#dec_deconv3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv3/strided_slice_2/stack_2?
dec_deconv3/strided_slice_2StridedSlicedec_deconv3/Shape:output:0*dec_deconv3/strided_slice_2/stack:output:0,dec_deconv3/strided_slice_2/stack_1:output:0,dec_deconv3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv3/strided_slice_2h
dec_deconv3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv3/mul/y?
dec_deconv3/mulMul$dec_deconv3/strided_slice_1:output:0dec_deconv3/mul/y:output:0*
T0*
_output_shapes
: 2
dec_deconv3/mulh
dec_deconv3/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv3/add/y}
dec_deconv3/addAddV2dec_deconv3/mul:z:0dec_deconv3/add/y:output:0*
T0*
_output_shapes
: 2
dec_deconv3/addl
dec_deconv3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv3/mul_1/y?
dec_deconv3/mul_1Mul$dec_deconv3/strided_slice_2:output:0dec_deconv3/mul_1/y:output:0*
T0*
_output_shapes
: 2
dec_deconv3/mul_1l
dec_deconv3/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv3/add_1/y?
dec_deconv3/add_1AddV2dec_deconv3/mul_1:z:0dec_deconv3/add_1/y:output:0*
T0*
_output_shapes
: 2
dec_deconv3/add_1l
dec_deconv3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
dec_deconv3/stack/3?
dec_deconv3/stackPack"dec_deconv3/strided_slice:output:0dec_deconv3/add:z:0dec_deconv3/add_1:z:0dec_deconv3/stack/3:output:0*
N*
T0*
_output_shapes
:2
dec_deconv3/stack?
!dec_deconv3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dec_deconv3/strided_slice_3/stack?
#dec_deconv3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv3/strided_slice_3/stack_1?
#dec_deconv3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv3/strided_slice_3/stack_2?
dec_deconv3/strided_slice_3StridedSlicedec_deconv3/stack:output:0*dec_deconv3/strided_slice_3/stack:output:0,dec_deconv3/strided_slice_3/stack_1:output:0,dec_deconv3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv3/strided_slice_3?
+dec_deconv3/conv2d_transpose/ReadVariableOpReadVariableOp4dec_deconv3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+dec_deconv3/conv2d_transpose/ReadVariableOp?
dec_deconv3/conv2d_transposeConv2DBackpropInputdec_deconv3/stack:output:03dec_deconv3/conv2d_transpose/ReadVariableOp:value:0dec_deconv2/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
dec_deconv3/conv2d_transpose?
"dec_deconv3/BiasAdd/ReadVariableOpReadVariableOp+dec_deconv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"dec_deconv3/BiasAdd/ReadVariableOp?
dec_deconv3/BiasAddBiasAdd%dec_deconv3/conv2d_transpose:output:0*dec_deconv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
dec_deconv3/BiasAdd?
dec_deconv3/ReluReludec_deconv3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
dec_deconv3/Relut
dec_deconv4/ShapeShapedec_deconv3/Relu:activations:0*
T0*
_output_shapes
:2
dec_deconv4/Shape?
dec_deconv4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
dec_deconv4/strided_slice/stack?
!dec_deconv4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv4/strided_slice/stack_1?
!dec_deconv4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv4/strided_slice/stack_2?
dec_deconv4/strided_sliceStridedSlicedec_deconv4/Shape:output:0(dec_deconv4/strided_slice/stack:output:0*dec_deconv4/strided_slice/stack_1:output:0*dec_deconv4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv4/strided_slice?
!dec_deconv4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv4/strided_slice_1/stack?
#dec_deconv4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv4/strided_slice_1/stack_1?
#dec_deconv4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv4/strided_slice_1/stack_2?
dec_deconv4/strided_slice_1StridedSlicedec_deconv4/Shape:output:0*dec_deconv4/strided_slice_1/stack:output:0,dec_deconv4/strided_slice_1/stack_1:output:0,dec_deconv4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv4/strided_slice_1?
!dec_deconv4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!dec_deconv4/strided_slice_2/stack?
#dec_deconv4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv4/strided_slice_2/stack_1?
#dec_deconv4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv4/strided_slice_2/stack_2?
dec_deconv4/strided_slice_2StridedSlicedec_deconv4/Shape:output:0*dec_deconv4/strided_slice_2/stack:output:0,dec_deconv4/strided_slice_2/stack_1:output:0,dec_deconv4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv4/strided_slice_2h
dec_deconv4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv4/mul/y?
dec_deconv4/mulMul$dec_deconv4/strided_slice_1:output:0dec_deconv4/mul/y:output:0*
T0*
_output_shapes
: 2
dec_deconv4/mulh
dec_deconv4/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv4/add/y}
dec_deconv4/addAddV2dec_deconv4/mul:z:0dec_deconv4/add/y:output:0*
T0*
_output_shapes
: 2
dec_deconv4/addl
dec_deconv4/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv4/mul_1/y?
dec_deconv4/mul_1Mul$dec_deconv4/strided_slice_2:output:0dec_deconv4/mul_1/y:output:0*
T0*
_output_shapes
: 2
dec_deconv4/mul_1l
dec_deconv4/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv4/add_1/y?
dec_deconv4/add_1AddV2dec_deconv4/mul_1:z:0dec_deconv4/add_1/y:output:0*
T0*
_output_shapes
: 2
dec_deconv4/add_1l
dec_deconv4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
dec_deconv4/stack/3?
dec_deconv4/stackPack"dec_deconv4/strided_slice:output:0dec_deconv4/add:z:0dec_deconv4/add_1:z:0dec_deconv4/stack/3:output:0*
N*
T0*
_output_shapes
:2
dec_deconv4/stack?
!dec_deconv4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dec_deconv4/strided_slice_3/stack?
#dec_deconv4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv4/strided_slice_3/stack_1?
#dec_deconv4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#dec_deconv4/strided_slice_3/stack_2?
dec_deconv4/strided_slice_3StridedSlicedec_deconv4/stack:output:0*dec_deconv4/strided_slice_3/stack:output:0,dec_deconv4/strided_slice_3/stack_1:output:0,dec_deconv4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dec_deconv4/strided_slice_3?
+dec_deconv4/conv2d_transpose/ReadVariableOpReadVariableOp4dec_deconv4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02-
+dec_deconv4/conv2d_transpose/ReadVariableOp?
dec_deconv4/conv2d_transposeConv2DBackpropInputdec_deconv4/stack:output:03dec_deconv4/conv2d_transpose/ReadVariableOp:value:0dec_deconv3/Relu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingVALID*
strides
2
dec_deconv4/conv2d_transpose?
"dec_deconv4/BiasAdd/ReadVariableOpReadVariableOp+dec_deconv4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"dec_deconv4/BiasAdd/ReadVariableOp?
dec_deconv4/BiasAddBiasAdd%dec_deconv4/conv2d_transpose:output:0*dec_deconv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
dec_deconv4/BiasAdd?
dec_deconv4/SigmoidSigmoiddec_deconv4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
dec_deconv4/Sigmoids
IdentityIdentitydec_deconv4/Sigmoid:y:0*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:????????? :::::::::::O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
D
(__inference_reshape_layer_call_fn_880327

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_8779062
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????@@6
KL0
StatefulPartitionedCall:0?????????@J
reconstruction8
StatefulPartitionedCall:1?????????@@tensorflow/serving/predict:??
?
	optimizer
inference_net_base

mu_net

logvar_net
generative_net
loss
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature
?encode"?
_tf_keras_model?{"class_name": "ganVAE", "name": "gan_vae", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "ganVAE"}, "training_config": {"loss": {"reconstruction": "reconstruction_loss_func", "KL": "kl_loss_func"}, "metrics": null, "weighted_metrics": null, "loss_weights": [1.0, 1.0], "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
iter

beta_1

beta_2
	decay
learning_rate.m?/m?0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?.v?/v?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?"
	optimizer
?5
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?2
_tf_keras_sequential?2{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "enc_conv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}}}, {"class_name": "Conv2D", "config": {"name": "enc_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "enc_conv3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "enc_conv4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "enc_conv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}}}, {"class_name": "Conv2D", "config": {"name": "enc_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "enc_conv3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "enc_conv4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}}}}
?
layer_with_weights-0
layer-0
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_1", "layers": [{"class_name": "Dense", "config": {"name": "enc_fc_mu", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024]}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "Dense", "config": {"name": "enc_fc_mu", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024]}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}}}
?
layer_with_weights-0
layer-0
 trainable_variables
!	variables
"regularization_losses
#	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_2", "layers": [{"class_name": "Dense", "config": {"name": "enc_fc_log_var", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024]}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "Dense", "config": {"name": "enc_fc_log_var", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024]}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}}}
?>
$layer_with_weights-0
$layer-0
%layer-1
&layer_with_weights-1
&layer-2
'layer_with_weights-2
'layer-3
(layer_with_weights-3
(layer-4
)layer_with_weights-4
)layer-5
*trainable_variables
+	variables
,regularization_losses
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?<
_tf_keras_sequential?;{"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_3", "layers": [{"class_name": "Dense", "config": {"name": "dec_dense1", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1, 1024]}}}, {"class_name": "Conv2DTranspose", "config": {"name": "dec_deconv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "dec_deconv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "dec_deconv3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "dec_deconv4", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "Dense", "config": {"name": "dec_dense1", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1, 1024]}}}, {"class_name": "Conv2DTranspose", "config": {"name": "dec_deconv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "dec_deconv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "dec_deconv3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "dec_deconv4", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}}}
 "
trackable_dict_wrapper
?
.0
/1
02
13
24
35
46
57
68
79
810
911
:12
;13
<14
=15
>16
?17
@18
A19
B20
C21"
trackable_list_wrapper
?
.0
/1
02
13
24
35
46
57
68
79
810
911
:12
;13
<14
=15
>16
?17
@18
A19
B20
C21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dlayer_regularization_losses
Emetrics
Flayer_metrics
trainable_variables
Gnon_trainable_variables
	variables
	regularization_losses

Hlayers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?	

.kernel
/bias
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "enc_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "enc_conv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}}
?	

0kernel
1bias
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "enc_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "enc_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 31, 31, 32]}}
?	

2kernel
3bias
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "enc_conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "enc_conv3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 64]}}
?	

4kernel
5bias
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "enc_conv4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "enc_conv4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 6, 128]}}
?
Ytrainable_variables
Z	variables
[regularization_losses
\	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
X
.0
/1
02
13
24
35
46
57"
trackable_list_wrapper
X
.0
/1
02
13
24
35
46
57"
trackable_list_wrapper
 "
trackable_list_wrapper
?
]layer_regularization_losses
^metrics
_layer_metrics
trainable_variables
`non_trainable_variables
	variables
regularization_losses

alayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

6kernel
7bias
btrainable_variables
c	variables
dregularization_losses
e	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_fc_mu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "enc_fc_mu", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
?
flayer_regularization_losses
gmetrics
hlayer_metrics
trainable_variables
inon_trainable_variables
	variables
regularization_losses

jlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

8kernel
9bias
ktrainable_variables
l	variables
mregularization_losses
n	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "enc_fc_log_var", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "enc_fc_log_var", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
?
olayer_regularization_losses
pmetrics
qlayer_metrics
 trainable_variables
rnon_trainable_variables
!	variables
"regularization_losses

slayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

:kernel
;bias
ttrainable_variables
u	variables
vregularization_losses
w	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dec_dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dec_dense1", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
xtrainable_variables
y	variables
zregularization_losses
{	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1, 1024]}}}
?	

<kernel
=bias
|trainable_variables
}	variables
~regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "dec_deconv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dec_deconv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 1, 1024]}}
?	

>kernel
?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "dec_deconv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dec_deconv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 128]}}
?	

@kernel
Abias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "dec_deconv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dec_deconv3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 64]}}
?	

Bkernel
Cbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "dec_deconv4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dec_deconv4", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 30, 32]}}
f
:0
;1
<2
=3
>4
?5
@6
A7
B8
C9"
trackable_list_wrapper
f
:0
;1
<2
=3
>4
?5
@6
A7
B8
C9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
*trainable_variables
?non_trainable_variables
+	variables
,regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( 2enc_conv1/kernel
: 2enc_conv1/bias
*:( @2enc_conv2/kernel
:@2enc_conv2/bias
+:)@?2enc_conv3/kernel
:?2enc_conv3/bias
,:*??2enc_conv4/kernel
:?2enc_conv4/bias
#:!	? 2enc_fc_mu/kernel
: 2enc_fc_mu/bias
(:&	? 2enc_fc_log_var/kernel
!: 2enc_fc_log_var/bias
$:"	 ?2dec_dense1/kernel
:?2dec_dense1/bias
.:,??2dec_deconv1/kernel
:?2dec_deconv1/bias
-:+@?2dec_deconv2/kernel
:@2dec_deconv2/bias
,:* @2dec_deconv3/kernel
: 2dec_deconv3/bias
,:* 2dec_deconv4/kernel
:2dec_deconv4/bias
 "
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
Itrainable_variables
?non_trainable_variables
J	variables
Kregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
Mtrainable_variables
?non_trainable_variables
N	variables
Oregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
Qtrainable_variables
?non_trainable_variables
R	variables
Sregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
Utrainable_variables
?non_trainable_variables
V	variables
Wregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
Ytrainable_variables
?non_trainable_variables
Z	variables
[regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
btrainable_variables
?non_trainable_variables
c	variables
dregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
ktrainable_variables
?non_trainable_variables
l	variables
mregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
ttrainable_variables
?non_trainable_variables
u	variables
vregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
xtrainable_variables
?non_trainable_variables
y	variables
zregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
|trainable_variables
?non_trainable_variables
}	variables
~regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
$0
%1
&2
'3
(4
)5"
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricp{"class_name": "Mean", "name": "KL_loss", "dtype": "float32", "config": {"name": "KL_loss", "dtype": "float32"}}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "reconstruction_loss", "dtype": "float32", "config": {"name": "reconstruction_loss", "dtype": "float32"}}
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
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:- 2Adam/enc_conv1/kernel/m
!: 2Adam/enc_conv1/bias/m
/:- @2Adam/enc_conv2/kernel/m
!:@2Adam/enc_conv2/bias/m
0:.@?2Adam/enc_conv3/kernel/m
": ?2Adam/enc_conv3/bias/m
1:/??2Adam/enc_conv4/kernel/m
": ?2Adam/enc_conv4/bias/m
(:&	? 2Adam/enc_fc_mu/kernel/m
!: 2Adam/enc_fc_mu/bias/m
-:+	? 2Adam/enc_fc_log_var/kernel/m
&:$ 2Adam/enc_fc_log_var/bias/m
):'	 ?2Adam/dec_dense1/kernel/m
#:!?2Adam/dec_dense1/bias/m
3:1??2Adam/dec_deconv1/kernel/m
$:"?2Adam/dec_deconv1/bias/m
2:0@?2Adam/dec_deconv2/kernel/m
#:!@2Adam/dec_deconv2/bias/m
1:/ @2Adam/dec_deconv3/kernel/m
#:! 2Adam/dec_deconv3/bias/m
1:/ 2Adam/dec_deconv4/kernel/m
#:!2Adam/dec_deconv4/bias/m
/:- 2Adam/enc_conv1/kernel/v
!: 2Adam/enc_conv1/bias/v
/:- @2Adam/enc_conv2/kernel/v
!:@2Adam/enc_conv2/bias/v
0:.@?2Adam/enc_conv3/kernel/v
": ?2Adam/enc_conv3/bias/v
1:/??2Adam/enc_conv4/kernel/v
": ?2Adam/enc_conv4/bias/v
(:&	? 2Adam/enc_fc_mu/kernel/v
!: 2Adam/enc_fc_mu/bias/v
-:+	? 2Adam/enc_fc_log_var/kernel/v
&:$ 2Adam/enc_fc_log_var/bias/v
):'	 ?2Adam/dec_dense1/kernel/v
#:!?2Adam/dec_dense1/bias/v
3:1??2Adam/dec_deconv1/kernel/v
$:"?2Adam/dec_deconv1/bias/v
2:0@?2Adam/dec_deconv2/kernel/v
#:!@2Adam/dec_deconv2/bias/v
1:/ @2Adam/dec_deconv3/kernel/v
#:! 2Adam/dec_deconv3/bias/v
1:/ 2Adam/dec_deconv4/kernel/v
#:!2Adam/dec_deconv4/bias/v
?2?
C__inference_gan_vae_layer_call_and_return_conditional_losses_879587
C__inference_gan_vae_layer_call_and_return_conditional_losses_878846
C__inference_gan_vae_layer_call_and_return_conditional_losses_879374
C__inference_gan_vae_layer_call_and_return_conditional_losses_879059?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_gan_vae_layer_call_fn_879689
(__inference_gan_vae_layer_call_fn_879638
(__inference_gan_vae_layer_call_fn_879110
(__inference_gan_vae_layer_call_fn_879161?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_877266?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????@@
?2??
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_sequential_layer_call_and_return_conditional_losses_879723
F__inference_sequential_layer_call_and_return_conditional_losses_879757
F__inference_sequential_layer_call_and_return_conditional_losses_877418
F__inference_sequential_layer_call_and_return_conditional_losses_877393?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_sequential_layer_call_fn_879799
+__inference_sequential_layer_call_fn_879778
+__inference_sequential_layer_call_fn_877465
+__inference_sequential_layer_call_fn_877511?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_1_layer_call_and_return_conditional_losses_877551
H__inference_sequential_1_layer_call_and_return_conditional_losses_879819
H__inference_sequential_1_layer_call_and_return_conditional_losses_877542
H__inference_sequential_1_layer_call_and_return_conditional_losses_879809?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_1_layer_call_fn_879837
-__inference_sequential_1_layer_call_fn_879828
-__inference_sequential_1_layer_call_fn_877570
-__inference_sequential_1_layer_call_fn_877588?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_2_layer_call_and_return_conditional_losses_879847
H__inference_sequential_2_layer_call_and_return_conditional_losses_879857
H__inference_sequential_2_layer_call_and_return_conditional_losses_877628
H__inference_sequential_2_layer_call_and_return_conditional_losses_877619?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_2_layer_call_fn_879875
-__inference_sequential_2_layer_call_fn_879866
-__inference_sequential_2_layer_call_fn_877665
-__inference_sequential_2_layer_call_fn_877647?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_3_layer_call_and_return_conditional_losses_880189
H__inference_sequential_3_layer_call_and_return_conditional_losses_880032
H__inference_sequential_3_layer_call_and_return_conditional_losses_877965
H__inference_sequential_3_layer_call_and_return_conditional_losses_877935?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_3_layer_call_fn_878076
-__inference_sequential_3_layer_call_fn_880239
-__inference_sequential_3_layer_call_fn_880214
-__inference_sequential_3_layer_call_fn_878021?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
3B1
$__inference_signature_wrapper_878633input_1
?2?
E__inference_enc_conv1_layer_call_and_return_conditional_losses_877278?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
*__inference_enc_conv1_layer_call_fn_877288?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
E__inference_enc_conv2_layer_call_and_return_conditional_losses_877300?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
*__inference_enc_conv2_layer_call_fn_877310?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
E__inference_enc_conv3_layer_call_and_return_conditional_losses_877322?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
*__inference_enc_conv3_layer_call_fn_877332?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
E__inference_enc_conv4_layer_call_and_return_conditional_losses_877344?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
*__inference_enc_conv4_layer_call_fn_877354?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
C__inference_flatten_layer_call_and_return_conditional_losses_880245?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_flatten_layer_call_fn_880250?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_enc_fc_mu_layer_call_and_return_conditional_losses_880260?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_enc_fc_mu_layer_call_fn_880269?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_enc_fc_log_var_layer_call_and_return_conditional_losses_880279?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_enc_fc_log_var_layer_call_fn_880288?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dec_dense1_layer_call_and_return_conditional_losses_880299?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dec_dense1_layer_call_fn_880308?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_reshape_layer_call_and_return_conditional_losses_880322?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_reshape_layer_call_fn_880327?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dec_deconv1_layer_call_and_return_conditional_losses_877704?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
,__inference_dec_deconv1_layer_call_fn_877714?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
G__inference_dec_deconv2_layer_call_and_return_conditional_losses_877753?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
,__inference_dec_deconv2_layer_call_fn_877763?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
G__inference_dec_deconv3_layer_call_and_return_conditional_losses_877802?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
,__inference_dec_deconv3_layer_call_fn_877812?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
G__inference_dec_deconv4_layer_call_and_return_conditional_losses_877851?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
,__inference_dec_deconv4_layer_call_fn_877861?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? ?
!__inference__wrapped_model_877266?./0123456789:;<=>?@ABC8?5
.?+
)?&
input_1?????????@@
? "k?h
"
KL?
KL?????????@
B
reconstruction0?-
reconstruction?????????@@?
G__inference_dec_deconv1_layer_call_and_return_conditional_losses_877704?<=J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
,__inference_dec_deconv1_layer_call_fn_877714?<=J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
G__inference_dec_deconv2_layer_call_and_return_conditional_losses_877753?>?J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
,__inference_dec_deconv2_layer_call_fn_877763?>?J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????@?
G__inference_dec_deconv3_layer_call_and_return_conditional_losses_877802?@AI?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
,__inference_dec_deconv3_layer_call_fn_877812?@AI?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
G__inference_dec_deconv4_layer_call_and_return_conditional_losses_877851?BCI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
,__inference_dec_deconv4_layer_call_fn_877861?BCI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
F__inference_dec_dense1_layer_call_and_return_conditional_losses_880299]:;/?,
%?"
 ?
inputs????????? 
? "&?#
?
0??????????
? 
+__inference_dec_dense1_layer_call_fn_880308P:;/?,
%?"
 ?
inputs????????? 
? "????????????
E__inference_enc_conv1_layer_call_and_return_conditional_losses_877278?./I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+??????????????????????????? 
? ?
*__inference_enc_conv1_layer_call_fn_877288?./I?F
??<
:?7
inputs+???????????????????????????
? "2?/+??????????????????????????? ?
E__inference_enc_conv2_layer_call_and_return_conditional_losses_877300?01I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????@
? ?
*__inference_enc_conv2_layer_call_fn_877310?01I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+???????????????????????????@?
E__inference_enc_conv3_layer_call_and_return_conditional_losses_877322?23I?F
??<
:?7
inputs+???????????????????????????@
? "@?=
6?3
0,????????????????????????????
? ?
*__inference_enc_conv3_layer_call_fn_877332?23I?F
??<
:?7
inputs+???????????????????????????@
? "3?0,?????????????????????????????
E__inference_enc_conv4_layer_call_and_return_conditional_losses_877344?45J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
*__inference_enc_conv4_layer_call_fn_877354?45J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
J__inference_enc_fc_log_var_layer_call_and_return_conditional_losses_880279]890?-
&?#
!?
inputs??????????
? "%?"
?
0????????? 
? ?
/__inference_enc_fc_log_var_layer_call_fn_880288P890?-
&?#
!?
inputs??????????
? "?????????? ?
E__inference_enc_fc_mu_layer_call_and_return_conditional_losses_880260]670?-
&?#
!?
inputs??????????
? "%?"
?
0????????? 
? ~
*__inference_enc_fc_mu_layer_call_fn_880269P670?-
&?#
!?
inputs??????????
? "?????????? ?
C__inference_flatten_layer_call_and_return_conditional_losses_880245b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????
? ?
(__inference_flatten_layer_call_fn_880250U8?5
.?+
)?&
inputs??????????
? "????????????
C__inference_gan_vae_layer_call_and_return_conditional_losses_878846?./0123456789:;<=>?@ABC;?8
1?.
(?%
inputs?????????@@
p
? "y?v
o?l
$
KL?
0/KL?????????@
D
reconstruction2?/
0/reconstruction?????????@@
? ?
C__inference_gan_vae_layer_call_and_return_conditional_losses_879059?./0123456789:;<=>?@ABC;?8
1?.
(?%
inputs?????????@@
p 
? "y?v
o?l
$
KL?
0/KL?????????@
D
reconstruction2?/
0/reconstruction?????????@@
? ?
C__inference_gan_vae_layer_call_and_return_conditional_losses_879374?./0123456789:;<=>?@ABC<?9
2?/
)?&
input_1?????????@@
p
? "y?v
o?l
$
KL?
0/KL?????????@
D
reconstruction2?/
0/reconstruction?????????@@
? ?
C__inference_gan_vae_layer_call_and_return_conditional_losses_879587?./0123456789:;<=>?@ABC<?9
2?/
)?&
input_1?????????@@
p 
? "y?v
o?l
$
KL?
0/KL?????????@
D
reconstruction2?/
0/reconstruction?????????@@
? ?
(__inference_gan_vae_layer_call_fn_879110?./0123456789:;<=>?@ABC;?8
1?.
(?%
inputs?????????@@
p
? "}?z
"
KL?
KL?????????@
T
reconstructionB??
reconstruction+????????????????????????????
(__inference_gan_vae_layer_call_fn_879161?./0123456789:;<=>?@ABC;?8
1?.
(?%
inputs?????????@@
p 
? "}?z
"
KL?
KL?????????@
T
reconstructionB??
reconstruction+????????????????????????????
(__inference_gan_vae_layer_call_fn_879638?./0123456789:;<=>?@ABC<?9
2?/
)?&
input_1?????????@@
p
? "}?z
"
KL?
KL?????????@
T
reconstructionB??
reconstruction+????????????????????????????
(__inference_gan_vae_layer_call_fn_879689?./0123456789:;<=>?@ABC<?9
2?/
)?&
input_1?????????@@
p 
? "}?z
"
KL?
KL?????????@
T
reconstructionB??
reconstruction+????????????????????????????
C__inference_reshape_layer_call_and_return_conditional_losses_880322b0?-
&?#
!?
inputs??????????
? ".?+
$?!
0??????????
? ?
(__inference_reshape_layer_call_fn_880327U0?-
&?#
!?
inputs??????????
? "!????????????
H__inference_sequential_1_layer_call_and_return_conditional_losses_877542f679?6
/?,
"?
input_2??????????
p

 
? "%?"
?
0????????? 
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_877551f679?6
/?,
"?
input_2??????????
p 

 
? "%?"
?
0????????? 
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_879809e678?5
.?+
!?
inputs??????????
p

 
? "%?"
?
0????????? 
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_879819e678?5
.?+
!?
inputs??????????
p 

 
? "%?"
?
0????????? 
? ?
-__inference_sequential_1_layer_call_fn_877570Y679?6
/?,
"?
input_2??????????
p

 
? "?????????? ?
-__inference_sequential_1_layer_call_fn_877588Y679?6
/?,
"?
input_2??????????
p 

 
? "?????????? ?
-__inference_sequential_1_layer_call_fn_879828X678?5
.?+
!?
inputs??????????
p

 
? "?????????? ?
-__inference_sequential_1_layer_call_fn_879837X678?5
.?+
!?
inputs??????????
p 

 
? "?????????? ?
H__inference_sequential_2_layer_call_and_return_conditional_losses_877619f899?6
/?,
"?
input_3??????????
p

 
? "%?"
?
0????????? 
? ?
H__inference_sequential_2_layer_call_and_return_conditional_losses_877628f899?6
/?,
"?
input_3??????????
p 

 
? "%?"
?
0????????? 
? ?
H__inference_sequential_2_layer_call_and_return_conditional_losses_879847e898?5
.?+
!?
inputs??????????
p

 
? "%?"
?
0????????? 
? ?
H__inference_sequential_2_layer_call_and_return_conditional_losses_879857e898?5
.?+
!?
inputs??????????
p 

 
? "%?"
?
0????????? 
? ?
-__inference_sequential_2_layer_call_fn_877647Y899?6
/?,
"?
input_3??????????
p

 
? "?????????? ?
-__inference_sequential_2_layer_call_fn_877665Y899?6
/?,
"?
input_3??????????
p 

 
? "?????????? ?
-__inference_sequential_2_layer_call_fn_879866X898?5
.?+
!?
inputs??????????
p

 
? "?????????? ?
-__inference_sequential_2_layer_call_fn_879875X898?5
.?+
!?
inputs??????????
p 

 
? "?????????? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_877935?
:;<=>?@ABC8?5
.?+
!?
input_4????????? 
p

 
? "??<
5?2
0+???????????????????????????
? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_877965?
:;<=>?@ABC8?5
.?+
!?
input_4????????? 
p 

 
? "??<
5?2
0+???????????????????????????
? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_880032t
:;<=>?@ABC7?4
-?*
 ?
inputs????????? 
p

 
? "-?*
#? 
0?????????@@
? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_880189t
:;<=>?@ABC7?4
-?*
 ?
inputs????????? 
p 

 
? "-?*
#? 
0?????????@@
? ?
-__inference_sequential_3_layer_call_fn_878021z
:;<=>?@ABC8?5
.?+
!?
input_4????????? 
p

 
? "2?/+????????????????????????????
-__inference_sequential_3_layer_call_fn_878076z
:;<=>?@ABC8?5
.?+
!?
input_4????????? 
p 

 
? "2?/+????????????????????????????
-__inference_sequential_3_layer_call_fn_880214y
:;<=>?@ABC7?4
-?*
 ?
inputs????????? 
p

 
? "2?/+????????????????????????????
-__inference_sequential_3_layer_call_fn_880239y
:;<=>?@ABC7?4
-?*
 ?
inputs????????? 
p 

 
? "2?/+????????????????????????????
F__inference_sequential_layer_call_and_return_conditional_losses_877393t./012345@?=
6?3
)?&
input_1?????????@@
p

 
? "&?#
?
0??????????
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_877418t./012345@?=
6?3
)?&
input_1?????????@@
p 

 
? "&?#
?
0??????????
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_879723s./012345??<
5?2
(?%
inputs?????????@@
p

 
? "&?#
?
0??????????
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_879757s./012345??<
5?2
(?%
inputs?????????@@
p 

 
? "&?#
?
0??????????
? ?
+__inference_sequential_layer_call_fn_877465g./012345@?=
6?3
)?&
input_1?????????@@
p

 
? "????????????
+__inference_sequential_layer_call_fn_877511g./012345@?=
6?3
)?&
input_1?????????@@
p 

 
? "????????????
+__inference_sequential_layer_call_fn_879778f./012345??<
5?2
(?%
inputs?????????@@
p

 
? "????????????
+__inference_sequential_layer_call_fn_879799f./012345??<
5?2
(?%
inputs?????????@@
p 

 
? "????????????
$__inference_signature_wrapper_878633?./0123456789:;<=>?@ABCC?@
? 
9?6
4
input_1)?&
input_1?????????@@"k?h
"
KL?
KL?????????@
B
reconstruction0?-
reconstruction?????????@@