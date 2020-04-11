# Octave 2D Convolution Keras

A simple yet effective technique for producing cross -channel correlations.
It builds upon the idea of severity of variance of features of an image & treats the structure information 
as low variance information thus treating them as low frequency inputs and the detail information as high variance or high frequency.

Divides or factors the input as 
~~~
X = {Xh, Xl}
~~~

and prediction on 

~~~
Y = {Yh, Yl}
~~~

<i> where Yh & Yl sre broken broken down as 

~~~
Yh = {Yh->h, Yl->h}
Yl = {Yl->l, Yh->l}
~~~
