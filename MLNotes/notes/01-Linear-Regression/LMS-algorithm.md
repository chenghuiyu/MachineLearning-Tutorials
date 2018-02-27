在介绍线性回归方程之前简单提一下监督学习，线性回归是监督学习的一种，监督学习就是通过一些训练集得到一个优化模型*h*，通过这个模型对新的输入变量x得到输出变量y，监督学习就是为了得到一个最理想的*h*模型。

 <img src="../../images/01/supervised.jpg" width = "60%"/>

# **线性回归方程**

在上面提到过优化模型*h*可以看成是输入*x*的函数，为了方便起见记为<img src="../../images/common/h(x).jpg" width = "4%"/>，并假设该方程是关于*x*的连续函数，这是监督学习最为简单的一种。具体表达式如下所示：

<img src="../../images/01/ehtaX.jpg" width = "60%"/>

线性回归方程的目的就是使得<img src="../../images/common/h(x).jpg" width = "4%"/>尽可能的逼近于*y*，为了便于表示这种近似关系，定义误差方程<img src="../../images/common/j.jpg" width = "3%"/>:

<img src="../../images/01/j2.jpg" width = "40%"/>

上述的表达式类似于最小均方值误差求解的方法，但是后面会给出数学上一般形式的表达方式。先做一个小结，上面主要讨论了在输入一些变量后求得的估计函数可以使后续的输入变量更加的接近真实值，这就是监督学习最简单的一种形式。

## **最小均方误差算法LMS**

LMS算法的目的就是如何选择权重<img src="../../images/common/ehta.jpg" width = "1.5%"/>使得<img src="../../images/01/j.jpg" width = "3%"/>达到最小值。**梯度下降算法**给出了一种求解<img src="../../images/common/ehta.jpg" width = "1.5%"/>的方法来使得误差方程最小。

<img src="../../images/01/ehtaJ.jpg" width = "28%"/>

下面来求解<img src="../../images/01/j.jpg" width = "3%"/>关于<img src="../../images/common/ehta.jpg" width = "1.5%"/>的偏导。

<img src="../../images/01/Jehta.jpg" width = "60%"/>

将求得的结果带入梯度下降算法得到结果。

<img src="../../images/01/piandao.jpg" width = "35%"/>


上述方程就是最小均方误差算法（LMS），这里仅仅是对一个特征量*j*来进行的计算，如果输入的*x*存在多个特征量就会有*{1,2...,j}*个那么就需要对上述算法进行改进。

### **batch梯度下降算法**

<img src="../../images/01/bath.jpg" width = "50%"/>

这个算法需要对输入量*x*的每个特征的参数更新都需要把所有的样本值计算一遍，这样梯度下降的结果总会收敛于一个整体的最小值。

### **incremental梯度下降算法**

<img src="../../images/01/loop.jpg" width = "50%"/>

这个算法不需要把输入的相同特征的所有样本数都计算一遍，而是每输入一个样本值就会对<img src="../../images/common/ehta.jpg" width = "1.5%"/>进行更新。这样做的好处就是更加迅速的得到使误差函数<img src="../../images/01/j.jpg" width = "3%"/>达到最小值的参数<img src="../../images/common/ehta.jpg" width = "1.5%"/>，特别是在样本数量特别大的时候，该算法收敛速度更快。

## **常规方程**

LMS算法主要采迭代的方式得到<img src="../../images/common/ehta.jpg" width = "1.5%"/>使得误差方程收敛到最小值，常规方程的解法在于对每个<img src="../../images/common/ehta.jpg" width = "1.5%"/>得到的误差方程进行求导，并使其逼近0，得到一个收敛的最小结果。

### **数学基础**

<img src="../../images/01/math.jpg" width = "80%"/>

### **基于梯度矩阵的最小均方算法**

下面来考虑具体的实现算法，给定一组训练集合*X*，如何通过构造梯度矩阵来使误差函数达到收敛的最小值。

<img src="../../images/01/juzhen.jpg" width = "80%"/>

对上述结果进行梯度运算有：

<img src="../../images/01/tidu.jpg" width = "80%"/>

最终得到结果为：

<img src="../../images/01/xy.jpg" width = "20%"/>

## **最大似然估计算法**

下面从统计概率的角度分析上述问题，首先定义输入和输出样本满足：


<img src="../../images/01/yi.jpg" width = "20%"/>

其中，<img src="../../images/common/e.jpg" width = "1.5%"/>为随机噪声误差，根据概率统计论可以将其分布看为高斯概率分布，即均方值为0，方差为<img src="../../images/common/delta.jpg" width = "1.5%"/>，有如下表达式：

<img src="../../images/01/pe.jpg" width = "35%"/>

将定义公式带入上式得到：

<img src="../../images/01/pyx.jpg" width = "50%"/>

其中，<img src="../../images/01/p.jpg" width = "9%"/>表示定义参数<img src="../../images/common/delta.jpg" width = "3%"/>下，在给定*x*时*y*出现的概率分布，记为：<img src="../../images/01/yx.png" width = "20%"/>。

按照在上一节的常规方程给出的方法，将输入的所有样本<img src="../../images/common/xi.jpg" width = "3%"/>构造成一个矩阵*X*，那么概率分布<img src="../../images/01/p.jpg" width = "9%"/>就可以表示为<img src="../../images/01/pyx1.jpg" width = "9%"/>，我们的目的是为了确定给定参数<img src="../../images/common/delta.jpg" width = "3%"/>的时候，概率的分布情况，那也可以将概率分布看为是关于<img src="../../images/common/delta.jpg" width = "3%"/>的函数，似然方程就是这么定义出来的。

<img src="../../images/01/l.jpg" width = "40%"/>

上述似然方程中*X*是一个样本矩阵，*y*也是矩阵向量，由于样本值是独立而且同分布的，那么就可以将似然函数改写为：

<img src="../../images/01/lta.jpg" width = "50%"/>

为了使得学习估计的结果更加精确，也就是输入的样本值*X*后，我们可以得到的结果和真实值更加的接近，应该使得概率分布函数的值达到最大，这就转换为如何确定参数<img src="../../images/common/delta.jpg" width = "3%"/>使得似然函数达到最大值。上述方程为乘积的形式，对于数学处理过程显得不方便，为了简化函数的处理过程，对等式两边同时取对数，转换为加和的形式，方便后续处理。

<img src="../../images/01/ll.jpg" width = "50%"/>

从上式可以看出为了使得<img src="../../images/01/hual.jpg" width = "4%"/>达到最大值，应该使得下述方程达到最小值。

<img src="../../images/01/yehta.jpg" width = "30%"/>

而上述方程正式我们之前求<img src="../../images/common/j.jpg" width = "1.5%"/>最小值时求得结果，一种隐藏的纽带将这几个方法的结果联系起来，后面会详细的讲解这种联系，并归纳出该类问题更加泛化的求解过程。同时注意到一个有意思的细节，也就是最大似然函数的结果并不依赖<img src="../../images/common/delta.jpg" width = "3%"/>，虽然在之前的讨论中为了表达的方便而引入了这个参数。

## links
   * [目录](../../README.md)
   * 下一节: [Classification and Logistic Regression](../02-Classification-and-Logistic-Regression/Classification-and-Logistic-Regression.md)
