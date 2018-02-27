# **广义线性模型**

之前讨论的高斯分布函数以及伯努利概率分布函数是属于广义线性模型（GLM）的特例，下面来具体讨论。


## **指数分布族**

为了更好的介绍GLMs，先来给出指数分布函数的表示：

<img src="../../images/01/py.jpg" width = "35%"/>

将高斯分布函数和伯努利概率函数改写为指数分布族的形式。

### *伯努利分布函数*


<img src="../../images/01/py1.jpg" width = "55%"/>

其中各个参数对应于指数分布族函数为：

<img src="../../images/01/canshu1.jpg" width = "25%"/>

### *高斯概率分布函数*


<img src="../../images/01/pyu.jpg" width = "40%"/>

其中各个参数对应于指数分布族函数为：

<img src="../../images/01/canshu2.jpg" width = "30%"/>

## **广义线性模型（GLMs）的构造方法**


<img src="../../images/01/glms.jpg" width = "55%"/>

- 1、首先*x*和*y*要满足指数族分布。
- 2、<img src="../../images/common/h(x).jpg" width = "4%"/>可以通过期望<img src="../../images/01/qiwang.jpg" width = "6%"/>来计算。
- 3、参数<img src="../../images/common/yita.jpg" width = "1.5%"/>和输入的样本*x*满足线性分布。

高斯概率分布函数

<img src="../../images/01/bnqw.jpg" width = "20%"/>

伯努利概率分布函数

<img src="../../images/01/gaosiqw.jpg" width = "25%"/>


## **广义逻辑回归模型**

下面讨论一个具体的问题，多元分布概率模型，输入一定数量的样本值，将其分为多个类别{1,2,...,k}，下面就详细的讨论模型构造的过程。
本例子将输入样本值的估计分为三类，而不同于之前讨论的0和1

<img src="../../images/01/ml1.jpg" width = "60%"/>

下面来对条件概率进行迭代求解

<img src="../../images/01/ml2.jpg" width = "60%"/>

下面来求解GLMs模型中各个参数的值：

<img src="../../images/01/ml3.jpg" width = "50%"/>

带入到条件概率中得到：

<img src="../../images/01/ml4.jpg" width = "35%"/>

估计函数可以表示为：

<img src="../../images/01/ml5.jpg" width = "35%"/>

最大似然估计函数表示为：

<img src="../../images/01/ml7.jpg" width = "60%"/>


然后可以参照之前的梯度下降算法求得最大似然函数的取值。
具体数学推导过程如下所示：

<img src="../../images/01/mltitidu.jpg" width = "80%"/>

## links
   * [目录](../../README.md)
   * 上一节: [Classification and Logistic Regression](../02-Classification-and-Logistic-Regression/Classification-and-Logistic-Regression.md)
   * 下一节: [Generative Learning Algorithms](../04-Generative-Learning-Algorithms/Generative-Learning-Algorithms.md)
