内容：机器学习

姓名：徐梓文

学号：2410306105



![image-20250409195918365](/Users/ziwenxu/Library/Application Support/typora-user-images/image-20250409195918365.png)

​													**1.(1) 图像**

![361744273388_.pic](/Users/ziwenxu/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/518e1465865b36fc3810ded2640515a6/Message/MessageTemp/518e1465865b36fc3810ded2640515a6/Image/361744273388_.pic.jpg)

![341744273334_.pic_hd](/Users/ziwenxu/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/518e1465865b36fc3810ded2640515a6/Message/MessageTemp/518e1465865b36fc3810ded2640515a6/Image/341744273334_.pic_hd.jpg)

![351744273336_.pic_hd](/Users/ziwenxu/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/518e1465865b36fc3810ded2640515a6/Message/MessageTemp/518e1465865b36fc3810ded2640515a6/Image/351744273336_.pic_hd.jpg)

* 原因：输入梯度$\frac{\partial L}{\partial x}$包含直接传递的原始梯度$g$，即使线性层部分衰减，仍能保留$g$的部分。避免了完全流经ReLU和权重矩阵，减少逐层递减的风险。