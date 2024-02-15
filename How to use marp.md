---
marp: true
theme: gaia        
header: Marp 教程  
footer: '2024/1/22'  
paginate: true
size: 16:9
math: katex

---



# <!-- fit --> 如何使用marp设置ppt封面 
</br>

### 目录
- 头文件作用
- 某页前预设
- 插入公式
- 插入代码和引用块
- 插入图片

---

# 头文件作用：
- theme：主题（defalut、uncover、gaia）
- footer：脚注 （在‘’中表示）
- paginate：true/false
- size：ppt页面大小比例
- math：选择数学公式输入方式
- 使用 --- 进行分页

---

<!-- _backgroundColor: blue-->
<!-- _color: white-->

# 某页前预设
- 页颜色 \<!-- _backgroundColor: 颜色英文-->
- 字颜色 \<!-- _color: 颜色英文-->
- 换行\</br> 
- 若要设置此页及其之后所有页的设置\<!--color: 颜色英文>

---

# 插入公式
## 行内公式
勾股定理为：$a^2+b^2=c^2$
## 行间公式
勾股定理为：
$$
a^2+b^2=c^2
$$

---

# 插入代码和引用块
## 行间代码
python代码如此```print("hello world)```
## 代码块
python代码如此 
```python
    print("hello world")
```
---

## 引用块
     第一层引用

        第二层引用

---

# 插入图片
## 插入图片
![w:200px, h:400px](模型总流程.jpg)

---

## 设置背景图
![bg fit](模型总流程.jpg)

---

### 横向排列
![bg](https://fakeimg.pl/800x600/0288d1/fff/?text=A)
![bg](https://fakeimg.pl/800x600/0288d1/fff/?text=B)
![bg](https://fakeimg.pl/800x600/0288d1/fff/?text=C)

---

### 纵向排列
![bg vertical](https://fakeimg.pl/800x600/0288d1/fff/?text=A)
![bg](https://fakeimg.pl/800x600/0288d1/fff/?text=B)
![bg](https://fakeimg.pl/800x600/0288d1/fff/?text=C)

---

## 固定背景图
![bg left h:600px](模型总流程.jpg)
<!-- color: black--> 
这是模型的总流程图

---

![bg right 100%](模型总流程.jpg)
![bg 100%](模型总流程.jpg)
<!-- color: black--> 
这是模型的总流程图
