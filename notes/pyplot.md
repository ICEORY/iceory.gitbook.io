# PyPlot 绘图笔记

## 改变字体类型
reference: http://blog.csdn.net/ginynu/article/details/70808962
```python
import matplotlib.pyplot as plt  
plt.rc('font',family='Times New Roman')  
```

## 自动调整边界范围
reference: https://stackoverflow.com/questions/4042192/reduce-left-and-right-margins-in-matplotlib-plot
```python
plt.tight_layout()
```