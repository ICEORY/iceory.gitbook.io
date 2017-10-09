# gitbook 安装及使用（windows）[参考](https://yuzeshan.gitbooks.io/gitbook-studying/content/index.html)
## 安装node.js
直接到官网下载exe安装即可，[参考网站](https://blog.gtwang.org/web-development/install-node-js-in-windows-mac-os-x-linux/)
## 使用npm安装gitbook
**注意**：不知道什么原因，gitbook命令只能在nodejs的终端中使用，在系统的cmd或者powershell中不被识别

npm install gitbook-cli -g

在node.js command prompt 中使用 gitbook -V 确认安装成功

## gitbook使用
打开node.js command prompt

运行gitbook

## 常用命令
本地预览： gitbook serve ./book_name

输出静态网站： gitbook build ./book\_path ./output\_path

查看帮助： gitbook help

## 构建github blog
**注意**： 仓库名称必须为usename.github.io，注意大小写，跟官方教程有点不一样，这个后期再弄。[参考](https://yuzeshan.gitbooks.io/gitbook-studying/content/publish/gitpages.html)

1. 创建github远程仓库
2. 克隆仓库到本地： git clone ....
3. 创建一个新的分支 git checkout -b gh-pages(分支名必须为gh-pages)
4. 将分支push到仓库
5. 切换到主分支master
6. 将gitbook build得到的静态网页文件复制到改仓库的本地目录下
7. 提交到远程仓库
8. 打开http://username.github.io可以看到网页版本的gitbook
