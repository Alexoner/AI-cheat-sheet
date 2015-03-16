# Chinese document in latex

## useful links
(http://www.sharelatex.com/learn/chinese)

The easiest way is (for Simplified Chinese document only):

## Simplified Chinese with ctexart
```latex
\documentclass{ctexart}
 
\setCJKmainfont{simsun.ttf}
\setCJKsansfont{simhei.ttf}
\setCJKmonofont{simfang.ttf}
 
\begin{document}
 
\tableofcontents
 
\begin{abstract}

这是在文件的开头的介绍文字.本文的主要话题的简短说明.
\end{abstract}
 
\section{ 前言 }
在该第一部分中的一些额外的元素可以被添加。巴贝尔包将采取的翻译服务.
 
\section{关于数学部分}
在本节中的一些数学会使用数学模型含中文字符显示。
 
這是一個傳統的中國文字
 
\end{document}
```

```latex
% UTF-8 encoding
% Compile with latex+dvipdfmx, pdflatex or xelatex
% XeLaTeX is recommanded
% Some Chinese fonts should be installed in your system (SimSun, SimHei, FangSong, KaiTi)
\documentclass[UTF8]{ctexart}
\begin{document}

文章内容。
\end{document}
```

The document type is a **ctexart** (Chinese TeX Article), this is the 
recommended manner of typing Chinese documents, but is not the only one 
and may have some limitations. The next sections will clearly explain 
these and other environments for Chinese LATEX typesetting.


It is designed for Chinese typesetting. Font sizes, indentation, name translation, line spacing, ... everything is set.

```latex
\documentclass[UTF8,nofonts]{ctexart}

\setCJKmainfont{WenQuanYi Micro Hei}
\setCJKsansfont{文泉驿等宽微米黑}\setCJKmainfont{SimSun} % or any font you have.
\setCJKsansfont{SimHei}
\setCJKmonofont{FangSong}
\begin{document}

文章内容。
\end{document}
```
There might be some problems on Linux with default font setting (for windows). Then you can define the fonts mannually using xeCJK.


## Traditional and Simplified Chinese,the CJK package
If you just want to typeset only a few Chinese charecter, you can use CJK with pdfLaTeX or xeCJK with XeLaTeX.

### XeLaTeX

```latex
\documentclass{article}
 
\usepackage{xeCJK}
 
\setCJKmainfont{simsun.ttf}
\setCJKsansfont{simhei.ttf}
\setCJKmonofont{simfang.ttf}
 
\begin{document}
 
\section{前言}
在该第一部分中的一些额外的元素可以被添加。巴贝尔包将采取的翻译服务.
 
\section{关于数学部分}
在本节中的一些数学会使用数学模型含中文字符显示。
 
\vspace{0.5cm}
 
這是一個傳統的中國文字
\end{document}
```
The command \usepackage{xeCJK} imports xeCJK, this package allows to use external fonts in your document, these fonts are imported using the same syntax explained in the previous section. Again, if the imported font includes traditional symbols these can be used.

In this case elements are not translated as in the previous example, but sometimes the final rendered document may look a bit more sharp. Also, you can use any document class you want (book, report, article and so on) so your document layout is not constrained to a single document type.

The xeCJK package only works when compiled with XƎLATEX.


```latex
% Compile with xelatex
% UTF-8 encoding
\documentclass[a4paper]{article}
\usepackage{xeCJK}
\setmainfont{DejaVu Serif}
\setCJKmainfont{SimSun}
\setCJKsansfont{simhei.ttf}
\setCJKmonofont{simfang.ttf}
\begin{document}

文章内容
\end{document}
```
or

### pdfLaTex

```latex
% UTF-8 encoding, pdflatex or latex+dvipdfmx
% Simplified Chinese fonts should be installed
\documentclass{article}
\usepackage{CJKutf8}
\AtBeginDvi{\input{zhwinfonts}}

\begin{document}

\begin{CJK*}{UTF8}{zhsong}

文章内容。
\clearpage\end{CJK*}
\end{document}
```
or


The CJTK package can also be used to generate a document with pdfLaTeX. You may not be able to use external fonts, but here you can use traditional and simplified characters as well as Latin characters. Perfect for documents in English with bits of Chinese text or vice-versa.


```latex
\documentclass{article}
\usepackage{CJKutf8}
 
\begin{document}
 
\begin{CJK*}{UTF8}{gbsn}
 
\section{前言}
在该第一部分中的一些额外的元素可以被添加。巴贝尔包将采取的翻译服务.
 
\section{关于数学部分}
在本节中的一些数学会使用数学模型含中文字符显示。
 
\end{CJK*}
 
\vspace{0.5cm} % A white space
 
\noindent
You can also insert Latin text in your document
 
\vspace{0.5cm}
 
\noindent

\begin{CJK*}{UTF8}{bsmi}

這是一個傳統的中國文字
\end{CJK*}
 
\end{document}
```
The line **\usepackage{CJKutf8}** imports CJKutf8 which enables utf8 
encoding for Chinese, Japanese and Korean fonts.

In this case every block of Chinese text must be typed inside a 
**\begin{CJK*}{UTF8}{gbsm}** environment. In this environment *UTF8* 
is the encoding and *gbsm* is the font to be used. You can use *gbsm* or 
*gkai* fonts for simplified characters, and *bmsi* or *bkai* for 
traditional characters.
