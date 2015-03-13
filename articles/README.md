The easiest way is (for Simplified Chinese document only):

% UTF-8 encoding
% Compile with latex+dvipdfmx, pdflatex or xelatex
% XeLaTeX is recommanded
% Some Chinese fonts should be installed in your system (SimSun, SimHei, FangSong, KaiTi)
\documentclass[UTF8]{ctexart}
\begin{document}
文章内容。
\end{document}
It is designed for Chinese typesetting. Font sizes, indentation, name translation, line spacing, ... everything is set.

There might be some problems on Linux with default font setting (for windows). Then you can define the fonts mannually using xeCJK:

\documentclass[UTF8,nofonts]{ctexart}
\setCJKmainfont{SimSun} % or any font you have.
\setCJKsansfont{SimHei}
\setCJKmonofont{FangSong}
\begin{document}
文章内容。
\end{document}
If you just want to typeset only a few Chinese charecter, you can use CJK with pdfLaTeX or xeCJK with XeLaTeX.

% Compile with xelatex
% UTF-8 encoding
\documentclass{article}
\usepackage{xeCJK}
\setCJKmainfont{SimSun}
\begin{document}
文章内容
\end{document}
or

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
or

% UTF-8 encoding
% bad-looking fonts (CJKfonts package)
% latex+dvips, latex+dvipdfm(x) or pdflatex
\documentclass{article}
\usepackage{CJKutf8}
\begin{document}
\begin{CJK*}{UTF8}{gbsn}
文章内容。
\clearpage\end{CJK*}
\end{document}
