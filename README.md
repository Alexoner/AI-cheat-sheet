Artificial Intelligence cheat sheet
===================================

This cheat sheet contains many formula equations,inference and diagrams on AI, which will help you quickly recall knowledge and ideas on machine learning.

The cheat sheet will also appeal to someone who is preparing for a job interview related to machine learning.

##Download PDF
[cheat-sheet.pdf](https://github.com/Alexoner/ai-cheat-sheet/raw/master/cheat-sheet.pdf) 

##LaTeX template
This open-source book adopts the [Springer latex template](http://www.springer.com/authors/book+authors?SGWID=0-154102-12-970131-0).

##How to compile
1. Install [Tex Live 2012](http://www.tug.org/texlive/).
2. Install [TeXstudio](http://texstudio.sourceforge.net/).
3. Configure TeXstudio.  
    Run TeXstudio, click `Options-->Configure Texstudio-->Commands`, set `XeLaTex` to `xelatex -synctex=1 -interaction=nonstopmode %.tex`.
    
    Click `Options-->Configure Texstudio-->Build`,   
    set `Build & View` to `Compile & View`,  
    set `Default Compiler` to `XeLaTex`,  
    set `PDF Viewer` to `Internal PDF Viewer(windowed)`, so that when previewing it will pop up a standalone window, which will be convenient.
4. Compile. Use Open `cheat-sheet.tex` with TeXstudio，click the green arrow on the menu bar, then it will start to compile.  
    In the messages window below we can see the compilation command that TeXstudio is using is `xelatex -synctex=1 --enable-write18 -interaction=nonstopmode "cheat-sheet".tex`
