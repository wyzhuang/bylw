%% This is file SUDAthesis.cls based on ...
%% This is file `CASthesis.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% CASthesis.dtx  (with options: `cls')
%% 
%% This is a generated file.
%% 
%% Copyright 2004-2005
%% CTEX.ORG and any individual authors listed elsewhere in this file.
%% 
%% This file was generated from file(s) of the CASthesis package project.
%% ----------------------------------------------------------------------
%% 
%% It may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.2
%% of this license or (at your option) any later version.
%% The latest version of this license is in
%%    http://www.latex-project.org/lppl.txt
%% and version 1.2 or later is part of all distributions of LaTeX
%% version 1999/12/01 or later.
%% 
%% This file may only be distributed together with a copy of original
%% source files. You may however distribute original source files
%% without such generated files. Copying of this file is authorized
%% only if either:
%% (1) you make absolutely no changes to your copy, including name; OR
%% (2) if you do make changes, you first rename it to some other name.
%% 
%% To produce the documentation run the original source files ending
%% with `.dtx' through LaTeX.
%% 
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}
%%

\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{SUDAthesis}
  [2005/10/11 v0.1e CASthesis
   document class]
\newif\ifCAST@typeinfo \CAST@typeinfotrue
\DeclareOption{notypeinfo}{\CAST@typeinfofalse}
\newif\ifCAST@dvips \CAST@dvipstrue
\DeclareOption{dvips}{\CAST@dvipstrue}
\DeclareOption{dvipdfm}{\CAST@dvipsfalse}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{ctexbook}}
\ProcessOptions
\LoadClass[cs4size,a4paper,fancyhdr]{ctexbook}[2004/08/14]
%% math packages
\RequirePackage{amsmath,amsthm,amsfonts,amssymb,bm}
%% graphics packages
\RequirePackage{graphicx,psfrag}

%%wuxian added
%\ifCLASSINFOpdf
%\typeout{-- Using PDF output.}
%\else
%\typeout{-- Using DVI output.}
%\fi
%%wuxian added ended

%% hyperref package
\ifx\pdfoutput\undefined % We're not running pdftex
  \ifCAST@dvips
    \RequirePackage{hyperref}
  \else
    \RequirePackage[dvipdfm]{hyperref}
  \fi
  \AtBeginDvi{\special{pdf:tounicode GBK-EUC-UCS2}} % GBK -> Unicode
\else
  %wx \RequirePackage[pdftex]{hyperref}
  \RequirePackage[pdftex,plainpages=false]{hyperref} %wx Add
\fi
\hypersetup{CJKbookmarks,%
  bookmarksnumbered,%
  colorlinks,%
  linkcolor=black,%
  citecolor=black,%
  plainpages=false,%
  pdfstartview=FitH}
\AtBeginDocument{\makeatletter\input{SUDAthesis.cfg}\makeatother}
\oddsidemargin 0 true cm
\if@twoside
  \evensidemargin 0 true cm
\else
  \evensidemargin 0 true cm
\fi
\textheight 22.5 true cm
\textwidth 15.5 true cm
\topmargin = -5mm
\headheight 13pt
\parskip 0.5ex plus 0.25ex minus 0.25ex
\def\cleardoublepage{\clearpage\if@twoside \ifodd\c@page\else
  \thispagestyle{empty}%
  \hbox{}\newpage\if@twocolumn\hbox{}\newpage\fi\fi\fi}
\renewcommand{\floatpagefraction}{0.80}
%\bibliographystyle{plain}
\newcommand\CASTspace{\protect\CTEX@spaceChar\protect\CTEX@spaceChar}
\newcommand{\fancyheadzihao}{\zihao{-5}}
\def\CAST@getfileinfo#1 #2 #3\relax#4\relax{%
  \def\CASTfiledate{#1}%
  \def\CASTfileversion{#2}%
  \def\CASTfileinfo{#3}}%
\expandafter\ifx\csname ver@CASthesis.cls\endcsname\relax
  \edef\reserved@a{\csname ver@ctextemp_CASthesis.cls\endcsname}
\else
  \edef\reserved@a{\csname ver@CASthesis.cls\endcsname}
\fi
\expandafter\CAST@getfileinfo\reserved@a\relax? ? \relax\relax
\def\CAST@underline[#1]#2{%
  \underline{\hbox to #1{\hfill#2\hfill}}}
\def\CASTunderline{\@ifnextchar[\CAST@underline\underline}
\newcommand\classification[1]{\def\CAST@value@classification{#1}}
\newcommand\confidential[1]{\def\CAST@value@confidential{#1}}
\newcommand\UDC[1]{\def\CAST@value@UDC{#1}}
\newcommand\serialnumber[1]{\def\CAST@value@serialnumber{#1}}
\newcommand\school[1]{\def\CAST@value@school{#1}}
\newcommand\degree[1]{\def\CAST@value@degree{#1}}
\renewcommand\title[2][\CAST@value@title]{%
  \def\CAST@value@title{#2}
  \def\CAST@value@titlemark{\MakeUppercase{#1}}}
\renewcommand\author[1]{\def\CAST@value@author{#1}}
\newcommand\advisor[1]{\def\CAST@value@advisor{#1}}
\newcommand\advisorinstitute[1]{\def\CAST@value@advisorinstitute{#1}}
\newcommand\major[1]{\def\CAST@value@major{#1}}
\newcommand\submitdate[1]{\def\CAST@value@submitdate{#1}}
\newcommand\defenddate[1]{\def\CAST@value@defenddate{#1}}
\newcommand\institute[1]{\def\CAST@value@institute{#1}}
\newcommand\chairman[1]{\def\CAST@value@chairman{#1}}
\renewcommand\maketitle{%
  \cleardoublepage
  \thispagestyle{empty}
  \begin{center}
    \bf\songti\zihao{-4}
      \CAST@label@classification~
      \CASTunderline[100pt]{\CAST@value@classification}
        \hfill
      \CAST@label@confidential~
      \CASTunderline[100pt]{\CAST@value@confidential}
    \vskip 10pt
      \CAST@label@UDC~
      \CASTunderline[108pt]{\CAST@value@UDC}
        \hfill
      \CAST@label@serialnumber~
      \CASTunderline[100pt]{\CAST@value@serialnumber}
    \vskip \stretch{2}
      \bf\songti\zihao{-1} \CAST@value@school \\
      \CAST@value@degree\CAST@label@thesis
    \vskip \stretch{2}
      \bf\heiti\zihao{3} \CASTunderline{~\CAST@value@title~}
    \vskip \stretch{1}
      \bf\kaishu\zihao{4} \CASTunderline{~\CAST@value@author~}
    \vskip \stretch{2}
    \bf\fangsong\zihao{4}
    \def\tabcolsep{1pt}
    \def\arraystretch{1.5}
    \begin{tabular}{llcrl}
      \CAST@label@advisor &
      \multicolumn{4}{l}{\CASTunderline[299pt]{\CAST@value@advisor}}
    \\
      & \multicolumn{4}{l}{\CASTunderline[299pt]{\CAST@value@advisorinstitute}}
    \\
      \CAST@label@degree &
      \CASTunderline[80pt]{\CAST@value@degree} & \hspace{5pt} &
      \CAST@label@major &
      \CASTunderline[120pt]{\CAST@value@major}
    \\
      \CAST@label@submitdate &
      \CASTunderline[80pt]{\CAST@value@submitdate} & &
      \CAST@label@defenddate &
      \CASTunderline[120pt]{\CAST@value@defenddate}
    \\
      \CAST@label@institute &
      \multicolumn{4}{l}{\CASTunderline[299pt]{\CAST@value@institute}}
    \\
      \CAST@label@school &
      \multicolumn{4}{l}{\CASTunderline[299pt]{\CAST@value@school}}
    \\[15pt]
      \multicolumn{4}{r}{\CAST@label@chairman} &
      \CASTunderline[100pt]{\CAST@value@chairman}
    \end{tabular}
  \end{center}
  \clearpage
  \if@twoside
    \thispagestyle{empty}
    \ifCAST@typeinfo
      \vspace*{\stretch{1}}
      \begin{footnotesize}
        \noindent
        Typeset by \LaTeXe{} at \CTEX@todayold \\
        With package \texttt{CASthesis} \CASTfileversion{} of C\TeX{}.ORG
      \end{footnotesize}
    \fi
    \cleardoublepage
  \fi
}
\newcommand\englishtitle[1]{\def\CAST@value@englishtitle{#1}}
\newcommand\englishfancytitle[1]{\def\CAST@value@englishfancytitle{#1}}
\newcommand\englishAbstractTitle[1]{\def\CAST@value@englishAbstractTitle{#1}}
\newcommand\englishauthor[1]{\def\CAST@value@englishauthor{#1}}
\newcommand\englishadvisor[1]{\def\CAST@value@englishadvisor{#1}}
\newcommand\englishinstitute[1]{\def\CAST@value@englishinstitute{#1}}
\newcommand\englishdate[1]{\def\CAST@value@englishdate{#1}}
\newcommand\englishdegree[1]{\def\CAST@value@englishdegree{#1}}
\newcommand\englishmajor[1]{\def\CAST@value@englishmajor{#1}}
\newcommand\makeenglishtitle{%
  \cleardoublepage
  \thispagestyle{empty}
  \begin{center}
    \vspace*{20pt}
      \sf\zihao{-1} \CAST@value@englishtitle
    \vskip \stretch{1}
      \bf\zihao{4} \CAST@value@englishauthor
    \vskip \stretch{1}
      \normalfont\zihao{4} \CAST@label@englishadvisor
    \vskip 3pt
      \normalfont\zihao{4} \CAST@value@englishadvisor
    \vskip \stretch{2}
      \normalfont\normalsize \CAST@value@englishinstitute
    \vskip 30pt
      \normalfont\normalsize \CAST@value@englishdate
    \vskip 20pt
      \it\normalsize \CAST@label@englishstatement
  \end{center}
  \clearpage
  \if@twoside
    \thispagestyle{empty}
    \cleardoublepage
  \fi
}

\newcommand\setstyle{
    \fancypagestyle{plain}{
      \fancyhf{}
      \fancyhead[LO,RE]{\songti \fancyheadzihao \CAST@value@titlemark}
      \fancyhead[RO,LE]{\songti \fancyheadzihao \leftmark}
      \fancyfoot[C]{\fancyheadzihao ~\thepage~}
    }
    \pagestyle{fancy}
    \fancyhf{}
    \fancyhead[LO,RE]{\songti \fancyheadzihao \CAST@value@titlemark}
    \fancyhead[RO,LE]{\songti \fancyheadzihao \leftmark}
    \fancyfoot[C]{\fancyheadzihao ~\thepage~}
    \renewcommand{\headrulewidth}{0.4pt}
}

\newcommand\setAbstractStyle{
  \fancypagestyle{plain}{
    \fancyhf{}
    \fancyhead[LO]{\fancyheadzihao \CAST@value@titlemark}
    \fancyhead[RE]{\fancyheadzihao \CAST@value@titlemark}
    %\fancyhead[RE]{\small \CAST@value@englishfancytitle }
    \fancyhead[RO,LE]{\fancyheadzihao \leftmark}
    \fancyfoot[C]{\fancyfoot ~\thepage~}
  }
  \pagestyle{fancy}
  \fancyhf{}
  \fancyhead[LO]{\fancyheadzihao \CAST@value@titlemark}
  \fancyhead[RE]{\fancyheadzihao \CAST@value@titlemark}
  %\fancyhead[RE,LO]{\small \CAST@value@englishfancytitle }
  \fancyhead[RO,LE]{\fancyheadzihao \leftmark}
  \fancyfoot[C]{\fancyheadzihao ~\thepage~}
  \renewcommand{\headrulewidth}{0.4pt}
}

\newcommand\setEnAbstractStyle{
  \fancypagestyle{plain}{
    \fancyhf{}
    \fancyhead[LO]{\fancyheadzihao \CAST@value@englishfancytitle}
    \fancyhead[RE]{\fancyheadzihao \CAST@value@englishfancytitle }
    \fancyhead[RO,LE]{\fancyheadzihao \leftmark}
    \fancyfoot[C]{\fancyheadzihao ~\thepage~}
  }
  \pagestyle{fancy}
  \fancyhf{}
  \fancyhead[LO]{\fancyheadzihao \CAST@value@englishfancytitle}
  \fancyhead[RE]{\fancyheadzihao \CAST@value@englishfancytitle}
  \fancyhead[RO,LE]{\fancyheadzihao \leftmark}
  \fancyfoot[C]{\fancyheadzihao ~\thepage~}
  \renewcommand{\headrulewidth}{0.4pt}
}

\newcommand\setContentsStyle{
  
    \fancypagestyle{plain}{
      \fancyhf{}
      \fancyhead[LO,RE]{\fancyheadzihao }
      \fancyhead[RO,LE]{\fancyheadzihao }
      \fancyfoot[C]{}
      \renewcommand{\headrulewidth}{0pt}
    }
    \pagestyle{fancy}
    \fancyhf{}
    \fancyhead[LO,RE]{\fancyheadzihao }
    \fancyhead[RO,LE]{\fancyheadzihao }
    \fancyfoot[C]{}
    \renewcommand{\headrulewidth}{0pt}
} 

\newcommand\setThanksStyle{
    \fancypagestyle{plain}{
      \fancyhf{}
      \fancyhead[LO,RE]{\songti \fancyheadzihao \CAST@value@titlemark}
      \fancyhead[RO,LE]{\songti \fancyheadzihao 致谢}
      %\fancyhead[RO,LE]{\songti \small \leftmark}
      \fancyfoot[C]{\fancyheadzihao ~\thepage~}
    }
    \pagestyle{fancy}
    \fancyhf{}
    \fancyhead[LO,RE]{\songti \fancyheadzihao \CAST@value@titlemark}
    \fancyhead[RO,LE]{\songti \fancyheadzihao 致谢}
    %\fancyhead[RO,LE]{\songti \small \leftmark}
    \fancyfoot[C]{\fancyheadzihao ~\thepage~}
    \renewcommand{\headrulewidth}{0.4pt}
}

%\setstyle

\newenvironment{abstract}{
  %\notocchapter{\CAST@label@abstract}
  \notocchapter{\CAST@value@titlemark \\ \CAST@label@abstract}
  \chaptermark{摘要}
  \pagenumbering{Roman}
  }{}
\newcommand\keywords[1]{%
  \vspace{2ex}\noindent{\heiti \CAST@label@keywords} #1}
  
\newenvironment{englishabstract}{
  %\isenglishabstract
  %\notocchapter{\CAST@label@englishabstract}
  \notocchapter{\CAST@value@englishAbstractTitle \\ \CAST@label@englishabstract}
  \chaptermark{\CAST@label@englishabstract}

  }{}

\newcommand\englishkeywords[1]{%
  \vspace{2ex}\noindent{\bf \CAST@label@englishkeywords} #1}
  
  
\renewcommand\tableofcontents{%
  
  \fancypagestyle{plain}{
    \fancyhf{}
    \fancyfoot[C]{}
  }
 
  \fancyhead[RO,LE]{}
   \fancyfoot[C]{}
 
  
  \if@twocolumn
    \@restonecoltrue\onecolumn
  \else
    \@restonecolfalse
  \fi
  \notocchapter{\contentsname}%
  \@mkboth{\MakeUppercase\contentsname}{\MakeUppercase\contentsname}%
  \@starttoc{toc}%
  \if@restonecol\twocolumn\fi
}

\setstyle

\addtocontents{toc}{\let\string\CTEX@spaceChar\relax}

\newcommand\notocchapter[1]{%
  \if@mainmatter%
    \@mainmatterfalse%
    \chapter*{#1}%
    \@mainmattertrue%
  \else
    \chapter*{#1}%
  \fi
}

\newcommand\Nchapter[1]{%
  \if@mainmatter%
    \@mainmatterfalse%
    \chapter{#1}%
    \@mainmattertrue%
  \else
    \chapter{#1}%
  \fi
}

\def\@dottedtocline#1#2#3#4#5{%
  \ifnum #1>\c@tocdepth \else
    \vskip \z@ \@plus.2\p@
    {\leftskip #2\relax \rightskip \@tocrmarg \parfillskip -\rightskip
     \parindent #2\relax\@afterindenttrue
     \interlinepenalty\@M
     \leavevmode
     \@tempdima #3\relax
     \advance\leftskip \@tempdima \null\nobreak\hskip -\leftskip
     {#4}\nobreak
     \leaders\hbox{$\m@th\mkern 1.5mu\cdot\mkern 1.5mu$}\hfill
     \nobreak
     \hb@xt@\@pnumwidth{\hfil\normalfont \normalcolor #5}%
     \par}%
  \fi}
\renewcommand*\l@part[2]{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty{-\@highpenalty}%
    \addvspace{2.25em \@plus\p@}%
    \setlength\@tempdima{3em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      {\leavevmode
       \large \bfseries #1
       \leaders\hbox{$\m@th\mkern 1.5mu\cdot\mkern 1.5mu$}
       \hfil \hb@xt@\@pnumwidth{\hss #2}}\par
       \nobreak
         \global\@nobreaktrue
         \everypar{\global\@nobreakfalse\everypar{}}%
    \endgroup
  \fi}
\renewcommand*\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak
      \leaders\hbox{$\m@th\mkern 1.5mu\cdot\mkern 1.5mu$}
      \hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}

\renewenvironment{thebibliography}[1]
     {\Nchapter{\bibname}%
      \@mkboth{\MakeUppercase\bibname}{\MakeUppercase\bibname}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
\newenvironment{publications}[1]
     {\Nchapter{\CAST@label@publications}%
      \@mkboth{\MakeUppercase\CAST@label@publications}
              {\MakeUppercase\CAST@label@publications}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `publications' environment}}%
      \endlist}
\newenvironment{resume}
  {\Nchapter{\CAST@label@resume}}
  {}
\newenvironment{resumesection}[1]
  {{\noindent\normalfont\bfseries #1}
   \list{}{\labelwidth\z@
           \leftmargin 2\ccwd}
   \item\relax}
  {\endlist}
\newenvironment{resumelist}[1]
  {{\noindent\normalfont\bfseries #1}
   \list{}{\labelwidth\z@
           \leftmargin 4\ccwd
           \itemindent -2\ccwd
           \listparindent\itemindent}
   \item\relax}
  {\endlist}
\renewenvironment{thanks}
  {\Nchapter{\CAST@label@thanks}}
  {}
\setcounter{secnumdepth}{4}
\newtheoremstyle{break}% name
  {}%      Space above, empty = `usual value'
  {}%      Space below
  {\itshape}% Body font
  {}%         Indent amount (empty = no indent, \parindent = para indent)
  {\bfseries}% Thm head font
  {.}%        Punctuation after thm head
  {\newline}% Space after thm head: \newline = linebreak
  {}%         Thm head spec

%%soochow
\newcommand\minor[1]{\def\CAST@value@minor{#1}}
\newcommand\grade[1]{\def\CAST@value@grade{#1}}

\newcommand\soochowauthor[1]{%
  \vspace{4ex}\noindent { \hfill \textbf \CAST@label@sooauthor} #1 }
\newcommand\soochowtutor[1]{%
  \vspace{2ex}\noindent { \hfill \textbf \CAST@label@sootutor} #1 } 

\newcommand\ensoochowauthor[1]{%
  \vspace{2ex}\noindent{\hfill \textbf \CAST@label@ensooauthor} #1 }
\newcommand\ensoochowtutor[1]{%
  \vspace{2ex}\noindent{\hfill \textbf \CAST@label@ensootutor} #1 }

\newcommand\Soounderline{\@ifnextchar[\CAST@underline\underline}
%%
% SUDA thesis cover page
%------------------------论文封面---------------------
\newcommand\CoverPage{%
\newpage
\thispagestyle{empty}
\begin{figure}[t]
  \centering
  \includegraphics[width=4in,]{soochowLogo.pdf}\\
\end{figure}

\vskip 3mm

\begin{center}
\zihao{1}\heiti\bf{\CAST@value@degree 学~位~论~文}

\vskip 5mm

\Huge\songti{(\CAST@value@grade)}
\end{center}

\vskip 1cm
\begin{center}
\huge\heiti\bf \CAST@value@titlemark
\end{center}
%\vskip 0.5cm

\begin{center}
\zihao{4}\bf \CAST@value@englishtitle
\end{center}

\vskip 2cm
\begin{center}
\large
    \begin{tabular}{lc}

      研~究~生~姓~名~ &
     \Soounderline[250pt] {\kaishu \CAST@value@author}\\ & \\

      指导教师姓名 &
     \Soounderline[250pt]{ \kaishu \CAST@value@advisor}\\ & \\
      专~~~业~~~名~~~称 &
     \Soounderline[250pt]{\kaishu \CAST@value@major}\\ & \\
      研~~~究~~~方~~~向 &
     \Soounderline[250pt]{\kaishu \CAST@value@minor}\\ & \\
      论文提交日期 &
     \Soounderline[250pt]{\kaishu \CAST@value@submitdate}
    \end{tabular}
\end{center}
  \clearpage
  \if@twoside
    \thispagestyle{empty}
    \ifCAST@typeinfo
      \vspace*{\stretch{1}}
      \begin{footnotesize}
        \noindent
        %Typeset by \LaTeXe{} at \CTEX@todayold \\
        %With package \texttt{Sudathesis 1.0} based on \texttt{CASthesis 0.1e} of C\TeX{}.ORG
      \end{footnotesize}
    \fi
    \cleardoublepage
  \fi
}

%%Show SUDA Copyright page
\newcommand\CopyrightPage{%
  \cleardoublepage
  \thispagestyle{empty}
  \begin{center}
    \vspace*{0pt}
     \bfseries\heiti\zihao{-2}  \CAST@label@sooundeclear~
    \vskip 30pt
      \bf\normalfont\zihao{3} \CAST@label@undeclear~
    \vskip 30pt
  \end{center}   
      \noindent\normalfont\normalsize\CAST@label@declear~
  \begin{center} 
    \vskip 20pt
     \hfill \normalfont\normalsize \CAST@label@mastename {\CASTunderline[100pt]{\CAST@value@chairman}} \CAST@label@soodata {\CASTunderline[100pt]{\CAST@value@chairman}}   
    \vskip 50pt
      \bf\normalfont\zihao{3} \CAST@label@usedeclear~
  \end{center}       
    \vskip 20pt
      \noindent\normalfont\normalsize \CAST@label@usedeclearcontent~
    \vskip 20pt
  \begin{center}  
      \vskip 20pt
      \hfill \CAST@label@mastename {\CASTunderline[100pt]{\CAST@value@chairman}} \CAST@label@soodata {\CASTunderline[100pt]{\CAST@value@chairman}}   
      \vskip 20pt
      \hfill \CAST@label@tutorname {\CASTunderline[100pt]{\CAST@value@chairman}} \CAST@label@soodata {\CASTunderline[100pt]{\CAST@value@chairman}}   
  \end{center}
  \clearpage
  \if@twoside
    \thispagestyle{empty}
    \cleardoublepage
  \fi
}

%%\newcommand{\xiaosihao}{\fontsize{12pt}{\baselineskip}\selectfont}%%小四号
%%\newcommand{\sihao}{\fontsize{14pt}{\baselineskip}\selectfont} %%四号 
%\renewcommand\tablename{\xiaosihao 表}
%\renewcommand\figurename{\xiaosihao 图}
\renewcommand{\thetable}{\arabic{chapter}-\arabic{table}}
\renewcommand{\thefigure}{\arabic{chapter}-\arabic{figure}}

\newcommand{\xiaosihao}{\fontsize{12pt}{18pt}\selectfont}%%小四号
\newcommand{\sihao}{\fontsize{14pt}{18pt}\selectfont} %%四号 
\renewcommand\tablename{\bf\wuhao 表}
\renewcommand\figurename{\bf\wuhao 图}
\newcommand{\wuhao}{\fontsize{10.5pt}{14pt}\selectfont}%%五号
\newcommand {\smallcaption}[1]{\caption{\bf\wuhao #1} \wuhao}  %设置表、图的标题为五号
\newcommand\supercite[1]{\textsuperscript{\cite{#1}}}      %设置引用为上标
\makeatletter
\def\fnum@figure#1{\figurename\nobreakspace\thefigure\hspace{0.1em}} %去除图后面的冒号并加上空格
\def\fnum@table#1{\tablename\nobreakspace\thetable\hspace{0.1em}} %去除表后面的冒号并加上空格
\makeatother
\endinput
%%
