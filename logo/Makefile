all: ecco.pdf ecco.png ecco.xpm

logo.pdf: logo.tex
	pdflatex logo

ecco.pdf: logo.pdf
	pdfcrop --hires logo.pdf ecco.pdf

ecco.xpm: ecco.pdf
	pdftoppm ecco.pdf ecco
	ppmtoxpm ecco-1.ppm > ecco.xpm
	rm ecco-1.ppm

ecco.png: ecco.pdf
	pdftocairo -r 500 -png ecco.pdf
	mv ecco-1.png ecco.png
