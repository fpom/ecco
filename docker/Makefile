remote:
	docker run -p 8000:8000 franckpommereau/ecco jupyterhub

local: build
	docker run -p 8000:8000 ecco jupyterhub

build:
	cp -af ../doc .
	docker build -t ecco .
	rm -rf doc
	docker tag ecco franckpommereau/ecco

push:
	docker push franckpommereau/ecco
