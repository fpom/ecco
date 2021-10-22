VERSION=$$(git tag|tail -n1)

remote:
	docker run -p 8000:8000 franckpommereau/ecco jupyterhub

local: build
	docker run -p 8000:8000 ecco jupyterhub

build:
	cp -af ../doc .
	docker build -t ecco .
	rm -rf doc
	docker tag ecco ecco:${VERSION}

rebuild:
	cp -af ../doc .
	docker build --no-cache -t ecco .
	rm -rf doc
	docker tag ecco ecco:${VERSION}

push:
	docker tag ecco franckpommereau/ecco
	docker push franckpommereau/ecco
	docker tag ecco:${VERSION} franckpommereau/ecco:${VERSION}
	docker push franckpommereau/ecco:${VERSION}
