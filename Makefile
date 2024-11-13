VERSION=$$(git tag|tail -n1)

lark: ecco/mrr/mrrparse.py ecco/mrr/pattparse.py

ecco/mrr/mrrparse.py: ecco/mrr/mrr.lark
	python -m lark.tools.standalone --start start --maybe_placeholders $< > $@

ecco/mrr/pattparse.py: ecco/mrr/mrr.lark
	python -m lark.tools.standalone --start pattern --maybe_placeholders $< > $@

remote:
	docker run -p 8000:8000 franckpommereau/ecco bash --login -c "jupyter-lab --no-browser --ip=0.0.0.0 --port=8000"

local:
	docker run -p 8000:8000 -u ecco -w /home/ecco ecco bash --login -c "jupyter-lab --no-browser --ip=0.0.0.0 --port=8000"

build:
	docker build -t ecco .
	docker tag ecco ecco:${VERSION}

rebuild:
	docker build --no-cache -t ecco .
	docker tag ecco ecco:${VERSION}

push:
	docker tag ecco franckpommereau/ecco
	docker push franckpommereau/ecco
	docker tag ecco:${VERSION} franckpommereau/ecco:${VERSION}
	docker push franckpommereau/ecco:${VERSION}
