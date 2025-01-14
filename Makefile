.ONESHELL:
.SHELLFLAGS = -ec
SHELL := /bin/bash

build-publish-docker-image:
	./build-publish-docker-image.sh

deploy-website-locally:
	@echo "Videos won't work. As the embdeded files feature of docsify does work with multiple videos,"
	@echo "they have to be embdeded with HTML, where the path is not the good one locally"
	cd website && python -m http.server 3000

force-online-website-to-update:
	curl https://glonet.lab.dive.edito.eu/update
