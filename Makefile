SHELL := /bin/bash
BUNDLE := bundle
YARN := yarn
VENDOR_DIR = assets/vendor/
JEKYLL := $(BUNDLE) exec jekyll

PROJECT_DEPS := Gemfile package.json

.PHONY: all clean install update

all : serve

check:
	$(JEKYLL) doctor
	$(HTMLPROOF) --check-html \
		--http-status-ignore 999 \
		--internal-domains localhost:4000 \
		--assume-extension \
		_site

install: $(PROJECT_DEPS)
	$(BUNDLE) install --path vendor/bundler
	$(YARN) install

update: $(PROJECT_DEPS)
	$(BUNDLE) update
	$(YARN) upgrade

include-yarn-deps:
	mkdir -p $(VENDOR_DIR)
	cp node_modules/jquery/dist/jquery.min.js $(VENDOR_DIR)
	cp node_modules/popper.js/dist/umd/popper.min.js $(VENDOR_DIR)
	cp node_modules/bootstrap/dist/js/bootstrap.min.js $(VENDOR_DIR)
	cp node_modules/anchor-js/anchor.min.js $(VENDOR_DIR)

build: install include-yarn-deps
	$(JEKYLL) build --config _config.yml
	rm _site/Gemfile
	rm _site/Gemfile.lock
	rm _site/Makefile
	rm _site/package.json
	rm -f _site/yarn-error.log
	rm _site/yarn.lock
	rm -rf _site/scripts

serve: install include-yarn-deps
	JEKYLL_ENV=development $(JEKYLL) serve --config _config.yml

build_deploy: include-yarn-deps
	JEKYLL_ENV=production $(JEKYLL) build
	rm _site/Gemfile
	rm _site/Gemfile.lock
	rm _site/Makefile
	rm _site/package.json
	rm -f _site/yarn-error.log
	rm _site/yarn.lock
	rm -rf _site/scripts
