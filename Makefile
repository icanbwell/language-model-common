LANG=en_US.utf-8

export LANG

.PHONY: Pipfile.lock
Pipfile.lock: ## Locks Pipfile and updates the Pipfile.lock on the local file system
	/bin/sh -c '\
		if [ -z "$$JFROG_READ_USER" ] || [ -z "$$JFROG_READ_TOKEN" ]; then \
			echo "ERROR: JFROG_READ_USER and JFROG_READ_TOKEN must be set."; \
			exit 1; \
		fi; \
		export DOCKER_BUILDKIT=1; \
		docker build --no-cache --build-arg RUN_PIPENV_LOCK=true \
			--secret id=jfrog_user,env=JFROG_READ_USER \
			--secret id=jfrog_token,env=JFROG_READ_TOKEN \
			-t languagemodelcommon:local . && \
		CONTAINER_ID=$$(docker create languagemodelcommon:local); \
		docker cp $$CONTAINER_ID:/tmp/Pipfile.lock Pipfile.lock && \
		docker rm $$CONTAINER_ID \
	'

.PHONY:devdocker
devdocker: ## Builds the docker for dev
	/bin/sh -c '\
		if [ -z "$$JFROG_READ_USER" ] || [ -z "$$JFROG_READ_TOKEN" ]; then \
			echo "ERROR: JFROG_READ_USER and JFROG_READ_TOKEN must be set."; \
			exit 1; \
		fi; \
		docker compose build \
	'

.PHONY:init
init: Pipfile.lock devdocker up setup-pre-commit  ## Initializes the local developer environment

.PHONY: up
up:
	docker compose up --build -d --remove-orphans

.PHONY: down
down:
	docker compose down

.PHONY:clean-pre-commit
clean-pre-commit: ## removes pre-commit hook
	rm -f .git/hooks/pre-commit

.PHONY:setup-pre-commit
setup-pre-commit:
	cp ./pre-commit-hook ./.git/hooks/pre-commit && \
	chmod +x ./.git/hooks/pre-commit

.PHONY:run-pre-commit
run-pre-commit: setup-pre-commit
	./.git/hooks/pre-commit pre_commit_all_files

.PHONY:update
update: down Pipfile.lock setup-pre-commit  ## Updates all the packages using Pipfile
	make devdocker && \
	make pipenv-setup

.PHONY:tests
tests: up
	docker compose run --rm --name languagemodelcommon dev pytest tests languagemodelcommon

.PHONY:shell
shell:devdocker ## Brings up the bash shell in dev docker
	docker compose run --rm --name languagemodelcommon dev sh

.PHONY:build
build:
	docker compose run --rm --name languagemodelcommon dev rm -rf dist/
	docker compose run --rm --name languagemodelcommon dev python3 setup.py sdist bdist_wheel

.PHONY:testpackage
testpackage:build
	docker compose run --rm --name languagemodelcommon dev python3 -m twine upload -u __token__ --repository testpypi dist/*
# password can be set in TWINE_PASSWORD. https://twine.readthedocs.io/en/latest/

.PHONY:package
package:build
	docker compose run --rm --name languagemodelcommon dev python3 -m twine upload -u __token__ --repository pypi dist/*
# password can be set in TWINE_PASSWORD. https://twine.readthedocs.io/en/latest/ (note this is the token not your password)

.DEFAULT_GOAL := help
.PHONY: help
help: ## Show this help.
	# from https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY:pipenv-setup
pipenv-setup:devdocker ## Run pipenv-setup to update setup.py with latest dependencies
	docker compose run --rm dev sh -c "pipenv run pipenv install --skip-lock --categories \"pipenvsetup\" && pipenv run pipenv-setup sync --pipfile" && \
	make run-pre-commit
