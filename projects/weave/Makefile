WEAVE_REMOTE ?=
WEAVE_REMOTE_DIR ?= arm-skill-blender

.PHONY: phase0-audit phase1-smoke sync audit-remote fetch-results
phase0-audit:
	./scripts/run_phase0.sh

phase1-smoke:
	./scripts/run_phase1.sh

sync:
	@test -n "$(WEAVE_REMOTE)" || { echo "Set WEAVE_REMOTE to your SSH host alias."; exit 1; }
	cd ../.. && rsync -azR --exclude='__pycache__' \
		projects/weave/configs projects/weave/docs projects/weave/scripts projects/weave/src \
		projects/weave/Makefile projects/weave/PROGRESS.md projects/weave/DECISIONS.md projects/weave/RISKS.md \
		projects/robofactory_lift/src/skill_composer/__init__.py \
		projects/robofactory_lift/src/skill_composer/data.py \
		projects/robofactory_lift/src/skill_composer/common.py \
		"$(WEAVE_REMOTE):$(WEAVE_REMOTE_DIR)/"
	@revision=$$(git rev-parse HEAD) && printf "%s\n" "$$revision" | \
		ssh "$(WEAVE_REMOTE)" 'cd "$(WEAVE_REMOTE_DIR)/projects/weave" && cat > .weave-source-revision'

audit-remote:
	@test -n "$(WEAVE_REMOTE)" || { echo "Set WEAVE_REMOTE to your SSH host alias."; exit 1; }
	ssh "$(WEAVE_REMOTE)" 'cd "$(WEAVE_REMOTE_DIR)/projects/weave" && ./scripts/run_phase0.sh'

fetch-results:
	@test -n "$(WEAVE_REMOTE)" || { echo "Set WEAVE_REMOTE to your SSH host alias."; exit 1; }
	rsync -az "$(WEAVE_REMOTE):$(WEAVE_REMOTE_DIR)/projects/weave/results/" results/
