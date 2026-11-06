WEAVE_REMOTE ?=
WEAVE_REMOTE_DIR ?= arm-skill-blender

.PHONY: phase0-audit sync audit-remote fetch-results
phase0-audit:
	./scripts/run_phase0.sh

sync:
	@test -n "$(WEAVE_REMOTE)" || { echo "Set WEAVE_REMOTE to your SSH host alias."; exit 1; }
	rsync -azR --exclude='__pycache__' configs docs scripts src/weave \
		src/skill_composer/__init__.py src/skill_composer/data.py src/skill_composer/common.py \
		Makefile PROGRESS.md DECISIONS.md RISKS.md "$(WEAVE_REMOTE):$(WEAVE_REMOTE_DIR)/"
	@revision=$$(git rev-parse HEAD) && printf "%s\n" "$$revision" | \
		ssh "$(WEAVE_REMOTE)" 'cd "$(WEAVE_REMOTE_DIR)" && cat > .weave-source-revision'

audit-remote:
	@test -n "$(WEAVE_REMOTE)" || { echo "Set WEAVE_REMOTE to your SSH host alias."; exit 1; }
	ssh "$(WEAVE_REMOTE)" 'cd "$(WEAVE_REMOTE_DIR)" && ./scripts/run_phase0.sh'

fetch-results:
	@test -n "$(WEAVE_REMOTE)" || { echo "Set WEAVE_REMOTE to your SSH host alias."; exit 1; }
	rsync -az "$(WEAVE_REMOTE):$(WEAVE_REMOTE_DIR)/results/" results/
