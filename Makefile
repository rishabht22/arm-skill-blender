.PHONY: phase0-audit phase1-smoke sync audit-remote fetch-results

phase0-audit phase1-smoke sync audit-remote fetch-results:
	$(MAKE) -C projects/weave $@
