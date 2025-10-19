SNAKEMAKE ?= snakemake
SNAKEFILE ?= Snakefile

.PHONY: pipeline-run pipeline-status pipeline-graph pipeline-graph-png pipeline-shell

pipeline-run:
	$(SNAKEMAKE) --snakefile $(SNAKEFILE) --cores 1

pipeline-status:
	$(SNAKEMAKE) --snakefile $(SNAKEFILE) --summary

pipeline-graph:
	$(SNAKEMAKE) --snakefile $(SNAKEFILE) --dag > pipeline_graph.dot
	@echo "DOT graph written to pipeline_graph.dot"

pipeline-graph-png:
	$(SNAKEMAKE) --snakefile $(SNAKEFILE) --dag | dot -Tpng > pipeline_graph.png
	@echo "Pipeline graph rendered to pipeline_graph.png"

pipeline-shell:
	python scripts/pipeline_shell.py --include-openai
