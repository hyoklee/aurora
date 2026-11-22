# QICK readout reproducibility bench (CLIO vs Genesis on ALCF models)

Question: can a second team reproduce the Fig. 14 readout plot and its 91.7 % fidelity
(FERMILAB-PUB-26-0193) from `data/qick/2025-10-19_12-46-02_ss_ge_results_batch_1_Num_per_batch1.h5`?

Setup (each run gets a fresh workspace with `ss_ge.h5` + the paper text, outside the repo):

    pdftotext -layout fermilab-pub-26-0193-csaid-etd-ppd.pdf fermilab-pub-26-0193.txt
    source bench.env
    ./run_coder.sh   <model>      # CLIO Coder  (clio-coder run, target alcf-tara)
    ./run_agent_all.sh <models..> # CLIO Agent  (clio-agent server + clio-core, driven by clio_agent_drive.py)
    ./run_genesis.sh <model>      # Genesis Agent = Claude Code + Genesis Agent Skills on ALCF Tara
    python3 collect.py            # -> results.json, answers/*.md

`prompt.txt` is the exact prompt. `invalid/` keeps first-attempt runs that are excluded
from scoring (workspace was inside the repo, so agents could see CLAUDE.md and the
ground-truth script `src/qick/ss_fidelity.py`). See `doc/20261121_qick_reproducibility_alcf_agents.md`.
