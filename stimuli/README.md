# Stimuli

This directory contains the stimulus construction and validation scripts for the MEG/EEG specificity experiment.

## Scripts

**`specificity_stim_properties.ipynb`**
Builds the final stimulus set from the hand-curated specificity triads. Looks up psycholinguistic properties for each target noun from published norm databases (SUBTLEX-UK, Brysbaert et al. 2014, Scott et al. 2019, Kuperman et al. 2012, BLP), retrieves bigram frequencies for the mid-specificity phrases from Google Books Ngrams, and reduces the item set from 120 to 100 triads. Run this first.

**`specificity_norming_final.ipynb`**
Analyses the behavioural norming data collected via Gorilla, in which participants completed a forced-choice specificity rating task (n=52). Computes sensitivity (d') and RT per subject per condition, fits linear mixed effects models, and includes a robustness check excluding five WordNet-flagged triads.

**`computational_validation.ipynb`**
Validates the specificity gradient using two computational approaches: Sentence-BERT cosine similarity (does the mid-specificity phrase move the representation closer to the high-specificity target?) and WordNet taxonomic depth (are high-specificity nouns deeper in the hypernym hierarchy?).

## Running order

Run the notebooks in the order listed above. `specificity_norming_final.ipynb` and `computational_validation.ipynb` both depend on `stimuli_specificity_complete.xlsx`, which is the output of `specificity_stim_properties.ipynb`.