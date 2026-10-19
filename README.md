# Conceptual specificity across single words and phrases: neural correlates and computational bases

> **👋** Note: This repository is currently under active development. While it contains the core analysis scripts for the manuscript, we are currently tidying the documentation and validating the environment.

The study uses concurrent **MEG/EEG** and **structural MRI** to investigate whether neural responses in the Anterior Temporal Lobe (ATL) scale continuously with conceptual specificity regardless of whether meaning is accessed lexically (single words) or compositionally (adjective–noun phrases), using a parametric three-level specificity design (e.g., *bird* → *nocturnal bird* → *owl*).

## Pipeline overview

* **Stimulus construction and validation:** Parametric specificity triads matched for psycholinguistic a number of variables.
* **Behavioural norming:** Forced-choice specificity rating task (n=52, via Prolific/Gorilla) validating the intended specificity gradient across triads, analysed using mixed-effects models and signal detection (d').
* **Computational validation:** WordNet hypernym depth analysis and Sentence-BERT cosine similarity metrics to independently verify the specificity structure of the stimulus set.
* **Univariate ERP analysis:** N400 and late component analyses across centroparietal electrodes (350–500 ms and 600–800 ms post-noun onset).
* **Univariate ROI analysis:** Mass-univariate ROI analysis using cluster-based permutation tests (Eelbrain), with FDR correction across eight bilateral ROIs (ATL, PTL, IFC, TPC).
* **Multivariate decoding analysis:** Time-resolved decoding and temporal generalisation (Scikit-learn), plus hybrid Python/R mixed-effects modelling.

`mne.yml` contains all the main Python packages I used for this study. To create a conda environment from this file:
`conda env create -f mne.yml`
`eelbrain.yml` contains the packages for using eelbrain for running the ROI analyses.

## Data availability

The raw MEG, EEG, and MRI data are BIDS formatted.