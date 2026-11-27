# Dataset: *Poems by Emily Dickinson, Three Series, Complete*

**What it is:** the three volumes of Emily Dickinson's poems published after her death. The first two were edited by Mabel Loomis Todd and T. W. Higginson (1890, 1891) and the third by Mabel Loomis Todd (1896). After parsing, the corpus has **446 short lyric poems** (about 28,000 words, about 43,000 GPT-2 tokens) in four thematic sections: Life, Love, Nature, and Time and Eternity.

**Source:** Project Gutenberg eBook #12242 (released May 1, 2004). https://www.gutenberg.org/ebooks/12242
**File used:** `12242-0.txt` (UTF-8 plain text, 176,641 bytes), downloaded on 2026-11-26 from the official Project Gutenberg mirror at https://gutenberg.pglaf.org/1/2/2/4/12242/12242-0.txt (the main gutenberg.org site was down for maintenance that day).
SHA-256: `c3af83011e494155b576cf0ecbb675d2db15b68efa273ba3769c8bd0aaf191e7`

**Licence:** the poems and these editions are in the public domain in the United States. The mirror copy holds only the text between Project Gutenberg's START/END markers, with no licence block. The Project Gutenberg licence (https://www.gutenberg.org/policy/license.html) applies to redistribution under the Project Gutenberg name. Use outside the U.S. depends on local copyright law.

**Human-created, not synthetic:** every poem was written by Emily Dickinson (1830–1886). Every generation prompt is the real first line of a held-out poem from this file, so no prompt was written by an AI tool.

**Caveat (affects the model):** these are the *edited* 1890s texts. The editors added titles, regularized punctuation (Dickinson's many dashes became mostly commas and " --") and sometimes changed words. The model therefore learns the editors' version of Dickinson, not her manuscripts. The transcriber's note near the top of the file shows one poem in both versions.

## Files in this folder

| File | Purpose |
|---|---|
| `12242-0.txt` | Raw Gutenberg text, unchanged |
| `poems.csv` | One row per parsed poem (Series, section, number, editor's title, text, size statistics, and the train/val/test split it belongs to). Written by the notebook. |
| `README.md` | This file |

## How to reproduce

The raw file is committed, so the notebook needs no download for the data. If the file is missing, download it again from either URL above and save it as `data/12242-0.txt`. The SHA-256 above confirms it is the same file.

The GPT-2 weights (about 550 MB) are downloaded automatically from the Hugging Face Hub (`gpt2`) the first time the notebook runs.
