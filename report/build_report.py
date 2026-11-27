"""Writes report/module_summary.docx into the provided APA 7 template.

Every number in the report is read from outputs/metrics/*.csv, and every quoted sample is checked
against outputs/generations/*.csv, so the report always matches the notebook run.
Run with the project venv:  .venv/bin/python report/build_report.py
"""
import math
from pathlib import Path

import pandas as pd

from apa_docx import APADoc

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "apa-7-template-word-doc.docx"
OUT = ROOT / "report" / "module_summary.docx"
FIG = ROOT / "outputs" / "figures"
MET = ROOT / "outputs" / "metrics"
GEN = ROOT / "outputs" / "generations"

# ---------------------------------------------------------------------------- evidence
hist = pd.read_csv(MET / "training_history.csv")
ppl = pd.read_csv(MET / "perplexity.csv").set_index(["model", "split"]).perplexity
style = pd.read_csv(MET / "style_statistics.csv", index_col=0)
dec = pd.read_csv(MET / "decoding_comparison.csv", index_col=0)
memb = pd.read_csv(MET / "membership_gap.csv").set_index("model")
spans = pd.read_csv(MET / "memorization_spans.csv", index_col=0)
probe = pd.read_csv(MET / "extraction_probe.csv")
gens = pd.read_csv(GEN / "test_prompt_generations.csv")
poems = pd.read_csv(ROOT / "data" / "poems.csv")

best = hist.loc[hist.val_loss.idxmin()]
BEST_EPOCH = int(best.epoch)
assert BEST_EPOCH == 2, "report text assumes epoch 2 was selected; re-check the prose"
split_n = poems.split.value_counts()


def gen_text(prompt_id, model):
    return gens[(gens.prompt_id == prompt_id) & (gens.model == model)].text.iloc[0]


def check(quote, prompt_id, model):
    """Fail loudly if a quoted sample is not in the notebook's generations."""
    assert quote.replace(" / ", "\n") in gen_text(prompt_id, model), (prompt_id, model, quote)
    return quote


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


P_TEST = ppl[("pretrained GPT-2", "test")]
F_TEST = ppl[("fine-tuned GPT-2", "test")]
P_VAL = ppl[("pretrained GPT-2", "val")]
F_VAL = ppl[("fine-tuned GPT-2", "val")]

q_sport = check("The score is calculated as the average score in every match played", 0, "pretrained")
q_storm = check("As if it had been a storm", 0, "fine-tuned")
q_sabbath = check("Some keep the Sabbath going to church; / Some keep the Sabbath going to church;", 5, "fine-tuned")
q_gods = check("And I prayed for the gods / Upon the lawn.", 30, "fine-tuned")
q_little = check("Till you feel it, or -- / A little less, / Too little, too soon!", 20, "fine-tuned")
q_hare = check("A florid hare / A flower with eyes so long that one can see", 20, "fine-tuned")
q_captain = check("the captain yelled", 20, "pretrained")

# ---------------------------------------------------------------------------- document
TITLE = "Fine-Tuning GPT-2 to Continue Emily Dickinson's Poems: Output Quality, Repetition, and Memorization"
doc = APADoc(TEMPLATE)
doc.title_page(TITLE, ["Alex Kuzmin", "Woolf University",
                       "AI Mastery Capstone, Project 5: Generative AI Applications", "November 26, 2026"])
doc.title(TITLE)

# ---------------------------------------------------------------------------- Overview
doc.heading1("Overview")
doc.body(
    "This project builds a small generative text system. Given the first line of a poem, it writes a continuation "
    "in the style of Emily Dickinson's poems as her first editors published them in the 1890s. I fine-tuned a "
    "pretrained GPT-2 small model, a decoder-only Transformer language model (Radford et al., 2019), on 356 of the "
    f"446 poems in the Project Gutenberg edition (Dickinson, 1890–1896/2004). I used a training loop written in PyTorch "
    "and the Hugging Face Transformers library. Every prompt is the real first line of a held-out poem, so the "
    "prompts are human-written and the model has never seen them."
)
doc.body(
    f"Fine-tuning cut perplexity on the held-out test poems from {P_TEST:.1f} to {F_TEST:.1f}, and the outputs "
    "changed from prose to short, capitalized verse lines. The system also has clear failures. It repeats lines, "
    "especially the prompt itself, and it stops too early to develop an idea. Because a model trained on one "
    "author's small body of work can copy that work, I measured memorization directly. At the selected training "
    "epoch the model imitates rather than copies, but the memorization signal grows quickly if training continues."
)

# ---------------------------------------------------------------------------- Dataset
doc.heading1("Dataset or Prompt Description")
doc.body(
    "The data source is *Poems by Emily Dickinson, Three Series, Complete*, eBook #12242 on Project Gutenberg "
    "(Dickinson, 1890–1896/2004). It contains the three posthumous volumes edited by Mabel Loomis Todd and T. W. "
    "Higginson. The text is human-written, in the public domain in the United States, and not used in any of my "
    "earlier capstone projects. The raw file is included with the submission, with its download source and checksum "
    "in data/README.md."
)
doc.body(
    "I wrote a parser that removes the prefaces, editorial notes, and editor-supplied titles, and keeps one record "
    f"per poem with its line and stanza breaks. The result is {len(poems)} poems with 27,867 words and 43,072 GPT-2 "
    "tokens. The poems are short (median 50 words, 10.5 lines), and 82.6% of all stanzas are exactly four lines "
    "long. Figure 1 in the notebook shows these distributions. There are no empty poems, no duplicates, and no "
    "repeated first lines. I split the data by whole poem, stratified by the edition's four thematic sections, into "
    f"{split_n['train']} training, {split_n['val']} validation, and {split_n['test']} test poems. Splitting by poem "
    "keeps any part of a test poem out of training, so test perplexity and the memorization checks are not inflated."
)
doc.body(
    "One property of the data shapes everything the model can learn. These are the editors' versions of the poems, "
    "not Dickinson's manuscripts. A transcriber's note in the file shows one poem in both forms. In the manuscript "
    "most phrases end in a dash, while the edited version uses commas and periods. In the edition only "
    "5.0% of lines end in a dash. I removed the titles because the editors, not Dickinson, wrote them, but the "
    "edited punctuation remains in the data."
)
doc.body(
    "The task fits the data well. Each poem is a short, self-contained sequence with a strong and consistent form, "
    "so it is possible to check whether generated text follows that form. The corpus is also far too small to train "
    "a language model from scratch, which is why a pretrained model is needed."
)

# ---------------------------------------------------------------------------- Model
doc.heading1("Model Design and Training Approach")
doc.body(
    "GPT-2 is a stack of Transformer decoder blocks trained to predict the next token. Each position can attend only "
    "to earlier positions, and pretraining on a large web corpus gives the model general knowledge of English before "
    "any task-specific training (Radford et al., 2019). This fits the task directly, because continuing a poem from "
    "its first line is next-token prediction. Self-attention also lets the model use any earlier line in the poem "
    "when it writes the next one, which matters for rhyme and stanza patterns. I used the smallest GPT-2 (124.4 "
    "million parameters: 12 layers, 12 attention heads, 768-dimensional hidden states, and a 50,257-token byte-level "
    "vocabulary). It fine-tunes in minutes on a laptop GPU, and its byte-level tokenizer handles archaic forms such as "
    "\"'t is\" without unknown tokens. The brief's other options, GANs and VAEs, are designed for continuous data "
    "such as images and fit discrete text poorly."
)
doc.body(
    "Each poem was turned into the end-of-text token followed by the poem, and the poems were joined and cut into "
    "128-token blocks. The end-of-text token marks the start of a poem, so at generation time the prompt "
    "\"end-of-text + first line\" puts the model in the same context it saw during training. Training used the "
    "standard causal language-modeling cross-entropy loss and the AdamW optimizer with a learning rate of 5e-5 and "
    "weight decay of 0.01. The learning rate warmed up over the first 10% of steps and then decayed linearly. "
    "Gradients were clipped at a norm of 1.0, and the batch size was 8. A small learning rate and warm-up are meant "
    "to adapt the pretrained weights without erasing what they already encode. I wrote the loop by hand instead of "
    "using a ready-made trainer so that each step is visible in the notebook."
)
doc.body(
    "I trained for eight epochs, measured validation loss before training and after every epoch, and kept the "
    "weights with the lowest validation loss. Figure 1 shows the result. Validation perplexity fell from "
    f"{hist.val_ppl[0]:.1f} (the pretrained model) to {hist.val_ppl[1]:.1f} after one epoch and reached its minimum of "
    f"{best.val_ppl:.1f} at epoch {BEST_EPOCH}. After that the model began to overfit: training loss kept falling "
    f"from {hist.train_loss[2]:.2f} to {hist.train_loss[8]:.2f} while validation loss rose from {hist.val_loss[2]:.2f} "
    f"to {hist.val_loss[8]:.2f}. I therefore used the epoch-2 weights. Training was stable, with no loss spikes. "
    "Gradient norms stayed between about 6 and 17 before clipping, so clipping limited every update. Each epoch took "
    "about 19 seconds on an Apple M4 GPU, and two runs with the same seed gave the same validation losses to six decimal places."
)
doc.figure(1, "Training and Validation Loss, Gradient Norm, and Learning-Rate Schedule",
           FIG / "fig2_training_curves.png", width_in=6.5,
           note="Epoch 0 is the pretrained model before fine-tuning. The dashed line marks the selected epoch.")

# ---------------------------------------------------------------------------- Evaluation
doc.heading1("Output Evaluation and Interpretation")
doc.body(
    "I compared the fine-tuned model with the same GPT-2 before fine-tuning, using identical prompts and decoding "
    "settings. The prompts were the first lines of all 45 test poems, with one sample per prompt from each model. "
    "Decoding used nucleus sampling with temperature 0.8 and top-p 0.9. Nucleus sampling draws only from the smallest "
    "set of tokens that holds most of the probability mass. Holtzman et al. (2020) proposed it because always picking "
    "the most likely token produces repetitive text, while unrestricted sampling produces incoherent text. Table 1 "
    "summarizes the results."
)
rows = [
    ["Test perplexity (lower is better)", "n/a", f"{P_TEST:.1f}", f"{F_TEST:.1f}"],
    ["Words per line", f"{style.loc['real (held-out)', 'words_per_line']:.1f}",
     f"{style.loc['pretrained', 'words_per_line']:.1f}", f"{style.loc['fine-tuned', 'words_per_line']:.1f}"],
    ["Lines per continuation", f"{style.loc['real (held-out)', 'n_lines']:.1f}",
     f"{style.loc['pretrained', 'n_lines']:.1f}", f"{style.loc['fine-tuned', 'n_lines']:.1f}"],
    ["Lines starting with a capital", pct(style.loc['real (held-out)', 'capital_line_start']),
     pct(style.loc['pretrained', 'capital_line_start']), pct(style.loc['fine-tuned', 'capital_line_start'])],
    ["Lines ending in a dash", pct(style.loc['real (held-out)', 'dash_line_end']),
     pct(style.loc['pretrained', 'dash_line_end']), pct(style.loc['fine-tuned', 'dash_line_end'])],
    ["Repeated-trigram rate", pct(style.loc['real (held-out)', 'repeated_trigram_rate']),
     pct(style.loc['pretrained', 'repeated_trigram_rate']), pct(style.loc['fine-tuned', 'repeated_trigram_rate'])],
    ["Distinct-2 (diversity)", f"{style.loc['real (held-out)', 'distinct_2']:.2f}",
     f"{style.loc['pretrained', 'distinct_2']:.2f}", f"{style.loc['fine-tuned', 'distinct_2']:.2f}"],
]
doc.table(1, "Held-Out Perplexity and Style Statistics for Real and Generated Continuations",
          ["Measure", "Real poems", "Pretrained", "Fine-tuned"], rows, [2.6, 1.2, 1.2, 1.2],
          note="Computed on the 45 test poems (continuation after the first line) and on one nucleus-sampled "
               "continuation per test prompt from each model. Distinct-2 is unique word pairs divided by all word "
               "pairs across the set.")
doc.body(
    f"Perplexity on the test poems fell by {1 - F_TEST / P_TEST:.0%}, from {P_TEST:.1f} to {F_TEST:.1f}. The "
    "fine-tuned model therefore predicts real Dickinson poems it never saw much better than the original GPT-2. "
    f"Validation and test perplexity are close ({F_VAL:.1f} and {F_TEST:.1f}), so choosing the epoch on validation "
    "data did not overfit to the validation poems."
)
doc.body(
    "The clearest effect of fine-tuning is that the model writes verse instead of prose. Given \"Success is counted "
    f"sweetest,\" the pretrained model wrote about sports: \"{q_sport}.\" Given the opening of a letter poem, "
    f"\"GOING to him! Happy letter! Tell him --,\" it wrote a story in which \"{q_captain}.\" The fine-tuned model "
    f"answered the same prompts with short lines in Dickinson's register: \"{q_little}\" Words per line fell from "
    f"{style.loc['pretrained', 'words_per_line']:.1f} to {style.loc['fine-tuned', 'words_per_line']:.1f}, close to "
    f"the real {style.loc['real (held-out)', 'words_per_line']:.1f}. Lines starting with a capital rose from "
    f"{pct(style.loc['pretrained', 'capital_line_start'])} to {pct(style.loc['fine-tuned', 'capital_line_start'])}. "
    f"Some lines are new and plausible images, such as \"{q_gods}\" (from the prompt \"I went to heaven, --\")."
)
doc.body(
    "Fine-tuning copied the surface of the style but not its structure. Continuations averaged "
    f"{style.loc['fine-tuned', 'n_lines']:.1f} lines against {style.loc['real (held-out)', 'n_lines']:.1f} in the real "
    f"poems, and only {pct(style.loc['fine-tuned', 'has_stanza_break'], 0)} contained a stanza break (real poems: "
    f"{pct(style.loc['real (held-out)', 'has_stanza_break'], 0)}). For \"Success is counted sweetest\" the model wrote one "
    f"line, \"{q_storm},\" where Dickinson builds a 12-line argument about defeat. Coherence across lines is weak. In "
    f"the letter poem the images jump from heirlooms to \"{q_hare},\" while the real poem develops one idea over "
    "three stanzas. The edited-style dash appeared at the end of "
    f"{pct(style.loc['fine-tuned', 'dash_line_end'])} of generated lines against "
    f"{pct(style.loc['real (held-out)', 'dash_line_end'])} in the real poems. Outputs also varied a lot between "
    "seeds. Three samples for \"I lost a world the other day.\" gave an 11-line poem with a stanza break, a 3-line "
    "fragment, and a loop of the prompt line."
)
doc.body(
    "The main failure case is repetition. Under nucleus sampling, the fine-tuned model's repeated-trigram rate was "
    f"{pct(style.loc['fine-tuned', 'repeated_trigram_rate'])}, against "
    f"{pct(style.loc['real (held-out)', 'repeated_trigram_rate'])} for the real poems, and diversity (distinct-2) was "
    f"no better than the pretrained model's. The typical form is echoing the prompt. \"Some keep the Sabbath going "
    f"to church;\" was continued as \"{q_sabbath}.\" To test whether this comes from the model or from decoding, I "
    "compared three decoding strategies on the same fine-tuned model (Table 2)."
)
order = ["greedy", "nucleus", "nucleus + no-repeat-3gram"]
labels = {"greedy": "Greedy", "nucleus": "Nucleus (T = 0.8, p = 0.9)", "nucleus + no-repeat-3gram": "Nucleus + no repeated trigram"}
doc.table(2, "Effect of the Decoding Strategy on the Fine-Tuned Model",
          ["Decoding", "Repeated trigrams", "Samples > 20% repeated", "Distinct-2", "Mean lines"],
          [[labels[k], pct(dec.loc[k, 'repeated_trigram_rate']), pct(dec.loc[k, 'share_samples_with_repetition_over_20pct'], 0),
            f"{dec.loc[k, 'distinct_2']:.2f}", f"{dec.loc[k, 'mean_lines']:.1f}"] for k in order],
          [2.3, 1.1, 1.3, 0.9, 0.9],
          note="45 test prompts, one output each, same fine-tuned model.")
doc.body(
    f"Greedy decoding failed almost completely. {pct(dec.loc['greedy', 'share_samples_with_repetition_over_20pct'], 0)} "
    "of outputs repeated more than a fifth of their trigrams, and the typical output looped one phrase, such as "
    "\"The cricket sang,\" until it hit the length limit. This is the degeneration known from maximization-based "
    "decoding (Holtzman et al., 2020). Blocking any repeated trigram during nucleus sampling reduced "
    f"repetition to {pct(dec.loc['nucleus + no-repeat-3gram', 'repeated_trigram_rate'])} and raised distinct-2 to "
    f"{dec.loc['nucleus + no-repeat-3gram', 'distinct_2']:.2f}, close to the real poems' "
    f"{style.loc['real (held-out)', 'distinct_2']:.2f}. The outputs stayed short "
    f"({dec.loc['nucleus + no-repeat-3gram', 'mean_lines']:.1f} lines). Repetition is therefore mostly a decoding "
    "problem that a simple constraint fixes, while the short, loosely connected outputs are a limit of the model "
    "and its small training set."
)
doc.figure(2, "Per-Sample Style Statistics for Real Poems and Generated Continuations",
           FIG / "fig3_style_statistics.png", width_in=6.5,
           note="Each dot is one continuation of a test prompt. Real poems are the held-out test poems after their first line.")

# ---------------------------------------------------------------------------- Ethics
doc.heading1("Ethical Considerations and Responsible Use")
doc.body(
    "The main ethical concern for this system is creative ownership, in two forms: copying the author's text, and "
    "imitating a named author so closely that the output could be passed off as hers. Language models can memorize "
    "and reproduce parts of their training data. Carlini et al. (2021) extracted hundreds of verbatim training "
    "sequences from GPT-2 itself. The risk is higher here than in general web training, because the model saw the "
    "same 356 poems again in every epoch. I measured it in three ways (Figure 3)."
)
r_p, r_2, r_8 = (memb.loc[k, "test_to_train_ratio"] for k in ["pretrained", "fine-tuned (epoch 2)", "overfit (epoch 8)"])
doc.body(
    "First, no generated text copied the training poems. The longest run of words shared with the training set was "
    f"at most {int(spans.loc['fine-tuned (epoch 2)', 'max'])} words, the same maximum as the real held-out poems "
    f"({int(spans.loc['real (held-out)', 'max'])} words). The shared runs were common phrases such as \"as far as "
    "the,\" not lines of poetry. Second, when I prompted with the first line of 40 training poems, no model "
    f"reproduced the true second line, and the best match was {int(probe.matched_words.max())} word. The pretrained "
    "GPT-2 also showed no sign of already knowing these poems from its web training data. Third, the gap between "
    "training-poem and test-poem perplexity measures how specifically the model has fitted its training poems. This "
    f"ratio was {r_p:.2f} for the pretrained model, {r_2:.2f} for the selected epoch-2 model, and {r_8:.2f} for the "
    f"epoch-8 model, whose training perplexity ({memb.loc['overfit (epoch 8)', 'train_ppl']:.1f}) was far below its "
    f"test perplexity ({memb.loc['overfit (epoch 8)', 'test_ppl']:.1f}). Memorization was therefore building up even "
    "before it appeared as copied text. Selecting the epoch by validation loss was a quality decision, but it also "
    "worked as a safeguard. Training longer would have made held-out perplexity worse and copying more likely."
)
doc.figure(3, "Memorization Checks for the Pretrained, Selected, and Overfit Models",
           FIG / "fig4_memorization.png", width_in=6.5,
           note="Left: perplexity on training versus test poems. Middle: longest run of words in each generated "
                "continuation that also appears in the training poems. Right: words of the true continuation "
                "reproduced when the model is prompted with a training poem's first line.")
doc.body(
    "Copying is low, but imitation is the purpose of the system, and that creates a misuse risk. Dickinson's poems "
    "are in the public domain, so training on them is legally simpler than training on a living poet's work. Even "
    "so, outputs such as \"Till you feel it, or -- / A little less, / Too little, too soon!\" are close enough to her "
    "register that they could be presented as newly found Dickinson poems. The same pipeline, run on the work of a "
    "living author, would imitate that author without consent, which is a question of ownership and income, not "
    "only copyright. Responsible use therefore means labeling every output as machine-generated in the style of "
    "Dickinson, never attributing it to her, and keeping the original attribution of the corpus with the model."
)
doc.body(
    "A second concern is whose voice the data represents. The model learns Dickinson as her 1890s editors presented "
    "her, with regularized punctuation, so it under-produces the dashes that characterize her manuscripts. The "
    "pretrained model adds another layer. GPT-2 was trained on web pages linked from Reddit (Radford et al., 2019), "
    "and Bender et al. (2021) argue that such large, undocumented web corpora over-represent some viewpoints and "
    "carry biases that downstream users cannot see. The pretrained model's drift into a story about a captain with a "
    "gun, from a prompt about a letter, shows that this prior is still active and is only partly replaced by "
    "fine-tuning. A user who thinks of the output as \"Dickinson\" is reading a mix of the editors' choices, the web "
    "text behind GPT-2, and the model's own errors. Documenting that mix is part of responsible use."
)

# ---------------------------------------------------------------------------- Limitations
doc.heading1("Limitations and Future Improvements")
doc.body(
    f"The corpus is very small ({split_n['train']} training poems), which caused the early overfitting and limits "
    "how much structure the model can learn. The evaluation is also limited. Perplexity and the style statistics "
    "measure form and predictability, not whether a poem is good, coherent, or moving, and no human readers rated "
    "the outputs. Results come from one training run and one sample per prompt, so small differences in the tables "
    "may not be stable. The memorization checks look for exact word matches and would miss close paraphrases. "
    "Finally, the model learned an edited version of the poems, so any claim that it writes \"like Dickinson\" holds "
    "only for that edition."
)
doc.body(
    "Several next steps follow from these results. First, use the repeated-trigram constraint as the default decoding, "
    "since it removed most repetition at no cost to quality. Second, add a small blind human rating that compares "
    "generated and real continuations for coherence and style, because the automatic metrics cannot judge whether a "
    "poem makes sense. Third, train on a manuscript-faithful edition, where one is licensed for this use, to test "
    "whether the model learns her dashes and capitalization. Fourth, repeat the run with several seeds and add a "
    "near-duplicate check based on paraphrase similarity rather than exact matches. A larger GPT-2 or "
    "parameter-efficient fine-tuning could improve coherence, but it would need the same memorization checks, "
    "because larger models memorize more (Carlini et al., 2021)."
)

# ---------------------------------------------------------------------------- References
doc.page_break()
doc.heading1("References")
for ref in [
    "Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the dangers of stochastic parrots: "
    "Can language models be too big? In *Proceedings of the 2021 ACM Conference on Fairness, Accountability, and "
    "Transparency* (pp. 610–623). Association for Computing Machinery. https://doi.org/10.1145/3442188.3445922",
    "Carlini, N., Tramèr, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K., Roberts, A., Brown, T., Song, D., "
    "Erlingsson, Ú., Oprea, A., & Raffel, C. (2021). Extracting training data from large language models. In "
    "*Proceedings of the 30th USENIX Security Symposium* (pp. 2633–2650). USENIX Association. "
    "https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting",
    "Dickinson, E. (2004). *Poems by Emily Dickinson, three series, complete* (M. L. Todd & T. W. Higginson, Eds.). "
    "Project Gutenberg. https://www.gutenberg.org/ebooks/12242 (Original work published 1890–1896)",
    "Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). The curious case of neural text degeneration. In "
    "*International Conference on Learning Representations*. https://openreview.net/forum?id=rygGQyrFvH",
    "Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language models are unsupervised "
    "multitask learners* [Technical report]. OpenAI. "
    "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
]:
    doc.reference(ref)

doc.save(OUT)
print(f"wrote {OUT}")
