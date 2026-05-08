# Speech Draft: AI Virtual Cell Models for Reverse Perturbation Retrieval

Hello everyone, my name is **Anrui Wang**. Today I will present our course project, supervised by **Prof. Jie Zheng**. The project is titled **AI Virtual Cell Models for Reverse Perturbation Retrieval from Transcriptomic Cell States**. It was developed in **AI for Science and Engineering**, where we explored how AI models can be adapted to scientific questions.


## Slide 1

Can AI read a cell state and infer its hidden cause?

**Single-cell biology** technologies now allow us to measure cells at a detailed molecular level. Instead of treating a tissue as one average sample, we can observe many cell types and abnormal cell states.

In this project, we treat **gene expression** as a **molecular fingerprint** of a cell state. The question is not only what the fingerprint looks like, but what may have caused it.


## Slide 2

A human cell contains thousands of genes, but only a small number may be useful targets in a specific disease context.

Testing every possible gene in the lab is expensive and slow. So AI can be useful if it helps **narrow the search space**.

Here, the input is a **transcriptomic fingerprint**. The output is a **ranked shortlist** of candidate genes. This is not a final biological answer. It is a set of **hypotheses** that can guide later experiments.


## Slide 3

In a standard biology experiment, the direction is forward. Researchers start with cells, apply a **gene perturbation**, and measure how the cell state changes.

Our project reverses that direction. We start from the observed abnormal cell state and ask which perturbation gene could have produced a similar change.

This is **reverse perturbation retrieval**. The model does not prove the true cause. It ranks possible causes for **follow-up validation**.


## Slide 4

We did not train a model from scratch. We adapted **scGPT**, a **single-cell foundation model**, as an encoder for transcriptomic cell states.

The perturbed cell state is encoded into a compact representation. **Matched control cells** are also encoded as a reference baseline.

By comparing the perturbed state with this baseline, the model focuses on the change that may be caused by the perturbation, rather than unrelated cell variation.

Then a **gene scoring head** converts this perturbation signal into a ranked list of candidate genes.

This design helped us see that the model is only one part of the system. The **framing** and **evaluation setting** also affect whether the output is useful.


## Slide 5

Our main takeaway is the process of turning an **interdisciplinary question** into a working AI research task.

First, we learned to translate a **biomedical need** into an AI task. This required us to ask what kind of output would actually help a biologist.

Second, we learned to adapt a **foundation model** for a real dataset, connecting single-cell information to candidate ranking.

Third, we learned to **redesign test cases** to better reflect real biological use. Public datasets are valuable, but their default settings do not always match the practical question.

More broadly, the course project trained us to think across disciplines: from **biological motivation** to **AI model** and **evaluation design**.
