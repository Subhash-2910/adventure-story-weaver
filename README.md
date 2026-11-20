# Adventure Story Weaver

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/vqsbzz3tav/badge.svg)](https://ask.thili.ai/b/vqsbzz3tav)

This project guides you through building a custom GPT from scratch, specifically tailored for generating text adventure story segments. By training on existing narrative data, your GPT will learn to create dynamic and coherent continuations, providing a powerful tool for game developers to rapidly prototype story ideas.

## What you are building

**Goal.** You will build a text adventure story segment generator using a custom GPT, trained on adventure game narratives, to create dynamic story continuations.

**Done when.** The project is done when your custom GPT can generate coherent story continuations based on a given prompt, accessible through a command-line interface, demonstrating the full pipeline from tokenizer to trained model.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Assemble a small corpus of text from public domain adventure stories or fantasy novels.**  
      Hint: Gather and clean your raw text data.
- [ ] **2. Implement the byte-pair tokenizer and generate embeddings from the story text.**  
      Hint: Apply Byte-Pair Encoding and create token embeddings.
- [ ] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      Hint: Construct the Causal Self-Attention layer and the full Transformer block.
- [ ] **4. Set up the training loop and train the GPT model on the narrative data.**  
      Hint: Implement the training loop and optimize your model.
- [ ] **5. Implement a text generation feature that creates coherent story continuations.**  
      Hint: Develop the text generation logic using sampling.
- [ ] **6. Design a CLI where users can input a story prompt and receive generated segments.**  
      Hint: Create a user-friendly command-line interface.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow users to specify parameters for generation, such as desired length or 'creativity' (temperature).
- Implement a feature to save and load generated story segments, enabling longer narrative development.
- Experiment with different corpus sources (e.g., specific genres, authors) to observe changes in narrative style.
