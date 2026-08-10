# Reflection — Support Ticket Classifier

## What would I improve with more data or time?

1. **Pre-trained embeddings over TF-IDF.** With a larger corpus, sentence-transformer embeddings (e.g. `all-MiniLM-L6-v2`) would capture semantic meaning far better than bag-of-words — "can't log in" and "authentication broken" would map to nearby vectors even though they share zero tokens.

2. **Active learning loop.** In production, tickets flagged as "Needs Human Review" should feed back into the training set once a human labels them. This creates a virtuous cycle: the model improves on exactly the edge cases it struggles with, and the human-review queue shrinks over time.

3. **Multi-label support.** Real tickets often span categories — a billing complaint that also describes a technical outage. Moving to multi-label classification (e.g. binary relevance with calibrated classifiers) would let the system assign *all* relevant tags instead of forcing a single choice.

4. **Richer priority detection.** The current keyword-rule approach is brittle. A small fine-tuned model (or even a regex-scored urgency heuristic combining keyword density, punctuation patterns like "!!!", and ALL-CAPS frequency) would be more robust.

5. **A/B testing and monitoring.** Before full deployment, I'd shadow-run the model alongside human triage for a few weeks, tracking agreement rate and mean-time-to-resolution to validate that automated routing actually improves throughput.
