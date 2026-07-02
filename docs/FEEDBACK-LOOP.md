# The feedback flywheel

Every round played in the app becomes a labeled training example. This is not
reinforcement learning and we do not call it that: there is no policy, no reward
signal over a trajectory, no exploration. It is the older, boring, effective
thing: a supervised data flywheel with a human in the loop.

## The loop

```
app round revealed / upload corrected
        |            (image + true country + model guess)
        v
data/feedback/*.jpg + feedback.jsonl          gear 1: the app appends (HopsFS)
        |
        v
pipelines/feedback_pipeline.py                gear 2: scheduled job, embeds with
        |                                     the SAME shared module -> geo_feedback FG
        v
pipelines/train.py --with-feedback            gear 3: retrain heads on
        |                                     OSV5M rows + feedback rows
        v
geo_country vN+1 -> leaderboard FG            gear 4: rescoring -- same held-out
        |                                     cells, every version comparable
        v
serving/deploy.py                             gear 5: redeploy the champion
```

## Why each gear is shaped this way

- **The app logs images, not embeddings.** The endpoint could return vectors, but
  then a served-model change would silently mix embedding spaces in the feedback
  store. Re-embedding at ingestion with the shared module keeps one code path and
  makes the feedback reusable if the backbone ever changes.
- **The label is the human's, the guess is logged next to it.** `correct` per row
  gives a live accuracy stream: the leaderboard of deployed model versions against
  real usage, not just the frozen test split.
- **Rescoring is against the same held-out cells.** Feedback rows go into
  training only. The eval set never moves, so v1 vs v2 is a fair fight.
- **Playset rounds are auto-labeled** (ground truth known), upload corrections are
  volunteered. Both carry selection bias -- people replay what fools the model --
  which is exactly what makes them worth more than random extra OSV5M rows.

## Honest limits

- A few hundred feedback rows against 330k training rows will not move top-1.
  The flywheel pays off at thousands of rounds, or when concentrated on
  systematically-missed countries (the bias above helps here).
- Upload corrections are unverified human input. A poisoned label is one row of
  noise in an MLP fit -- acceptable at this scale; revisit if the app goes viral.
