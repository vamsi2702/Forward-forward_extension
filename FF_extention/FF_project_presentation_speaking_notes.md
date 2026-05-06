# Speaking Notes: Forward-Forward Algorithm Extension Study

## Slide 1: Title
- Opening line:
  - Good morning everyone. This presentation covers our extension study of Hinton's Forward-Forward algorithm on MNIST.
- Set expectation:
  - I will first recap the original FF idea, then explain our three objectives, results, and key limitations.

## Slide 2: Project Goals
- Say clearly:
  - We had three focused objectives.
  - Objective 1: vary goodness functions with ReLU fixed.
  - Objective 2: vary activation functions with goodness fixed.
  - Objective 3: test local spatial FF and analyze goodness margins.
- Transition:
  - Before showing results, let me quickly summarize the original paper concepts.

## Slide 3: Original Paper Core Concepts
- Key message:
  - FF is different from backprop because learning is local per layer.
- Explain positive/negative data in plain words:
  - Positive means correct label is overlaid on the input.
  - Negative means wrong label is overlaid.
- End with inference:
  - At test time, we try all labels and pick the label that gives maximum total goodness.

## Slide 4: Original Paper Key Equations
- Walk through equations slowly:
  - Goodness in the original paper is typically the sum of squared activations.
  - The loss pushes positive goodness above threshold and negative goodness below threshold.
- Intuition sentence:
  - So FF is essentially a local discrimination game at each layer.
- Mention prediction equation:
  - Prediction uses argmax over summed goodness across layers.

## Slide 5: What We Implemented
- Mention implementation fidelity:
  - We used MNIST and label overlay exactly in the FF spirit.
- Mention architectures:
  - Objectives 1 and 2 use fully connected FF.
  - Objective 3 uses locally connected spatial layers without weight sharing.
- Mention reproducibility:
  - We set deterministic seeds for stable comparisons.

## Slide 6: Objective 1 Goodness Functions
- Explain why this matters:
  - In FF, goodness definition is central because it directly shapes local learning signal.
- Briefly explain the four highlighted functions:
  - sum_sq: baseline L2-style energy.
  - sum_huber_like: robust between linear and quadratic behavior.
  - pnorm_1_5: softer growth than L2.
  - pnorm_3: steeper growth, emphasizes strong activations.

## Slide 7: Objective 1 Full Results Part 1
- Point to top row:
  - pnorm_3 with ReLU is best at 95.54 percent accuracy.
- Baseline reminder:
  - Baseline sum_sq with ReLU is 92.63 percent.
- Practical takeaway:
  - Goodness choice alone can shift accuracy by several points.

## Slide 8: Objective 1 Full Results Part 2
- Mention lower half behavior:
  - Some functions collapse badly, for example sum_softplus, showing FF is sensitive to goodness shape.
- Key comparison line:
  - pnorm_3 beats sum_sq baseline by 2.91 accuracy points.
- Transition:
  - Next, we keep goodness fixed and test activation choice.

## Slide 9: Why pnorm_3 Exceeds Baseline
- Intuition:
  - p=3 amplifies high-confidence activations more strongly than squared norm.
- Margin connection:
  - This can widen positive-negative goodness separation, which is exactly what FF needs.
- Pairing point:
  - With ReLU, this emphasis becomes even clearer due to non-negative activations.

## Slide 10: Objective 2 Method and ReLU Explanation
- What changed:
  - We fixed goodness at sum_sq and varied only activation over 17 candidates.
- Why ReLU likely won:
  - Non-saturating behavior supports stable optimization.
  - Sparse responses can help separate positive vs negative examples.
  - Simpler activation can be easier to optimize under local FF updates.

## Slide 11: Objective 2 Full Results Part 1
- Point to top region:
  - ReLU is first, GELU and SiLU are close but slightly behind.
- Interpretation:
  - Advanced smooth activations do not automatically beat ReLU in this FF setup.

## Slide 12: Objective 2 Full Results Part 2
- Mention tail behavior:
  - Some activations perform much worse, indicating mismatch with FF goodness dynamics.
- Broad lesson:
  - Activation design still matters, but not all popular activations transfer well to local FF learning.

## Slide 13: Objective 3 Local Spatial Goodness Detail
- Explain what local spatial FF does:
  - Instead of a global fully connected view, it learns local patch-level features with no weight sharing.
- Explain goodness margin:
  - Margin equals train_pos_goodness minus train_neg_goodness.
  - Bigger margin means better separation of positive and negative local representations.
- Clarify metric role:
  - In the updated objective, we report both margin and test accuracy.
  - Key message: good margin alone does not guarantee high classification accuracy.

## Slide 14: Objective 3 Full Results (Updated)
- Point out best setup:
  - unsquared_goodness gives the best test accuracy at about 57.81 percent.
- Explain interpretation carefully:
  - student_t_activation has the highest margin but not the best accuracy.
  - So, in our run, margin and final classification performance are not perfectly aligned.

## Slide 15: Objective 3 vs Original Baseline
- State the headline clearly:
  - Original baseline (sum_sq + relu, fully connected FF) is 92.63 percent accuracy.
  - Best local-spatial setup is 57.81 percent accuracy.
- Quantify the gap:
  - Local-spatial is lower by 34.82 percentage points.
- Interpretation:
  - In this configuration, local-spatial FF underperformed strongly versus the original baseline.

## Slide 16: Limitations and Gap vs 1 Percent Error
- Core explanation:
  - Paper-level results use larger compute budget, heavier model/training recipe, and more extensive tuning.
- Additional limitations to mention:
  - Subset-based sweeps were used for practicality.
  - Limited hyperparameter search and mostly single-seed analysis.
  - Objective 3 local architecture is likely under-tuned (patch size, stride, channels).
  - Local objectives may need different hyperparameters than fully connected baseline.
- Position honestly:
  - Our study is strong for comparative insights, not a full reproduction claim.

## Slide 17: Conclusion and Next Steps
- Summarize objective-wise:
  - Objective 1: goodness design has strong impact.
  - Objective 2: ReLU remained strongest in this setup.
  - Objective 3: local-spatial model underperformed baseline in current run (57.81 percent vs 92.63 percent).
- Forward plan:
  - Scale model/training toward paper settings.
  - Add multi-seed confidence intervals.
  - Tune local-spatial architecture and optimization specifically for classification accuracy.

## Slide 18: Reference
- Closing line:
  - Our implementation and analysis are based on Hinton's Forward-Forward preliminary investigations paper.
- Q and A transition:
  - Thank you, I am happy to take questions.

## Optional Delivery Timing (10 to 12 min)
- Slide 1 to 2: 1 minute
- Slide 3 to 5: 2 minutes
- Slide 6 to 9: 3 minutes
- Slide 10 to 12: 2 minutes
- Slide 13 to 15: 2 minutes
- Slide 16 to 18: 1.5 to 2.5 minutes
