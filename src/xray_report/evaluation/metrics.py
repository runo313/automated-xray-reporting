"""
Evaluation metrics.
  - findings classifier: per-finding AUC, F1
  - decoder: BLEU-4, findings-overlap check (generated report vs predicted/true findings)
TODO: implement once first training runs produce predictions to evaluate against.
"""


def per_finding_auc_f1(y_true, y_pred_probs, findings: list[str]):
    raise NotImplementedError


def bleu4(references: list[list[str]], hypotheses: list[list[str]]):
    raise NotImplementedError


def findings_overlap(generated_reports: list[str], predicted_findings, true_findings):
    raise NotImplementedError
