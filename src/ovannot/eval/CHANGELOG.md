# Evaluation changelog

The evaluation is frozen once the owner confirms the end of M1 (see `question.txt`, Q4). From
then on, every change to `src/ovannot/eval/` gets a dated entry here, with the runs to redo.

- 2026-10-28: first version. RF100-VL protocol: pycocotools `COCOeval` on boxes,
  `maxDets = (1, 10, 500)`, one dataset at a time. Precision, recall and F1 at IoU 0.5, at the
  pipeline's `score_threshold`, greedy matching by score within each class. Sanity tests in
  `tests/test_eval.py`. Not yet frozen.
