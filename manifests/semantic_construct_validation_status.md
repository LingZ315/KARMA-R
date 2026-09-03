# Pre-target semantic construct-validation status

Status: **PARTIAL — HUMAN SCHEMA EVIDENCE VERIFIED; FROZEN CLASSIFIER COMPARISON NOT YET AVAILABLE**

The retained anonymized human audit contains 1,000 unique hashed items. Direct recomputation from the file gives initial annotator agreement 0.937 and Cohen's kappa 0.9223777542; 110 rows required adjudication, 11 carried the ambiguity flag, and 76 carried a quality-control flag. These are genuine human-label provenance checks, not Qwen classifier metrics.

The ten adjudicated labels have a frozen, many-to-one mapping to the seven Panel-C primary classes and four optional subtypes. However, the public-safe derivative contains hashed IDs only and no images/questions, while no compatible frozen Qwen3-VL-32B output ledger keyed to those hashes is present. Therefore primary-class accuracy, macro-F1, confusion matrix, per-class recall, subtype accuracy, and ambiguity behavior for the classifier cannot be computed without inventing a linkage or rerunning linked private images.

`validate_semantic_constructs.py` freezes the exact comparison once a legally available, exactly linked classifier ledger exists. It rejects candidate correctness, route, and utility fields. This check may be completed before target candidate scoring, but it cannot tune the classifier or alter the externally timestamped protocol. Until then, construct-validation audit item 25 is PARTIAL, not PASS.
