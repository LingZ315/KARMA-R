# Panel-C model-pool audit

Status: **FROZEN WITHOUT PANEL-C PREDICTIONS OR OUTCOMES**

| Route | Family / nominal scale | Architecture description used for audit | Revision | Prior panel? | Role |
|---|---|---|---|---:|---|
| `smolvlm_incumbent` | SmolVLM / repository-named compact scale | Repository-native vision encoder + causal language model | `81cd9a775a4d644f2faf4e7becff4559b46b14c7` | Yes, Replication B | Incumbent |
| `granite4_vision` | Granite 4.0 Vision / 3B | Repository-native vision-language transformer | `bf108f36960fb4df79bf035e506c592f4ee3c2d3` | Yes, Replication B | Retained candidate |
| `ovis25_9b` | Ovis 2.5 / 9B | Repository-native multimodal causal language model | `d73b2283ae2a930b7762f8d7b8b8a3f0f3b5c3bd` | Yes, Replication B | Retained candidate |
| `phi4_mm` | Phi-4 Multimodal | Repository-native multimodal causal language model | `93f923e1a7727d1c4f446756212d9d3e8fcc5d81` | Yes, Replication B | Retained candidate |
| `qwen3vl_4b` | Qwen3-VL / 4B | Repository-native vision-language transformer | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | No | Refreshed candidate |
| `internvl35_4b` | InternVL 3.5 / 4B | Repository-native vision-language transformer | `6bd4487402110ef9889ba50eb7aefeb302526fed` | No | Refreshed candidate |

The pool spans compact/medium parameter scales, six distinct weight sets, and multiple independently developed model families. Three candidates and the incumbent preserve continuity with Replication B; two refreshed families reduce dependence on that exact candidate universe. Resolved weight bytes and card-reported licenses are recorded in `panel_c_model_pool.json`.

Decision: **do not expand the pool before external timestamp**. Adding models merely to increase count would raise inference cost and introduce a new outcome-independent design decision without a demonstrated diversity gap. Any later pool change requires a new protocol version, manifests, archive, and external timestamp; it cannot be based on Panel-C performance.
