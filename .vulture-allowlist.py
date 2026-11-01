# This file is used to allowlist unused code in the project.
# https://github.com/jendrikseipp/vulture?tab=readme-ov-file#handling-false-positives
# type: ignore

# Context-manager and API-shape false positives: the signatures are fixed by the
# protocol being implemented, not by what the body reads.
exc_val  # unused variable (src/mini/progress_display.py:85, 92 — __exit__/__aexit__)
exc_tb  # unused variable (src/mini/progress_display.py:85, 92 — __exit__/__aexit__)
Styler  # unused import (src/mini/temporal/dopesheet.py:13 — TYPE_CHECKING-only, used in overload return)
_.create_artists  # unused method (src/mini/vis/plt.py:184 — matplotlib HandlerBase override)
fontsize  # unused variable (src/mini/vis/plt.py:184 — create_artists' signature)

# Standard-library hooks: http.server and socketserver read these attributes and call
# these methods on the subclass.
protocol_version  # unused variable (src/mini/lit/serve.py:108 — BaseHTTPRequestHandler)
_.close_connection  # unused attribute (src/mini/lit/serve.py:163 — BaseHTTPRequestHandler)
request_queue_size  # unused variable (src/mini/lit/serve.py:176 — socketserver.TCPServer)
_.log_message  # unused method (src/mini/lit/serve.py:169, src/mini/report_print.py:154 — overridden to silence the log)
# Python-Markdown calls this on each extension it is given.
_.extendMarkdown  # unused method (src/mini/lit/notes.py:537)
# pymdownx.superfences calls a custom fence formatter with this signature.
css_class  # unused variable (src/mini/lit/page.py:49 — _mermaid_fence)
# JAX reads its compilation-cache module state; install() sets it after swapping in the shared cache.
_._cache_initialized  # unused attribute (src/mini/compile_cache.py:421)
# pypdf reads the page box on save.
_.CropBox  # unused attribute (src/mini/report_print.py:368)

# Fixtures requested for their side effect (binding an ambient store, setting the
# Modal Environment); the test body never names them.
local_store  # unused variable (tests/mini/test_apparatus.py, tests/mini/test_store_gc.py)
modal_env_dev  # unused variable (tests/mini/test_apparatus.py:613, 636)

# Pydantic metadata fields: written at construction, read only via serialization.
author  # unused variable (src/sca/config.py:80)
fixes  # unused variable (src/sca/config.py:85)
total_chars  # unused variable (src/sca/config.py:88, 102)
language  # unused variable (src/sca/config.py:91; also src/mini/lit/page.py:49, _mermaid_fence's signature)
training_tokens  # unused variable (src/sca/training/metrics.py:8)

# Logging config knobs: part of SimpleLoggingConfig's public surface.
_.base_level  # unused method (src/mini/logging.py:67)
_.to_stream  # unused method (src/mini/logging.py:72)
_.critical  # unused method (src/mini/logging.py:77)
SimpleLoggingConfig  # unused class (src/mini/logging.py:44)

# Dormant infra, kept deliberately: candidates for deletion if no M2 experiment picks
# them up.
EntropySeries  # unused class (src/subline/series.py:29)
plot_lr_finder  # unused function (src/utils/lr_finder/vis.py:10)
group_properties_by_scale  # unused function (src/mini/temporal/vis.py:40)
Debouncer  # unused class (src/mini/_debounce.py:16 — BackgroundEmitter took over the hot path)
# The flags a role sets to reproduce a run from before the compile cache (eng/determinism.md).
DETERMINISTIC_XLA_FLAGS  # unused variable (src/mini/compile_cache.py:34)

# Experiments define `smoke(ctx)` beside `main(ctx)`: a small run of the same DAG,
# started by hand in place of `main` while the experiment is being built.
smoke  # unused function (docs/m2/*/experiment.py)
# The pack-timing probe has no report; its memo in the project files gives costs per
# seed-step at these prices.
PRICE_PER_HOUR  # unused variable (docs/m2/pack-timing/experiment.py:81)

# Published reports. Their code is part of the record and pinned in docs/publish.lock,
# so leftovers stay rather than change a frozen source. Prose reads names through
# f-strings, so vulture already sees those; these are unused.
c_nt  # docs/m1/ex-2.9.2/report.py:295
ANTI_CENTROID_OP  # docs/m2/ex-2.1.11/experiment.py:162
AMENDMENT_RULES  # docs/m2/ex-2.1.11/experiment.py:283 — named in prose as `experiment.AMENDMENT_RULES`
OFFKEY_CEILING  # docs/m2/ex-2.1.12/experiment.py:46
H2_PARTIAL  # docs/m2/ex-2.1.12/experiment.py:68
hex_train  # docs/m2/ex-2.1.5/experiment.py:141
crossings  # docs/m2/ex-2.1.5/report.py:1143
conds_by_name  # docs/m2/ex-2.1.7/report.py:224
geometry_effect  # docs/m2/ex-2.1.7/report.py:958
op1_ratio  # docs/m2/ex-2.1.7/report.py:1162
ratio_below  # docs/m2/ex-2.1.7/report.py:1169
measure_pull_dilution  # docs/m2/ex-2.1.7/experiment.py:288 — re-derives the constants recorded above it
sim_target  # docs/m2/ex-2.1.8/report.py:1146
metrics218  # docs/m2/ex-2.1.9/report.py:320
mellowmax_min  # docs/m2/ex-2.1.9/experiment.py:108, docs/m2/ex-2.1.10/experiment.py:138
EX229_ARRAYS_REF  # docs/m2/ex-2.2.10/experiment.py:74
HUE_ROTATION_CHECK  # docs/m2/ex-2.2.11/experiment.py:171
sv_line  # docs/m2/ex-2.2.11/experiment.py:207
LEVEL_REFS  # docs/m2/ex-2.2.11/experiment.py:227
ALPHA_REFS  # docs/m2/ex-2.2.11/experiment.py:235
EOL_ROW  # docs/m2/ex-2.2.11/experiment.py:240
_.stat_slices  # docs/m2/ex-2.2.12/report.py:107
_.is_reference  # docs/m2/ex-2.2.13/experiment.py:99 — property
SPREAD_STATISTIC  # docs/m2/ex-2.2.13/experiment.py:112, docs/m2/ex-2.2.16/experiment.py:270
skill_score  # docs/m2/ex-2.2.16/posterior.py:245
context_roles  # docs/m2/ex-2.2.16/report.py:519
N_PASS  # docs/m2/ex-2.2.16/report.py:619
DIST_N_OTHER  # docs/m2/ex-2.2.17/report.py:261
DIST_N_NONE  # docs/m2/ex-2.2.17/report.py:262
N_NOISE  # docs/m2/ex-2.2.17/report.py:272
NOISE_MEAN  # docs/m2/ex-2.2.17/report.py:283
COND_EEM  # docs/m2/ex-2.2.17/report.py:606
COND_KL  # docs/m2/ex-2.2.17/report.py:607
CONFUSION_BAYES  # docs/m2/ex-2.2.17/report.py:668
YARD_SPREAD  # docs/m2/ex-2.2.18/report.py:127
YARD_OP_SPREAD  # docs/m2/ex-2.2.18/report.py:129
BEST_PRIOR  # docs/m2/ex-2.2.18/report.py:343
ADOPTED  # docs/m2/ex-2.2.20/report.py:273
E1_BAND  # docs/m2/ex-2.2.20/report.py:313
STOCHASTIC  # docs/m2/ex-2.2.5/experiment.py:104
FIXES  # docs/m2/ex-2.2.7/report.py:37
red_acc_sd  # docs/m2/ex-2.2.8/report.py:128
deficit_sd  # docs/m2/ex-2.2.8/report.py:129
_.feasible_mix  # docs/m2/ex-2.2.8/report.py:144 — property
t00_whole_feasible  # docs/m2/ex-2.2.8/report.py:183
calibration_clears  # docs/m2/ex-2.2.9/report.py:106
_.ref_em  # docs/m2/ex-2.2.9/report.py:238
RED_ANSWER_LINES_BOTH_WAYS  # docs/m2/ex-2.2.9/experiment.py:327
POSITION_INDEX  # docs/m2/geometry-rsa/experiment.py:90
# Constants named in a review marker (`<!-- REVIEW: ... Verify: `NAME` -->`), which
# vulture reads as a comment.
ANCHOR_O  # docs/m2/filtered-corpus/report.py:274
CONTROL_O  # docs/m2/filtered-corpus/report.py:275
OTHER_O  # docs/m2/filtered-corpus/report.py:289
CONTROL_SPILL  # docs/m2/soft-pool-scout/report.py:422
# Unused, in published reports.
TASK_MISS  # docs/m2/anneal-confirm/report.py:121, docs/m2/fixed-confirm/report.py:141
NOT_KEPT  # docs/m2/anneal-confirm/report.py:186, docs/m2/fixed-confirm/report.py:213
LOW_SEEDS  # docs/m2/anneal-confirm/report.py:463
OWN_UNDER_EDIT  # docs/m2/block1-edit/report.py:96
ATTN1  # docs/m2/block1-edit/report.py:110
SHARE_SPREAD  # docs/m2/edit-effects/report.py:590
LATCH_COUNTS  # docs/m2/embedding-lean/report.py:113
N_SPILL_LONG  # docs/m2/embedding-lean/report.py:114
last_block  # docs/m2/ex-2.2.21/report.py:132
CTRL16_EEM  # docs/m2/ex-2.2.21/report.py:145
S1_PROMOTED  # docs/m2/ex-2.2.21/report.py:850
VERIFY  # docs/m2/ex-2.2.22/experiment.py:72 — records the condition setting
CAP_X  # docs/m2/ex-2.2.22/report.py:691
H2_VERDICT  # docs/m2/ex-2.2.22/report.py:1162
REPLICATE_CRIT  # docs/m2/ex-2.2.22/report.py:1164
VARIANTS_REF  # docs/m2/filtered-corpus/experiment.py:230
OTHER_GAP_ALONG  # docs/m2/filtered-corpus/report.py:229
TASK_DROPPED  # docs/m2/filtered-corpus/report.py:436
NULL_TASK  # docs/m2/filtered-corpus/report.py:439
IDEAL_TASK  # docs/m2/filtered-corpus/report.py:440
SPILL_WORST  # docs/m2/fixed-effects/report.py:213
MONOTONE  # docs/m2/fixed-effects/report.py:222, docs/m2/slice-terms/report.py:213
OVER_SLICES  # docs/m2/fixed-effects/report.py:800
D_ROW_DIAG  # docs/m2/fixed-effects/report.py:951
FALL_AFTER  # docs/m2/fixed-effects/report.py:971
N_FALL_AFTER  # docs/m2/fixed-effects/report.py:972
SLOPE_CONTROL  # docs/m2/late-pull/report.py:177
R_S1  # docs/m2/late-pull/report.py:179
R_705  # docs/m2/late-spill-scout/report.py:444
MID_UNDER_EDIT_EQ  # docs/m2/mid-block-edit/report.py:92
SEED_RANGE  # docs/m2/optimizer-scout/report.py:58
MEMO_SGD_LR  # docs/m2/optimizer-scout/report.py:59
ADAM_TOP  # docs/m2/optimizer-scout/report.py:687
ALPHA_Q  # docs/m2/recipe-compare/report.py:310
SHIFT  # docs/m2/reference-baseline/report.py:227
QUIETEST  # docs/m2/reference-baseline/report.py:254
SOFT_REM  # docs/m2/reference-baseline/report.py:272
SYNTAX_ON_SPIKE  # docs/m2/rotated-basis/report.py:196
FULL_REMOVAL  # docs/m2/slice-terms/report.py:208
POOL_OK  # docs/m2/slice-terms/report.py:216
N_FIN_LOST  # docs/m2/soft-pool-scout/report.py:333
OTHER_LONG  # docs/m2/spill-by-position/report.py:595
WITHIN  # docs/m2/spill-timing/report.py:134
LATE_KIND  # docs/m2/spill-timing/report.py:255
MIDDLE_RUNS  # docs/m2/tau-lambda-sweep/report.py:197
NEAR_RECIPE  # docs/m2/tau-lambda-sweep/report.py:200
RECIPE_LATCHED  # docs/m2/tau-lambda-sweep/report.py:201
RECIPE_CLEAN  # docs/m2/tau-lambda-sweep/report.py:202
BOOT_MAX  # docs/m2/write-floor/report.py:152
BOOT_MEDIAN  # docs/m2/write-floor/report.py:153
EXCESS_MIN  # docs/m2/write-floor/report.py:154
