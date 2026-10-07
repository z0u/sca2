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
protocol_version  # unused variable (src/mini/lit/serve.py:104 — BaseHTTPRequestHandler)
_.close_connection  # unused attribute (src/mini/lit/serve.py:159 — BaseHTTPRequestHandler)
request_queue_size  # unused variable (src/mini/lit/serve.py:172 — socketserver.TCPServer)
_.log_message  # unused method (src/mini/lit/serve.py:165, src/mini/report_print.py:153 — overridden to silence the log)
# pypdf reads the page box on save.
_.CropBox  # unused attribute (src/mini/report_print.py:367)

# Fixtures requested for their side effect (binding an ambient store, setting the
# Modal Environment); the test body never names them.
local_store  # unused variable (tests/mini/test_apparatus.py, tests/mini/test_store_gc.py)
modal_env_dev  # unused variable (tests/mini/test_apparatus.py:611, 634)

# Pydantic metadata fields: written at construction, read only via serialization.
author  # unused variable (src/sca/config.py:78)
fixes  # unused variable (src/sca/config.py:83)
total_chars  # unused variable (src/sca/config.py:86, 100)
language  # unused variable (src/sca/config.py:89)
training_tokens  # unused variable (src/sca/training/metrics.py:8)

# Logging config knobs: part of SimpleLoggingConfig's public surface.
_.base_level  # unused method (src/mini/logging.py:67)
_.to_stream  # unused method (src/mini/logging.py:72)
_.critical  # unused method (src/mini/logging.py:77)
_.trace  # unused method (src/mini/logging.py:102)
SimpleLoggingConfig  # unused class (src/mini/logging.py:44)

# Dormant infra, kept deliberately: candidates for deletion if no M2 experiment picks
# them up.
EntropySeries  # unused class (src/subline/series.py:29)
plot_lr_finder  # unused function (src/utils/lr_finder/vis.py:10)
group_properties_by_scale  # unused function (src/mini/temporal/vis.py:40)
Debouncer  # unused class (src/mini/_debounce.py:16 — BackgroundEmitter took over the hot path)

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
ADOPTED  # docs/m2/ex-2.2.20/report.py:280
E1_BAND  # docs/m2/ex-2.2.20/report.py:314
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
