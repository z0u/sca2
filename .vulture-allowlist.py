# This file is used to allowlist unused code in the project.
# https://github.com/jendrikseipp/vulture?tab=readme-ov-file#handling-false-positives
# type: ignore

# Context-manager and API-shape false positives: the signatures are fixed by the
# protocol being implemented, not by what the body reads.
exc_val  # unused variable (src/mini/progress_display.py:115, 122 — __exit__/__aexit__)
exc_tb  # unused variable (src/mini/progress_display.py:115, 122 — __exit__/__aexit__)
create_if_missing  # unused variable (tests/mini/test_apparatus.py:271 — mimics modal.Volume.from_name)
Styler  # unused import (src/mini/temporal/dopesheet.py:13 — TYPE_CHECKING-only, used in overload return)

# Pydantic metadata fields: written at construction, read only via serialization.
author  # unused variable (src/sca/config.py:64)
fixes  # unused variable (src/sca/config.py:69)
total_chars  # unused variable (src/sca/config.py:72, 86)
language  # unused variable (src/sca/config.py:75)
total_tokens  # unused variable (src/sca/config.py:83)
training_tokens  # unused variable (src/sca/training/metrics.py:8)

# Logging config knobs: part of SimpleLoggingConfig's public surface.
_.base_level  # unused method (src/mini/logging.py:67)
_.to_stream  # unused method (src/mini/logging.py:72)
_.critical  # unused method (src/mini/logging.py:77)
_.trace  # unused method (src/mini/logging.py:102)
SimpleLoggingConfig  # unused class (src/mini/logging.py:44)

# Dormant infra, kept deliberately (see todo.md): candidates for deletion if the
# first M2 transformer experiments don't pick them up.
EntropySeries  # unused class (src/subline/series.py:26)
Subline  # unused class (src/subline/subline.py:11)
lr_finder_search  # unused function (src/utils/lr_finder/lr_finder.py:18)
plot_lr_finder  # unused function (src/utils/lr_finder/vis.py:10)
group_properties_by_scale  # unused function (src/mini/temporal/vis.py:41)
