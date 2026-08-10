"""TEMPORARY: a deliberate red build, to verify that one CI run reports every
failing check rather than stopping at the first. Reverted in the next commit."""


def  test_ci_probe():
    assert  1 == 2, "deliberate failure"
