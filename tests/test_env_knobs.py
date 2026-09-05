"""No-GPU checks for agforge.env_knobs. Run from a checkout of this branch:

    python tests/test_env_knobs.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agforge.env_knobs import EnvKnobError, env_bool, env_float, env_int, env_str  # noqa: E402


class EnvKnobTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("AGF_TEST_KNOB", None)

    def test_unset_returns_default(self):
        os.environ.pop("AGF_TEST_KNOB", None)
        self.assertEqual(env_float("AGF_TEST_KNOB", 1.5e-5), 1.5e-5)
        self.assertIs(env_float("AGF_TEST_KNOB", 1.5e-5), 1.5e-5)

    def test_blank_ordinary_means_default(self):
        os.environ["AGF_TEST_KNOB"] = ""
        self.assertEqual(env_float("AGF_TEST_KNOB", 42.0), 42.0)
        os.environ["AGF_TEST_KNOB"] = "   "
        self.assertEqual(env_float("AGF_TEST_KNOB", 42.0), 42.0)

    def test_blank_pin_is_fatal(self):
        os.environ["AGF_TEST_KNOB"] = ""
        with self.assertRaises(EnvKnobError) as ctx:
            env_float("AGF_TEST_KNOB", 312225.769, blank_ok=False)
        msg = str(ctx.exception)
        self.assertIn("AGF_TEST_KNOB", msg)
        self.assertIn("Empty is not a pin", msg)
        self.assertNotIn("set it empty to use the default", msg.lower())

    def test_unset_pin_still_defaults(self):
        os.environ.pop("AGF_TEST_KNOB", None)
        self.assertEqual(
            env_float("AGF_TEST_KNOB", 312225.769, blank_ok=False),
            312225.769,
        )

    def test_malformed_names_the_knob(self):
        os.environ["AGF_TEST_KNOB"] = "not-a-number"
        with self.assertRaises(EnvKnobError) as ctx:
            env_float("AGF_TEST_KNOB", 1.0)
        self.assertIn("AGF_TEST_KNOB", str(ctx.exception))
        self.assertIn("not-a-number", str(ctx.exception))

    def test_malformed_pin_does_not_advertise_empty(self):
        os.environ["AGF_TEST_KNOB"] = "nope"
        with self.assertRaises(EnvKnobError) as ctx:
            env_float("AGF_TEST_KNOB", 1.0, blank_ok=False)
        self.assertIn("empty is invalid", str(ctx.exception).lower())

    def test_override(self):
        os.environ["AGF_TEST_KNOB"] = "1.5e-5"
        self.assertEqual(env_float("AGF_TEST_KNOB", 9.9), 1.5e-5)

    def test_bool_and_int(self):
        os.environ["AGF_TEST_KNOB"] = "1"
        self.assertTrue(env_bool("AGF_TEST_KNOB", False))
        os.environ["AGF_TEST_KNOB"] = "3"
        self.assertEqual(env_int("AGF_TEST_KNOB", 0), 3)
        os.environ["AGF_TEST_KNOB"] = "grid"
        self.assertEqual(env_str("AGF_TEST_KNOB", "none"), "grid")


if __name__ == "__main__":
    unittest.main()
