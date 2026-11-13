"""End-to-end tests of app_user.py with Streamlit's AppTest and an in-memory fake Sheet.

Run from the repository root:  python -m unittest discover -s tests -v
Nothing here opens a network connection: gspread's constructors are replaced by tests/fake_sheet.py.
"""
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from streamlit.testing.v1 import AppTest  # noqa: E402

import fake_sheet  # noqa: E402
import rde_core as core  # noqa: E402

URL = "fake://results-sheet"
STUDY = "study-code-for-tests"
V1_HEADER = ["Annotator_ID", "Dataset", "Task_ID", "Original_Label", "Instruction", "Workspace",
             "Clarity", "Ambiguity_Type", "Feasibility", "Safety", "Comments"]
TRAINING_VERSION, TRAINING = core.load_training(ROOT / "training_tasks.json")
SIMPLE = {"clarity": core.CLARITY_OPTIONS[0], "ambiguity_types": [],
          "feasibility": core.FEASIBILITY_OPTIONS[0], "safety": core.SAFETY_OPTIONS[0]}


def setUpModule():
    os.chdir(ROOT)  # the app opens its data files relative to the working directory


def new_app(study_code=STUDY):
    at = AppTest.from_file(str(ROOT / "app_user.py"), default_timeout=180)
    at.secrets["GOOGLE_CREDENTIALS"] = "{}"
    at.secrets["SHEET_URL"] = URL
    if study_code is not None:
        at.secrets["STUDY_CODE"] = study_code
    return at.run()


def login(at, annotator, pin, register=True, code=STUDY, pin_again=None):
    """Fill the sidebar sign-in form and press the button (register=True: first visit)."""
    at.radio(key="login_mode").set_value(core.REGISTER if register else core.RETURNING).run()
    at.text_input(key="login_study_code").input(code)
    at.text_input(key="login_id").input(annotator)
    at.text_input(key="login_pin").input(pin)
    if register:
        at.text_input(key="login_pin2").input(pin if pin_again is None else pin_again)
    at.run()
    return at.button(key="login_btn").click().run()


def answer(at, suffix, a):
    at.radio(key=f"clarity_{suffix}").set_value(a["clarity"]).run()
    if core.needs_types(a["clarity"]):
        at.multiselect(key=f"ambig_type_{suffix}").set_value(a["ambiguity_types"])
    at.radio(key=f"feas_{suffix}").set_value(a["feasibility"])
    at.radio(key=f"safe_{suffix}").set_value(a["safety"])
    return at.run()


def expected(t):
    return {"clarity": t["expected_clarity"], "ambiguity_types": t.get("expected_ambiguity_types") or [],
            "feasibility": t["expected_feasibility"], "safety": t["expected_safety"]}


def texts(elements):
    return [e.value for e in elements]


def do_training(test, at, wrong=False):
    for t in TRAINING:
        a = expected(t)
        if wrong:
            a["safety"] = next(s for s in core.SAFETY_OPTIONS if s != a["safety"])
        answer(at, f"train_{t['id']}", a)
        at.button(key=f"check_btn_{t['id']}").click().run()
        if wrong:
            test.assertTrue(any("differed" in w and "Safety" in w for w in texts(at.warning)), t["id"])
            test.assertFalse(any("matches" in s for s in texts(at.success)), t["id"])
        else:
            test.assertTrue(any("matches the expert benchmark" in s for s in texts(at.success)), t["id"])
            test.assertFalse(texts(at.warning), t["id"])
        at.button(key=f"next_train_btn_{t['id']}").click().run()
    at.button(key="proceed_live").click().run()
    test.assertTrue(at.session_state["is_trained"])
    return at


def current(at):
    flat = [tid for _, ids in at.session_state["order"] for tid in ids]
    return flat, flat[at.session_state["pos"]]


def submit_current(at, a=SIMPLE):
    _, tid = current(at)
    answer(at, tid, a)
    at.button(key=f"submit_{tid}").click().run()
    return tid


class AppCase(unittest.TestCase):
    def setUp(self):
        self.book = fake_sheet.install(URL, [V1_HEADER])
        self.sheet = self.book.sheet1

    def rows(self, annotator=None):
        return [core.pad(r) for r in self.sheet.values[1:] if annotator is None or r[0] == annotator]


def signed_in(at):
    return bool(at.session_state["annotator_id"]) if "annotator_id" in at.session_state else False


class TestSignIn(AppCase):
    def registrations(self):
        ws = self.book.sheets.get(core.REG_SHEET)
        return ws.values if ws else None

    def test_registration(self):
        at = login(new_app(), "  Jane Doe-1 ", "4821")
        self.assertTrue(signed_in(at))
        self.assertEqual(at.session_state["annotator_id"], "jane doe-1")    # trimmed, lowercased
        reg = self.registrations()
        self.assertEqual(reg[0], core.REG_HEADER)                            # worksheet created
        self.assertEqual(len(reg), 2)
        self.assertEqual(reg[1][0], "jane doe-1")
        self.assertTrue(reg[1][1].startswith("pbkdf2_sha256$"))
        self.assertTrue(core.verify_pin("4821", reg[1][1]))
        self.assertRegex(reg[1][2], r"^\d{4}-\d{2}-\d{2}T")
        for ws in self.book.sheets.values():                                 # the PIN is stored nowhere
            self.assertFalse(any("4821" in cell for row in ws.values for cell in row if cell != reg[1][1]))
        self.assertTrue(any("Calibration Task 1" in m.value for m in at.markdown))  # then training

    def test_returning_user(self):
        a = do_training(self, login(new_app(), "alice", "pin-alice"))
        done = submit_current(a)
        r = login(new_app(), "ALICE", "pin-alice", register=False)
        self.assertTrue(signed_in(r))
        self.assertTrue(r.session_state["is_trained"])
        self.assertIn(done, r.session_state["answers"])
        self.assertEqual(len(self.registrations()), 2)                       # no second registration

    def test_wrong_pin_and_unknown_id(self):
        login(new_app(), "alice", "pin-alice")
        for who, pin in (("alice", "pin-alicE"), ("alice", ""), ("mallory", "pin-alice")):
            at = login(new_app(), who, pin, register=False)
            self.assertEqual(texts(at.sidebar.error), [core.NOT_RECOGNISED], (who, pin))
            self.assertFalse(signed_in(at))

    def test_taken_id(self):
        login(new_app(), "alice", "pin-alice")
        at = login(new_app(), "Alice ", "other-pin")
        self.assertEqual(texts(at.sidebar.error),
                         ["This ID is already registered; choose another or enter your PIN."])
        self.assertFalse(signed_in(at))
        self.assertEqual(len(self.registrations()), 2)
        self.assertTrue(core.verify_pin("pin-alice", self.registrations()[1][1]))

    def test_missing_or_wrong_study_code(self):
        for app, code, message in ((new_app(study_code=None), STUDY, core.NO_STUDY_CODE),
                                   (new_app(study_code=""), "", core.NO_STUDY_CODE),
                                   (new_app(), "wrong-code", core.WRONG_STUDY_CODE)):
            at = login(app, "alice", "pin-alice", code=code)
            self.assertEqual(texts(at.sidebar.error), [message])
            self.assertFalse(signed_in(at))
        self.assertEqual(self.book.reads, 0)          # nothing is read before the study code is checked
        self.assertIsNone(self.registrations())

    def test_v1_id_registers_once(self):
        v1 = [["Carol", "SaGC", "5", "0", "goal", "ws", core.CLARITY_OPTIONS[0], "N/A",
               core.FEASIBILITY_OPTIONS[0], core.SAFETY_OPTIONS[0], "v1"],
              ["Carol", core.TRAINING_MARKER, "PASSED"] + ["N/A"] * 7 + ["Completed calibration"]]
        self.sheet.values += [list(r) for r in v1]
        at = login(new_app(), "carol", "pin-carol")
        self.assertTrue(signed_in(at))
        self.assertFalse(at.session_state["is_trained"])        # the v1 marker does not count for t3
        self.assertEqual(at.session_state["answers"], {})       # v1 rows are not v2 answers
        self.assertEqual(self.sheet.values[1:3], v1)             # v1 rows untouched
        again = login(new_app(), "carol", "another-pin")         # only once
        self.assertEqual(texts(again.sidebar.error), [core.TAKEN])

    def test_id_and_pin_rules(self):
        for who, pin, pin_again, message in (("bad!id", "4821", None, core.ID_RULE),
                                             ("dave", "123", None, core.PIN_RULE),
                                             ("dave", "4821", "4812", "The two PINs differ.")):
            at = login(new_app(), who, pin, pin_again=pin_again)
            self.assertEqual(texts(at.sidebar.error), [message], who)
            self.assertFalse(signed_in(at))
        self.assertEqual(self.registrations()[1:], [])
        # longer IDs cannot be typed (the field holds 40 characters; core.validate_id also rejects them)
        self.assertEqual(new_app().text_input(key="login_id").proto.max_chars, 40)

    def test_simultaneous_registration_first_wins(self):
        import app_user
        login(new_app(), "zoe", "pin-zoe-1")       # creates the worksheet
        ws = self.book.sheets[core.REG_SHEET]
        del ws.values[1:]
        original = ws.get_all_values
        calls = []

        def racing_read(*a, **k):                  # another session registers "eve" between our read and append
            values = original()
            if not calls:
                ws.values.append(core.registration_row("eve", "their-pin"))
            calls.append(1)
            return values
        ws.get_all_values = racing_read
        self.assertEqual(app_user.authenticate(self.book, "eve", "my-pin", "my-pin", core.REGISTER), core.TAKEN)
        ws.get_all_values = original
        self.assertTrue(core.verify_pin("their-pin", core.find_registration(ws.values, "eve")[1]))


class TestUnreadableSheet(AppCase):
    def test_read_failure_at_sign_in_stops(self):
        self.book.fail = {"read"}
        at = login(new_app(), "alice", "pin-alice")
        self.assertTrue(any("could not be reached" in e for e in texts(at.error)))
        self.assertFalse("annotator_id" in at.session_state and at.session_state["annotator_id"])

    def test_open_failure_at_sign_in_stops(self):
        self.book.fail = {"open"}
        at = login(new_app(), "alice", "pin-alice")
        self.assertTrue(any("could not be reached" in e for e in texts(at.error)))

    def test_write_failure_keeps_item_and_reports(self):
        at = do_training(self, login(new_app(), "alice", "pin-alice"))
        flat, tid = current(at)
        self.book.fail = {"write"}
        answer(at, tid, SIMPLE)
        at.button(key=f"submit_{tid}").click().run()
        self.assertTrue(any("Nothing was saved" in e for e in texts(at.error)))
        self.assertEqual(current(at)[1], tid)
        self.assertNotIn(tid, at.session_state["answers"])
        self.assertEqual([r for r in self.rows("alice") if r[1] != core.TRAINING_MARKER], [])
        self.book.fail = set()
        at.button(key=f"submit_{tid}").click().run()
        self.assertEqual([r[2] for r in self.rows("alice") if r[1] != core.TRAINING_MARKER], [tid])

    def test_read_failure_before_write_is_reported(self):
        at = do_training(self, login(new_app(), "alice", "pin-alice"))
        _, tid = current(at)
        answer(at, tid, SIMPLE)
        self.book.fail = {"read"}
        at.button(key=f"submit_{tid}").click().run()
        self.assertTrue(any("could not be reached" in e for e in texts(at.error)))
        self.assertNotIn(tid, at.session_state["answers"])


class TestSubmitBlocked(AppCase):
    def test_submit_disabled_until_every_question_answered(self):
        at = do_training(self, login(new_app(), "alice", "pin-alice"))
        _, tid = current(at)
        submit = lambda: at.button(key=f"submit_{tid}")  # noqa: E731
        # nothing preselected
        self.assertIsNone(at.radio(key=f"clarity_{tid}").value)
        self.assertIsNone(at.radio(key=f"feas_{tid}").value)
        self.assertIsNone(at.radio(key=f"safe_{tid}").value)
        self.assertTrue(submit().disabled)
        at.radio(key=f"clarity_{tid}").set_value(core.CLARITY_OPTIONS[3]).run()
        self.assertTrue(submit().disabled)
        at.radio(key=f"feas_{tid}").set_value(core.FEASIBILITY_OPTIONS[0])
        at.radio(key=f"safe_{tid}").set_value(core.SAFETY_OPTIONS[0]).run()
        self.assertTrue(submit().disabled)  # ambiguity type still missing
        at.multiselect(key=f"ambig_type_{tid}").set_value([core.AMBIGUITY_OPTIONS[6]]).run()
        self.assertFalse(submit().disabled)

    def test_training_check_disabled_until_answered(self):
        at = login(new_app(), "alice", "pin-alice")
        t = TRAINING[0]
        self.assertTrue(at.button(key=f"check_btn_{t['id']}").disabled)
        answer(at, f"train_{t['id']}", expected(t))
        self.assertFalse(at.button(key=f"check_btn_{t['id']}").disabled)


class TestInstructions(AppCase):
    def test_kitchen_rule_in_training_and_live_pages(self):
        line = core.load_instructions(ROOT / "training_tasks.json")[0]
        at = login(new_app(), "alice", "pin-alice")
        self.assertTrue(any(line in m.value for m in at.markdown))       # training intro
        do_training(self, at)
        self.assertTrue(any(line in m.value for m in at.markdown))       # live item page


class TestTrainingFeedback(AppCase):
    def test_expected_answers_match_for_every_item(self):
        do_training(self, login(new_app(), "alice", "pin-alice"))
        marker = [r for r in self.rows("alice") if r[1] == core.TRAINING_MARKER]
        self.assertEqual(len(marker), 1)
        self.assertEqual(marker[0][11:13], [core.INTERFACE_VERSION, TRAINING_VERSION])

    def test_wrong_answers_differ_for_every_item(self):
        do_training(self, login(new_app(), "alice", "pin-alice"), wrong=True)


class TestTwoAnnotators(AppCase):
    def test_simultaneous_sessions_do_not_interfere(self):
        v1_row = ["alice", "SaGC", "5", "0", "goal", "ws", core.CLARITY_OPTIONS[0], "N/A",
                  core.FEASIBILITY_OPTIONS[0], core.SAFETY_OPTIONS[0], "v1 comment"]
        self.sheet.values.append(list(v1_row))
        a = do_training(self, login(new_app(), "alice", "pin-alice"))
        b = do_training(self, login(new_app(), "bob", "pin-bob"))
        self.assertEqual(self.sheet.values[0][11:13], core.HEADER_V2)  # L/M headers added
        done_a, done_b = [], []
        for _ in range(3):
            done_a.append(submit_current(a))
            done_b.append(submit_current(b))
        # bob also labels alice's first item: two rows for one item, none overwritten
        flat_b, _ = current(b)
        b.session_state["pos"] = flat_b.index(done_a[0])
        b.run()
        submit_current(b)
        self.assertEqual(self.sheet.values[1], v1_row)  # v1 answer untouched
        rows_a = [r[2] for r in self.rows("alice") if r[11] == "v2" and r[1] != core.TRAINING_MARKER]
        rows_b = [r[2] for r in self.rows("bob") if r[11] == "v2" and r[1] != core.TRAINING_MARKER]
        self.assertEqual(rows_a, done_a)
        self.assertEqual(sorted(rows_b), sorted(done_b + [done_a[0]]))
        keys = [core.row_key(r) for r in self.rows()]
        self.assertEqual(len(keys), len(set(keys)))
        # the same block, a different order for each annotator, each logged once
        alloc = self.book.sheets[core.ALLOC_SHEET].values
        self.assertEqual(alloc[0], core.ALLOC_HEADER)
        logged = {(r[1], r[4]): json.loads(r[6]) for r in alloc[1:]}
        self.assertEqual(len(logged), len(alloc) - 1)
        self.assertEqual(set(logged), {("alice", "1"), ("bob", "1")})
        self.assertEqual(sorted(logged[("alice", "1")]), sorted(logged[("bob", "1")]))
        self.assertNotEqual(logged[("alice", "1")], logged[("bob", "1")])
        self.assertEqual(logged[("alice", "1")][:3], done_a)  # order shown = order logged
        for r in self.rows():
            if r[11] == "v2" and r[1] != core.TRAINING_MARKER:
                self.assertEqual(r[12], TRAINING_VERSION)


class TestResume(AppCase):
    def test_stop_and_resume_in_a_new_session(self):
        a = do_training(self, login(new_app(), "alice", "pin-alice"))
        ambiguous = {"clarity": core.CLARITY_OPTIONS[3], "ambiguity_types": [core.AMBIGUITY_OPTIONS[6]],
                     "feasibility": core.FEASIBILITY_OPTIONS[2], "safety": core.SAFETY_OPTIONS[2]}
        done = [submit_current(a), submit_current(a), submit_current(a, ambiguous)]
        # a new browser session (the old one is gone)
        r = login(new_app(), "Alice", "pin-alice", register=False)
        self.assertTrue(r.session_state["is_trained"])  # training detected from the sheet
        self.assertFalse(any("Calibration Task" in m.value for m in r.markdown))
        flat, tid = current(r)
        self.assertEqual(flat[:3], done)
        self.assertEqual(r.session_state["pos"], 3)
        # going back shows the saved answers (fix D)
        r.button(key=f"prev_{tid}").click().run()
        self.assertEqual(current(r)[1], done[2])
        self.assertEqual(r.radio(key=f"clarity_{done[2]}").value, ambiguous["clarity"])
        self.assertEqual(r.multiselect(key=f"ambig_type_{done[2]}").value, ambiguous["ambiguity_types"])
        self.assertEqual(r.radio(key=f"safe_{done[2]}").value, ambiguous["safety"])
        # re-submitting an edit updates the same row and returns to the next open item
        r.radio(key=f"safe_{done[2]}").set_value(core.SAFETY_OPTIONS[0]).run()
        r.button(key=f"submit_{done[2]}").click().run()
        mine = [x for x in self.rows("alice") if x[2] == done[2]]
        self.assertEqual(len(mine), 1)
        self.assertEqual(mine[0][9], core.SAFETY_OPTIONS[0])
        self.assertEqual(current(r)[1], flat[3])
        alloc = self.book.sheets[core.ALLOC_SHEET].values[1:]
        self.assertEqual(len([x for x in alloc if x[1] == "alice" and x[4] == "1"]), 1)

    def test_new_training_version_requires_training_again(self):
        self.sheet.values.append(["alice", core.TRAINING_MARKER, "PASSED"] + ["N/A"] * 8 + ["v2", "t1"])
        at = login(new_app(), "alice", "pin-alice")
        self.assertFalse(at.session_state["is_trained"])


class TestRowCheck(unittest.TestCase):
    """Fix C: an update goes to the row that was read, or not at all."""

    def test_row_moved_between_read_and_write(self):
        import app_user
        book = fake_sheet.install(URL, [V1_HEADER + core.HEADER_V2])
        row = ["alice", "SaGC", "7", "0", "g", "w", "c", "N/A", "f", "s", "", "v2", "t2"]
        book.sheet1.values.append(list(row))
        other = ["bob", "SaGC", "9", "0", "g", "w", "c", "N/A", "f", "s", "", "v2", "t2"]
        original_read = book.sheet1.get_all_values

        def stale_read(*a, **k):
            values = original_read()
            book.sheet1.values[1] = list(other)  # someone sorted the sheet after our read
            return values
        book.sheet1.get_all_values = stale_read
        with self.assertRaises(core.SheetError):
            app_user.write_row(book, row[:10] + ["edited", "v2", "t2"])
        self.assertEqual(book.sheet1.values[1], other)


if __name__ == "__main__":
    unittest.main()
