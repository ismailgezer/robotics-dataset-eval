"""The repository (public) must never hold keys, the Sheet URL or labels.

Checks the tracked files and .gitignore, and that the app writes no local files: answers go
only to the Google Sheet, credentials and access codes come only from st.secrets.
Run from the repository root:  python -m unittest discover -s tests -v
"""
import ast
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


IS_GIT = git("rev-parse", "--git-dir").returncode == 0

# Patterns are assembled so that this file does not match itself.
SECRET_PATTERNS = {
    "private key": "-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "service-account JSON": r'"type"\s*:\s*"service' + '_account"',
    "private_key_id": r'"private' + r'_key_id"\s*:\s*"[0-9a-f]{20,}"',
    "Google API key": "AI" + r"za[0-9A-Za-z_\-]{35}",
    "Sheet URL": r"docs\.google\.com/" + "spreadsheets/d/",
    "access-code table": r"(?m)^\s*\[" + r"annotators\]",
}
DATA_OK = {"ambik_test_900.csv"}  # dataset file, not labels


@unittest.skipUnless(IS_GIT, "not a git checkout")
class TrackedFiles(unittest.TestCase):
    def test_no_secrets_or_sheet_url(self):
        hits = []
        for name in git("ls-files").stdout.split():
            text = (ROOT / name).read_bytes().decode("utf-8", "replace")
            hits += [(name, what) for what, pat in SECRET_PATTERNS.items() if re.search(pat, text)]
        self.assertEqual(hits, [])

    def test_no_label_exports_or_secret_files(self):
        bad = [n for n in git("ls-files").stdout.split()
               if n not in DATA_OK and (re.search(r"\.(xlsx|xls|ods|tsv|csv|pem|key|p12)$", n)
                                         or re.search(r"(service_account|credentials|secrets|labels)", n)
                                         or n.startswith(".env"))]
        self.assertEqual(bad, [])

    def test_gitignore(self):
        must_ignore = [".streamlit/secrets.toml", "secrets.toml", "service_account.json",
                       "my-service_account-key.json", "credentials.json", ".env", ".env.local", "key.pem",
                       "Sheet1.xlsx", "export.csv", "sub/export.csv", "exports/anything.json",
                       "labels_human.jsonl", "labels_human.jsonl.report.json"]
        for path in must_ignore:
            self.assertEqual(git("check-ignore", "-q", "--no-index", path).returncode, 0, path)
        for path in ["ambik_test_900.csv", "eval_subset.json", "training_tasks.json", "app_user.py",
                     ".streamlit/config.toml"]:
            self.assertNotEqual(git("check-ignore", "-q", "--no-index", path).returncode, 0, path)


class AppWritesNoFiles(unittest.TestCase):
    WRITERS = {"to_csv", "to_excel", "to_json", "to_parquet", "write_text", "write_bytes", "dump"}

    def test_no_local_writes(self):
        for name in ("app_user.py", "rde_core.py"):
            tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                if isinstance(f, ast.Attribute) and f.attr in self.WRITERS:
                    self.fail(f"{name}:{node.lineno} calls {f.attr}")
                if isinstance(f, ast.Name) and f.id == "open":
                    modes = [a.value for a in node.args[1:2] if isinstance(a, ast.Constant)]
                    modes += [k.value.value for k in node.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)]
                    self.assertFalse(any(set(m) & set("wax+") for m in modes), f"{name}:{node.lineno}")

    def test_secrets_only_from_st_secrets(self):
        app = (ROOT / "app_user.py").read_text(encoding="utf-8")
        self.assertNotIn("os.environ", app)
        self.assertEqual(set(re.findall(r'secret\("(\w+)"\)', app)), {"GOOGLE_CREDENTIALS", "SHEET_URL"})
        self.assertNotIn("st.secrets[", app.replace('st.secrets[name]', ''))   # only through secret()

    def test_no_sign_in_secrets_or_registrations(self):
        # Sign-in is an ID only: no study code, no PINs, no registrations worksheet.
        for name in ("app_user.py", "rde_core.py"):
            text = (ROOT / name).read_text(encoding="utf-8")
            for word in ("STUDY" + "_CODE", "registrations", "pbkdf2", "PIN"):
                self.assertNotIn(word, text, (name, word))

if __name__ == "__main__":
    unittest.main()
