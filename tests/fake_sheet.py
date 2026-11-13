"""In-memory stand-in for the parts of gspread the app uses. Never touches the network."""
import copy
import re

import gspread

SPREADSHEETS = {}  # url -> FakeSpreadsheet


class FakeAPIError(Exception):
    pass


def _cell(ref):
    m = re.fullmatch(r"([A-Z]+)(\d+)", ref)
    col = 0
    for ch in m.group(1):
        col = col * 26 + ord(ch) - 64
    return int(m.group(2)), col


class FakeWorksheet:
    def __init__(self, book, title, values=None):
        self.book, self.title = book, title
        self.values = [list(map(str, r)) for r in (values or [])]
        self.writes = 0

    def _check(self, kind):
        if kind in self.book.fail:
            raise FakeAPIError(f"simulated {kind} failure")

    def get_all_values(self, *a, **k):
        self._check("read")
        self.book.reads += 1
        width = max((len(r) for r in self.values), default=0)
        return [list(r) + [""] * (width - len(r)) for r in copy.deepcopy(self.values)]

    def row_values(self, row, *a, **k):
        self._check("read")
        r = list(self.values[row - 1]) if row <= len(self.values) else []
        while r and r[-1] == "":
            r.pop()
        return r

    def update(self, values=None, range_name=None, raw=True, **k):
        self._check("write")
        assert raw, "the app must write RAW values"
        start = range_name.split(":")[0]
        row, col = _cell(start)
        for dr, new in enumerate(values):
            while len(self.values) < row + dr:
                self.values.append([])
            target = self.values[row + dr - 1]
            while len(target) < col - 1 + len(new):
                target.append("")
            for dc, v in enumerate(new):
                target[col - 1 + dc] = str(v)
        self.writes += 1

    def append_row(self, values, value_input_option="RAW", **k):
        self._check("write")
        assert value_input_option == "RAW"
        self.values.append([str(v) for v in values])
        self.writes += 1


class FakeSpreadsheet:
    def __init__(self, sheet1_values=None):
        self.fail = set()  # {"read", "write", "open"}
        self.reads = 0
        self.sheets = {"Sheet1": FakeWorksheet(self, "Sheet1", sheet1_values)}

    @property
    def sheet1(self):
        return self.sheets["Sheet1"]

    def worksheet(self, title):
        if title not in self.sheets:
            raise gspread.exceptions.WorksheetNotFound(title)
        return self.sheets[title]

    def add_worksheet(self, title, rows, cols, index=None):
        self.sheets[title] = FakeWorksheet(self, title)
        return self.sheets[title]


class FakeClient:
    def open_by_url(self, url):
        book = SPREADSHEETS[url]
        if "open" in book.fail:
            raise FakeAPIError("simulated open failure")
        return book


def install(url, sheet1_values=None):
    """Register a fresh fake spreadsheet and route gspread's constructors to it."""
    SPREADSHEETS.clear()
    SPREADSHEETS[url] = FakeSpreadsheet(sheet1_values)
    gspread.service_account_from_dict = lambda info, *a, **k: FakeClient()
    gspread.service_account = lambda *a, **k: FakeClient()
    return SPREADSHEETS[url]
