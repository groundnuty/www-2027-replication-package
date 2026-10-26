"""Reading and rewriting gold SPARQL for the fault bank, at the token level.

Why token level: a mutation must change one term (or one aggregate keyword, or one GROUP BY key)
and leave every other byte of the gold query as it was, so a mutated query differs from its gold
only by the fault. Re-serialising through rdflib's algebra would reshape subqueries, paths, HAVING
and prefixes. rdflib is used only as the parse gate (``parse_error``); the store is the second gate.

The reader is a small recursive-descent pass over the tokens of ``harness.sparql``'s scanner. It
covers the SPARQL 1.1 query forms the question generators emit (SELECT and ASK, subqueries,
GRAPH, OPTIONAL, MINUS, UNION, FILTER with EXISTS, BIND, VALUES, property paths, aggregates and
solution modifiers). It records, for every IRI token, its role:

    predicate   the verb of a triple pattern, or a step of a property path
    membership  rdf:type (``a``) or rdfs:subClassOf as a verb or path step; never mutated
    class       the object of a membership step that the caller's class test accepts (the fault
                bank's test reads only source-neutral facts: ``operators.Context.is_class``)
    constant    any other IRI in a subject or object position, in VALUES, or compared in an
                expression (rule 5); data nodes in a class position are constants (rule 32)

Scopes follow SPARQL's: each UNION branch has its own id (``Scope.branch``, the chain of enclosing
branches), each negated part its own id (rule 17), and a subquery's variables are its own unless it
projects them. A variable a subquery does not project gets the key ``?name__sq<n>`` (n is the
subquery's SELECT index; underscores are added until no written variable has the key), so two
patterns share a variable key exactly when they share the variable.

IRIs that are graph names, dataset clauses, datatypes, function names or in the XSD namespace are
not slots.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from itertools import pairwise

from harness.sparql import _TOKEN

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_SUBCLASSOF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
MEMBERSHIP = frozenset({RDF_TYPE, RDFS_SUBCLASSOF})
XSD = "http://www.w3.org/2001/XMLSchema#"
AGGREGATES = frozenset({"COUNT", "MIN", "MAX", "SUM", "AVG", "SAMPLE", "GROUP_CONCAT"})
_GROUP_KEYWORDS = frozenset({"OPTIONAL", "MINUS", "GRAPH", "SERVICE", "FILTER", "BIND", "VALUES", "UNION"})
_MODIFIER_KEYWORDS = frozenset({"GROUP", "HAVING", "ORDER", "LIMIT", "OFFSET", "VALUES"})
_PN_LOCAL = re.compile(r"^[A-Za-z0-9_](?:[A-Za-z0-9_\-.]*[A-Za-z0-9_\-])?$")


@dataclass(frozen=True)
class Token:
    kind: str  # space, comment, string, iri, var, lang, pname, bnode, word, number, other
    text: str
    start: int
    end: int


def tokenize(text: str) -> list[Token]:
    """Every character of ``text`` in exactly one token. Corrections to the shared scanner: a
    prefixed name or number never ends with '.' (PN_LOCAL and DECIMAL cannot), a lone '?' is a
    path modifier, and '_:x' is a blank node label."""
    out: list[Token] = []
    for m in _TOKEN.finditer(text):
        kind, s, start = m.lastgroup or "other", m.group(), m.start()
        if kind in ("pname", "number") and s.endswith(".") and s.rstrip("."):
            core = s.rstrip(".")
            out.append(Token(kind, core, start, start + len(core)))
            for i in range(len(core), len(s)):
                out.append(Token("other", ".", start + i, start + i + 1))
            continue
        if kind == "var" and len(s) == 1:
            kind = "other"
        elif kind == "pname" and s.startswith("_:"):
            kind = "bnode"
        out.append(Token(kind, s, start, m.end()))
    return out


@dataclass(frozen=True)
class Scope:
    negated: bool = False  # inside MINUS or NOT EXISTS
    optional: bool = False
    union: bool = False
    exists: bool = False  # inside a positive EXISTS
    subquery: int = 0  # depth of nested SELECTs
    graph: str | None = None
    negation: int | None = None  # id of the innermost MINUS or NOT EXISTS part (rule 17), None outside any
    branch: tuple[tuple[int, int], ...] = ()  # (UNION id, branch index) of every enclosing UNION branch, outermost first
    select: int | None = None  # index into ParsedQuery.selects of the innermost SELECT; None at an ASK's top level


@dataclass(frozen=True)
class Pattern:
    subject: str  # "?v", an IRI, "lit:<text>", or a fresh node "_:path<n>" / "_:bnode<n>"
    predicate: str  # an IRI or "?v"
    object: str
    modifier: str  # "", "+", "*" or "?" on this step
    scope: Scope
    in_path: bool  # a step of a sequence path
    complex_path: bool  # part of an alternative, negated set or nested path: approximate
    subject_token: int | None
    predicate_token: int | None
    object_token: int | None


@dataclass(frozen=True)
class Slot:
    token: int
    start: int
    end: int
    text: str
    iri: str
    role: str  # predicate, membership, class, constant
    context: str  # triple, path, values, filter, bind, having, order, select, group
    pattern: int | None  # index into ParsedQuery.patterns
    position: str | None  # subject, predicate, object
    scope: Scope
    variable: str | None = None  # the VALUES variable a constant binds


@dataclass(frozen=True)
class Aggregate:
    token: int  # the function name
    func: str
    open_paren: int
    distinct: int | None  # the DISTINCT token inside the call
    select: int  # index into ParsedQuery.selects


@dataclass(frozen=True)
class GroupKey:
    token: int  # first token of the key
    last: int  # last token of the key
    variable: str | None


@dataclass(frozen=True)
class OrderItem:
    first: int
    last: int
    variables: tuple[str, ...]


@dataclass
class Select:
    token: int
    parent: int | None = None  # the enclosing SELECT's index
    nested: bool = False  # a subquery
    end: int | None = None  # its last token (after its solution modifiers)
    where: int | None = None  # the '{' token that opens its WHERE group
    projection: list[int] = field(default_factory=list)  # tokens of plain projected variables
    projection_all: bool = False
    group_by: int | None = None  # the GROUP token
    group_keys: list[GroupKey] = field(default_factory=list)
    order_by: int | None = None  # the ORDER token
    order_items: list[OrderItem] = field(default_factory=list)


@dataclass
class ParsedQuery:
    text: str
    tokens: list[Token]
    prefixes: dict[str, str]
    body_start: int  # first token after the prologue
    patterns: list[Pattern] = field(default_factory=list)
    slots: list[Slot] = field(default_factory=list)
    aggregates: list[Aggregate] = field(default_factory=list)
    selects: list[Select] = field(default_factory=list)
    non_slot_tokens: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class Edit:
    start: int
    end: int
    text: str


def apply_edits(text: str, edits: list[Edit]) -> str:
    """Apply non-overlapping character edits."""
    ordered = sorted(edits, key=lambda e: (e.start, e.end))
    for a, b in pairwise(ordered):
        if b.start < a.end:
            raise ValueError(f"overlapping edits {a} and {b}")
    for e in reversed(ordered):
        text = text[: e.start] + e.text + text[e.end :]
    return text


def parse_error(query: str) -> str | None:
    """None when rdflib parses and translates the query, else the error text."""
    from harness.sparql_grammar import prepare_query

    try:
        prepare_query(query)
    except Exception as exc:  # noqa: BLE001 - any parser failure is a rejection reason
        return f"{type(exc).__name__}: {exc}"[:500]
    return None


def iri_text(q: ParsedQuery, iri: str) -> str:
    """The IRI as a prefixed name when a declared prefix covers it with a valid local part."""
    best = None
    for prefix, ns in q.prefixes.items():
        if iri.startswith(ns) and _PN_LOCAL.match(iri[len(ns):]) and (best is None or len(ns) > len(q.prefixes[best])):
            best = prefix
    return f"{best}:{iri[len(q.prefixes[best]):]}" if best is not None else f"<{iri}>"


class ParseFailure(ValueError):
    pass


class _Reader:
    def __init__(self, text: str, is_class: Callable[[str], bool]):
        self.q = ParsedQuery(text, tokenize(text), {}, 0)
        self.sig = [i for i, t in enumerate(self.q.tokens) if t.kind not in ("space", "comment")]
        self.pos = 0
        self.is_class = is_class
        self.fresh_path = 0
        self.fresh_bnode = 0
        self.negations = 0
        self.unions = 0
        self.select_stack: list[int] = []
        self.written = {"?" + t.text[1:] for t in self.q.tokens if t.kind == "var"}
        self.renamed: dict[tuple[int, str], str] = {}
        # (token, role-kind, context, pattern, position, scope, variable); roles resolved at the end
        self.pending: list[tuple[int, str, str, int | None, str | None, Scope, str | None]] = []

    def negate(self, scope: Scope) -> Scope:
        """A new negated part: its own id, so its patterns are read on their own (rule 17)."""
        self.negations += 1
        return replace(scope, negated=True, negation=self.negations)

    def projects(self, sid: int, name: str) -> bool:
        sel = self.q.selects[sid]
        return sel.projection_all or any("?" + self.q.tokens[i].text[1:] == name for i in sel.projection)

    def var_key(self, name: str, scope: Scope) -> str:
        """The variable's key: its own name where it is the top level's variable, else a key of the
        outermost subquery it reaches through projections (a subquery's variables are its own
        unless projected, SPARQL 1.1 section 18.2.1)."""
        owner = scope.select
        while owner is not None and self.q.selects[owner].nested and self.projects(owner, name):
            owner = self.q.selects[owner].parent
        if owner is None or not self.q.selects[owner].nested:
            return name
        key = self.renamed.get((owner, name))
        if key is None:
            key = f"{name}__sq{owner}"
            while key in self.written or key in self.renamed.values():
                key += "_"
            self.renamed[(owner, name)] = key
        return key

    # -- token access -------------------------------------------------------------------------
    def peek(self, k: int = 0) -> Token | None:
        j = self.pos + k
        return self.q.tokens[self.sig[j]] if j < len(self.sig) else None

    def index(self, k: int = 0) -> int:
        return self.sig[self.pos + k]

    def next(self) -> int:
        if self.pos >= len(self.sig):
            raise ParseFailure("unexpected end of query")
        i = self.sig[self.pos]
        self.pos += 1
        return i

    def is_(self, text: str, k: int = 0) -> bool:
        t = self.peek(k)
        return t is not None and t.kind == "other" and t.text == text

    def word(self, k: int = 0) -> str | None:
        t = self.peek(k)
        return t.text.upper() if t is not None and t.kind == "word" else None

    def expect(self, text: str) -> int:
        if not self.is_(text):
            t = self.peek()
            raise ParseFailure(f"expected {text!r}, found {t.text if t else 'end'!r}")
        return self.next()

    def expand(self, i: int) -> str:
        t = self.q.tokens[i]
        if t.kind == "iri":
            return t.text[1:-1]
        prefix, _, local = t.text.partition(":")
        if prefix not in self.q.prefixes:
            raise ParseFailure(f"undeclared prefix {prefix!r}")
        return self.q.prefixes[prefix] + local.replace("\\", "")

    # -- query --------------------------------------------------------------------------------
    def read(self) -> ParsedQuery:
        while self.word() in ("PREFIX", "BASE"):
            if self.word() == "PREFIX":
                self.next()
                name = self.q.tokens[self.next()].text
                iri = self.q.tokens[self.next()].text
                self.q.prefixes[name.rstrip(":")] = iri[1:-1]
            else:
                self.next()
                self.next()
        self.q.body_start = self.index() if self.pos < len(self.sig) else len(self.q.tokens)
        form = self.word()
        root = Scope()
        if form == "SELECT":
            root = self.select(root, nested=False)
        elif form == "ASK":
            self.next()
            self.dataset()
            if self.word() == "WHERE":
                self.next()
            self.group(root)
            self.modifiers(None, root)
        else:
            raise ParseFailure(f"unsupported query form {form!r}")
        if self.word() == "VALUES":
            self.next()
            self.values(root)
        if self.pos != len(self.sig):
            raise ParseFailure(f"trailing tokens from {self.peek().text!r}")
        self.resolve()
        return self.q

    def dataset(self) -> None:
        while self.word() == "FROM":
            self.next()
            if self.word() == "NAMED":
                self.next()
            self.q.non_slot_tokens.append(self.next())

    def select(self, scope: Scope, nested: bool) -> Scope:
        """A SELECT and its body; returns the body's scope."""
        sel = Select(self.next(), parent=self.select_stack[-1] if self.select_stack else None, nested=nested)
        self.q.selects.append(sel)
        sid = len(self.q.selects) - 1
        self.select_stack.append(sid)
        scope = replace(scope, select=sid)
        if self.word() in ("DISTINCT", "REDUCED"):
            self.next()
        while True:
            t = self.peek()
            if t is None:
                raise ParseFailure("unterminated SELECT")
            if t.kind == "var":
                sel.projection.append(self.next())
            elif self.is_("*"):
                self.next()
                sel.projection_all = True
            elif self.is_("("):
                self.expression(scope, "select")
            else:
                break
        self.dataset()
        if self.word() == "WHERE":
            self.next()
        sel.where = self.index()
        self.group(scope)
        self.modifiers(sel, scope)
        sel.end = self.sig[self.pos - 1]
        self.select_stack.pop()
        return scope

    def modifiers(self, sel: Select | None, scope: Scope) -> None:
        if self.word() == "GROUP":
            gtok = self.next()
            if self.word() != "BY":
                raise ParseFailure("GROUP without BY")
            self.next()
            if sel is not None:
                sel.group_by = gtok
            while True:
                t = self.peek()
                if t is None or self.word() in _MODIFIER_KEYWORDS or self.is_("}"):
                    break
                first = self.index()
                variable = None
                if t.kind == "var":
                    variable = "?" + t.text[1:]
                    self.next()
                elif self.is_("("):
                    self.expression(scope, "group")
                elif t.kind in ("word", "iri", "pname"):
                    self.call(scope, "group")
                else:
                    break
                if sel is not None:
                    sel.group_keys.append(GroupKey(first, self.sig[self.pos - 1], variable))
        if self.word() == "HAVING":
            self.next()
            while self.is_("(") or (self.peek() is not None and self.peek().kind == "word"
                                    and self.word() not in _MODIFIER_KEYWORDS):
                if self.is_("("):
                    self.expression(scope, "having")
                else:
                    self.call(scope, "having")
        if self.word() == "ORDER":
            otok = self.next()
            if self.word() != "BY":
                raise ParseFailure("ORDER without BY")
            self.next()
            if sel is not None:
                sel.order_by = otok
            while True:
                t = self.peek()
                if t is None or self.is_("}") or self.word() in ("LIMIT", "OFFSET", "VALUES"):
                    break
                first = self.index()
                before = self.pos
                if self.word() in ("ASC", "DESC"):
                    self.next()
                    self.expression(scope, "order")
                elif t.kind == "var":
                    self.next()
                elif self.is_("("):
                    self.expression(scope, "order")
                elif t.kind in ("word", "iri", "pname"):
                    self.call(scope, "order")
                else:
                    break
                last = self.sig[self.pos - 1]
                names = tuple("?" + self.q.tokens[self.sig[j]].text[1:] for j in range(before, self.pos)
                              if self.q.tokens[self.sig[j]].kind == "var")
                if sel is not None:
                    sel.order_items.append(OrderItem(first, last, names))
        for _ in range(2):
            if self.word() in ("LIMIT", "OFFSET"):
                self.next()
                self.next()

    # -- group graph patterns -------------------------------------------------------------------
    def matching_brace(self) -> int:
        """Position (in self.sig) just after the group that starts here."""
        depth, j = 0, self.pos
        while j < len(self.sig):
            t = self.q.tokens[self.sig[j]]
            if t.kind == "other" and t.text == "{":
                depth += 1
            elif t.kind == "other" and t.text == "}":
                depth -= 1
                if depth == 0:
                    return j + 1
            j += 1
        raise ParseFailure("unbalanced braces")

    def group(self, scope: Scope) -> None:
        self.expect("{")
        if self.word() == "SELECT":
            inner = self.select(replace(scope, subquery=scope.subquery + 1), nested=True)
            if self.word() == "VALUES":  # the subquery's own VALUES clause
                self.next()
                self.values(inner)
            self.expect("}")
            return
        while not self.is_("}"):
            w = self.word()
            if self.peek() is None:
                raise ParseFailure("unterminated group")
            if w == "OPTIONAL":
                self.next()
                self.group(replace(scope, optional=True))
            elif w == "MINUS":
                self.next()
                self.group(self.negate(scope))
            elif w == "GRAPH":
                self.next()
                t = self.next()
                tok = self.q.tokens[t]
                graph = None
                if tok.kind in ("iri", "pname"):
                    self.q.non_slot_tokens.append(t)
                    graph = self.expand(t)
                self.group(replace(scope, graph=graph))
            elif w == "SERVICE":
                self.next()
                if self.word() == "SILENT":
                    self.next()
                self.q.non_slot_tokens.append(self.next())
                self.group(scope)
            elif w == "FILTER":
                self.next()
                self.constraint(scope, "filter")
            elif w == "BIND":
                self.next()
                self.expression(scope, "bind")
            elif w == "VALUES":
                self.next()
                self.values(scope)
            elif self.is_("{"):
                end = self.matching_brace()
                if end < len(self.sig) and self.q.tokens[self.sig[end]].text.upper() == "UNION":
                    self.unions += 1
                    u, b = self.unions, 0
                    self.group(replace(scope, union=True, branch=(*scope.branch, (u, b))))
                    while self.word() == "UNION":
                        self.next()
                        b += 1
                        self.group(replace(scope, union=True, branch=(*scope.branch, (u, b))))
                else:
                    self.group(scope)
            elif self.is_("."):
                self.next()
            else:
                self.triples(scope)
        self.expect("}")

    def values(self, scope: Scope) -> None:
        names: list[str] = []
        if self.is_("("):
            self.next()
            while not self.is_(")"):
                names.append(self.var_key("?" + self.q.tokens[self.next()].text[1:], scope))
            self.next()
        else:
            names.append(self.var_key("?" + self.q.tokens[self.next()].text[1:], scope))
        self.expect("{")
        column = 0
        while not self.is_("}"):
            if self.is_("("):
                self.next()
                column = 0
                continue
            if self.is_(")"):
                self.next()
                continue
            i = self.next()
            t = self.q.tokens[i]
            if t.kind in ("iri", "pname"):
                if self._prev_is_datatype_marker():
                    self.q.non_slot_tokens.append(i)
                    continue
                var = names[column] if column < len(names) else None
                self.pending.append((i, "constant", "values", None, None, scope, var))
            if not (t.kind == "other" and t.text == "^"):
                column = column + 1 if len(names) > 1 else 0
        self.expect("}")

    def _prev_is_datatype_marker(self) -> bool:
        a, b = self.pos - 2, self.pos - 3
        return b >= 0 and self.q.tokens[self.sig[a]].text == "^" and self.q.tokens[self.sig[b]].text == "^"

    # -- expressions ----------------------------------------------------------------------------
    def constraint(self, scope: Scope, context: str) -> None:
        if self.word() == "NOT" and self.word(1) == "EXISTS":
            self.next()
            self.next()
            self.group(self.negate(scope))
        elif self.word() == "EXISTS":
            self.next()
            self.group(replace(scope, exists=True))
        elif self.is_("("):
            self.expression(scope, context)
        else:
            self.call(scope, context)

    def call(self, scope: Scope, context: str) -> None:
        i = self.next()
        if self.q.tokens[i].kind in ("iri", "pname"):
            self.q.non_slot_tokens.append(i)
        self.aggregate_site(i)
        if self.is_("("):
            self.expression(scope, context)

    def aggregate_site(self, i: int) -> None:
        t = self.q.tokens[i]
        if t.kind == "word" and t.text.upper() in AGGREGATES and self.is_("(") and self.select_stack:
            distinct = self.index(1) if self.word(1) == "DISTINCT" else None
            self.q.aggregates.append(Aggregate(i, t.text.upper(), self.index(), distinct, self.select_stack[-1]))

    def expression(self, scope: Scope, context: str) -> None:
        """A balanced parenthesised expression starting at '('."""
        self.expect("(")
        depth = 1
        while depth:
            t = self.peek()
            if t is None:
                raise ParseFailure("unbalanced parentheses")
            if self.word() == "NOT" and self.word(1) == "EXISTS":
                self.next()
                self.next()
                self.group(self.negate(scope))
                continue
            if self.word() == "EXISTS":
                self.next()
                self.group(replace(scope, exists=True))
                continue
            i = self.next()
            if t.kind == "other" and t.text == "(":
                depth += 1
            elif t.kind == "other" and t.text == ")":
                depth -= 1
            elif t.kind == "word":
                self.aggregate_site(i)
            elif t.kind in ("iri", "pname"):
                if self.is_("(") or self._prev_is_datatype_marker() or self.expand(i).startswith(XSD):
                    self.q.non_slot_tokens.append(i)
                else:
                    self.pending.append((i, "constant", context, None, None, scope, None))

    # -- triples --------------------------------------------------------------------------------
    def triples(self, scope: Scope) -> None:
        subject, stok = self.node(scope)
        if subject.startswith("_:bnode") and (self.is_(".") or self.is_("}")):
            return
        self.property_list(subject, stok, scope)

    def node(self, scope: Scope) -> tuple[str, int | None]:
        """A subject or object: returns its key and its token (None for fresh nodes)."""
        t = self.peek()
        if t is None:
            raise ParseFailure("expected a term")
        if t.kind == "var":
            i = self.next()
            return self.var_key("?" + t.text[1:], scope), i
        if t.kind in ("iri", "pname"):
            i = self.next()
            return self.expand(i), i
        if t.kind == "bnode":
            i = self.next()
            return t.text, i
        if t.kind == "string":
            i = self.next()
            text = "lit:" + t.text
            if self.peek() is not None and self.peek().kind == "lang":
                text += self.q.tokens[self.next()].text
            elif self.is_("^") and self.is_("^", 1):
                self.next()
                self.next()
                d = self.next()
                self.q.non_slot_tokens.append(d)
                text += "^^" + self.expand(d)
            return text, i
        if t.kind == "number" or (t.kind == "other" and t.text in "+-" and self.peek(1) is not None
                                  and self.peek(1).kind == "number"):
            i = self.next()
            text = t.text
            if t.kind == "other":
                text += self.q.tokens[self.next()].text
            return "lit:" + text, i
        if t.kind == "word" and t.text.lower() in ("true", "false"):
            i = self.next()
            return "lit:" + t.text.lower(), i
        if self.is_("["):
            self.next()
            key = f"_:bnode{self.fresh_bnode}"
            self.fresh_bnode += 1
            if not self.is_("]"):
                self.property_list(key, None, scope)
            self.expect("]")
            return key, None
        if self.is_("("):  # an RDF collection: members only, no list structure
            self.next()
            key = f"_:bnode{self.fresh_bnode}"
            self.fresh_bnode += 1
            while not self.is_(")"):
                self.node(scope)
            self.next()
            return key, None
        raise ParseFailure(f"unexpected token {t.text!r}")

    def property_list(self, subject: str, stok: int | None, scope: Scope) -> None:
        while True:
            t = self.peek()
            if t is None or self.is_(".") or self.is_("}") or self.is_("]") or self.is_("{") \
                    or self.word() in _GROUP_KEYWORDS:
                return
            verb = self.path(scope)
            while True:
                obj, otok = self.node(scope)
                self.emit(subject, stok, verb, obj, otok, scope)
                if self.is_(","):
                    self.next()
                    continue
                break
            if self.is_(";"):
                while self.is_(";"):
                    self.next()
                continue
            return

    # A path is a list of alternatives; each alternative a list of steps
    # (iri-or-var, token, inverse, modifier, complex).
    def path(self, scope: Scope) -> list[list[tuple[str, int | None, bool, str, bool]]]:
        t = self.peek()
        if t is not None and t.kind == "var":
            i = self.next()
            return [[(self.var_key("?" + t.text[1:], scope), i, False, "", False)]]
        alternatives = [self.sequence(scope)]
        while self.is_("|"):
            self.next()
            alternatives.append(self.sequence(scope))
        if len(alternatives) > 1:
            alternatives = [[(p, i, inv, mod, True) for p, i, inv, mod, _ in alt] for alt in alternatives]
        return alternatives

    def sequence(self, scope: Scope) -> list[tuple[str, int | None, bool, str, bool]]:
        steps = self.element(scope)
        while self.is_("/"):
            self.next()
            steps += self.element(scope)
        return steps

    def element(self, scope: Scope) -> list[tuple[str, int | None, bool, str, bool]]:
        inverse = False
        if self.is_("^"):
            self.next()
            inverse = True
        t = self.peek()
        if t is None:
            raise ParseFailure("expected a path")
        if t.kind == "word" and t.text == "a":
            steps = [(RDF_TYPE, self.next(), inverse, "", False)]
        elif t.kind in ("iri", "pname"):
            i = self.next()
            steps = [(self.expand(i), i, inverse, "", False)]
        elif self.is_("!"):
            self.next()
            steps = []
            if self.is_("("):
                self.next()
                while not self.is_(")"):
                    i = self.next()
                    if self.q.tokens[i].kind in ("iri", "pname"):
                        steps.append((self.expand(i), i, inverse, "", True))
                self.next()
            else:
                if self.is_("^"):
                    self.next()
                i = self.next()
                steps.append((RDF_TYPE if self.q.tokens[i].text == "a" else self.expand(i), i, inverse, "", True))
        elif self.is_("("):
            self.next()
            inner = self.path(scope)
            self.expect(")")
            steps = [(p, i, inv != inverse, mod, cx or len(inner) > 1 or inverse)
                     for alt in inner for p, i, inv, mod, cx in alt]
            if len(inner) == 1 and len(inner[0]) > 1:
                steps = [(p, i, inv, mod, True) for p, i, inv, mod, _ in steps]
        else:
            raise ParseFailure(f"unexpected path token {t.text!r}")
        if self.peek() is not None and self.peek().kind == "other" and self.peek().text in "+*?" \
                and self.peek().text:
            mod = self.q.tokens[self.next()].text
            if len(steps) == 1:
                p, i, inv, _, cx = steps[0]
                steps = [(p, i, inv, mod, cx)]
            else:
                steps = [(p, i, inv, mod, True) for p, i, inv, _, _ in steps]
        return steps

    def emit(self, subject: str, stok: int | None, verb, obj: str, otok: int | None, scope: Scope) -> None:
        if len(verb) == 1 and not any(step[4] for step in verb[0]):
            steps = verb[0]
            nodes = [(subject, stok)]
            for _ in steps[:-1]:
                nodes.append((f"_:path{self.fresh_path}", None))
                self.fresh_path += 1
            nodes.append((obj, otok))
            in_path = len(steps) > 1
            for k, (pred, ptok, inverse, mod, _) in enumerate(steps):
                (s, st), (o, ot) = nodes[k], nodes[k + 1]
                if inverse:
                    (s, st), (o, ot) = (o, ot), (s, st)
                self.add_pattern(Pattern(s, pred, o, mod, scope, in_path, False, st, ptok, ot))
        else:  # alternatives, negated sets, nested paths: one approximate pattern per step
            for alt in verb:
                for pred, ptok, inverse, mod, _ in alt:
                    s, st, o, ot = (obj, otok, subject, stok) if inverse else (subject, stok, obj, otok)
                    self.add_pattern(Pattern(s, pred, o, mod, scope, True, True, st, ptok, ot))

    def add_pattern(self, p: Pattern) -> None:
        self.q.patterns.append(p)
        k = len(self.q.patterns) - 1
        context = "path" if p.in_path else "triple"
        seen = {item[0] for item in self.pending}
        if p.subject_token is not None and p.subject_token not in seen \
                and self.q.tokens[p.subject_token].kind in ("iri", "pname"):
            self.pending.append((p.subject_token, "constant", context, k, "subject", p.scope, None))
        if p.predicate_token is not None and not p.predicate.startswith("?"):
            kind = "membership" if p.predicate in MEMBERSHIP else "predicate"
            self.pending.append((p.predicate_token, kind, context, k, "predicate", p.scope, None))
        if p.object_token is not None and p.object_token not in seen \
                and self.q.tokens[p.object_token].kind in ("iri", "pname"):
            kind = "class?" if p.predicate in MEMBERSHIP else "constant"
            self.pending.append((p.object_token, kind, context, k, "object", p.scope, None))

    def resolve(self) -> None:
        for i, kind, context, k, position, scope, variable in sorted(self.pending, key=lambda x: x[0]):
            iri = RDF_TYPE if self.q.tokens[i].text == "a" else self.expand(i)
            if kind == "class?":
                kind = "class" if self.is_class(iri) else "constant"
            if kind == "constant" and iri.startswith(XSD):
                self.q.non_slot_tokens.append(i)
                continue
            t = self.q.tokens[i]
            self.q.slots.append(Slot(i, t.start, t.end, t.text, iri, kind, context, k, position, scope, variable))


def parse(text: str, is_class: Callable[[str], bool] | frozenset[str] = frozenset()) -> ParsedQuery:
    """Read a query; ``is_class`` (a test, or a set of classes) decides class against constant in
    membership objects."""
    return _Reader(text, is_class if callable(is_class) else is_class.__contains__).read()
