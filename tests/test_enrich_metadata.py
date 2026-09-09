from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "enrich_metadata.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("enrich_metadata", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves field types via sys.modules[cls.__module__], so a
    # module loaded by path has to be registered before it is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def em() -> ModuleType:
    return _load_module()


# ---------------------------------------------------------------------------
# parse_mapping
# ---------------------------------------------------------------------------


def test_parse_mapping_left_right(em: ModuleType) -> None:
    assert em.parse_mapping("canonical_composer=Composer") == ("canonical_composer", "Composer")


def test_parse_mapping_identity(em: ModuleType) -> None:
    assert em.parse_mapping("Year") == ("Year", "Year")


@pytest.mark.parametrize("spec", ["", "=", "A=", "=B", "A=B=C", "  "])
def test_parse_mapping_rejects_malformed(em: ModuleType, spec: str) -> None:
    with pytest.raises(ValueError):
        em.parse_mapping(spec)


# ---------------------------------------------------------------------------
# load_annotations
# ---------------------------------------------------------------------------

# Verbatim lines from misc/MAESTRO_comp_year.txt exercising the quoting hazards
# from the PLAN (finding 1): an unbalanced leading quote, an unbalanced embedded
# quote, and embedded balanced quotes. All must survive verbatim when quoting is
# disabled (the default).
TRICKY_ANNOTATION_BODY = (
    "Johann Sebastian Bach / Ferruccio Busoni|\"Wachet  auf, ruft uns die Stimme' BWV 645|1898\r\n"
    "Pyotr Ilyich Tchaikovsky / Mikhail Pletnev|March from \"Nutcracker Suite|1977\r\n"
    "Sergei Rachmaninoff|Polka de \"W.R.\"|1911\r\n"
    "Johann Sebastian Bach|French Suite No. 5 in G Major|1723\r\n"
)


def _write_annotations(path: Path, text: str) -> Path:
    path.write_bytes(text.encode("utf-8"))
    return path


def test_load_annotations_skips_comments_delimiter_crlf_quotes_verbatim(
    em: ModuleType, tmp_path: Path
) -> None:
    ann_path = _write_annotations(
        tmp_path / "ann.txt",
        "# AI-compiled annotation file. May contain errors.\r\n"
        "# second leading comment line\r\n"
        "Composer|Piece|Year\r\n" + TRICKY_ANNOTATION_BODY,
    )

    header, rows = em.load_annotations(ann_path, delimiter="|")

    assert header == ["Composer", "Piece", "Year"]
    assert len(rows) == 4
    assert rows[0]["Composer"] == "Johann Sebastian Bach / Ferruccio Busoni"
    assert rows[0]["Piece"] == "\"Wachet  auf, ruft uns die Stimme' BWV 645"
    assert rows[0]["Year"] == "1898"
    assert rows[1]["Piece"] == "March from \"Nutcracker Suite"
    assert rows[1]["Year"] == "1977"
    assert rows[2]["Piece"] == "Polka de \"W.R.\""
    assert rows[3]["Piece"] == "French Suite No. 5 in G Major"


def test_load_annotations_explicit_quotechar_honours_rfc4180(
    em: ModuleType, tmp_path: Path
) -> None:
    ann_path = _write_annotations(
        tmp_path / "ann.txt",
        "Composer|Piece|Year\n"
        'Johann Sebastian Bach|"Goldberg, Variations"|1742\n',
    )

    header, rows = em.load_annotations(ann_path, delimiter="|", quotechar='"')

    assert header == ["Composer", "Piece", "Year"]
    assert rows[0]["Piece"] == "Goldberg, Variations"


def test_load_annotations_disabled_quoting_splits_quoted_delimiter(
    em: ModuleType, tmp_path: Path
) -> None:
    # QUOTE_NONE semantics: with quoting disabled a delimiter inside quotes is
    # still a delimiter. This pins the difference from the RFC4180 case above.
    ann_path = _write_annotations(
        tmp_path / "ann.txt",
        "Composer|Piece|Year\n" 'Johann Sebastian Bach|"Goldberg, Variations"|1742\n',
    )

    header, rows = em.load_annotations(ann_path, delimiter="|", quotechar=None)

    assert header == ["Composer", "Piece", "Year"]
    assert len(rows) == 1
    assert rows[0]["Piece"] == '"Goldberg, Variations"'


# ---------------------------------------------------------------------------
# build_annotation_index
# ---------------------------------------------------------------------------


def test_build_annotation_index_strips_fields_and_keys_on_tuple(
    em: ModuleType,
) -> None:
    header = ["Composer", "Piece", "Year"]
    rows = [{"Composer": "  Bach  ", "Piece": "Foo  ", "Year": "1723"}]

    index, _ = em.build_annotation_index(header, rows, ["Composer", "Piece"])

    assert list(index.keys()) == [("Bach", "Foo")]
    assert all(isinstance(key, tuple) for key in index)


def test_build_annotation_index_tuple_key_avoids_joined_string_collision(
    em: ModuleType,
) -> None:
    # paste(sep="|") would collapse these two keys into one ("A|B|C"); tuples
    # keep them distinct.
    header = ["Composer", "Piece", "Year"]
    rows = [
        {"Composer": "A", "Piece": "B|C", "Year": "1700"},
        {"Composer": "A|B", "Piece": "C", "Year": "1800"},
    ]

    index, _ = em.build_annotation_index(header, rows, ["Composer", "Piece"])

    assert len(index) == 2
    assert index[("A", "B|C")]["Year"] == "1700"
    assert index[("A|B", "C")]["Year"] == "1800"


def test_build_annotation_index_first_wins_on_identical_duplicates(
    em: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    header = ["Composer", "Piece", "Year"]
    rows = [
        {"Composer": "Bach", "Piece": "Foo", "Year": "1723"},
        {"Composer": "Bach", "Piece": "Foo", "Year": "1723"},
    ]

    index, dup_counts = em.build_annotation_index(header, rows, ["Composer", "Piece"])

    assert index[("Bach", "Foo")]["Year"] == "1723"
    assert dup_counts["identical"] == 1
    assert dup_counts["conflicting"] == 0
    assert "identical" in capsys.readouterr().err.lower()


def test_build_annotation_index_conflicting_duplicates_warn_distinctly(
    em: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    header = ["Composer", "Piece", "Year"]
    rows = [
        {"Composer": "Bach", "Piece": "Foo", "Year": "1723"},
        {"Composer": "Bach", "Piece": "Foo", "Year": "1800"},
    ]

    index, dup_counts = em.build_annotation_index(header, rows, ["Composer", "Piece"])

    assert index[("Bach", "Foo")]["Year"] == "1723"
    assert dup_counts["conflicting"] == 1
    assert dup_counts["identical"] == 0
    assert "conflicting" in capsys.readouterr().err.lower()


def test_build_annotation_index_missing_key_column_raises_naming_column(
    em: ModuleType,
) -> None:
    header = ["Composer", "Piece", "Year"]
    rows = [{"Composer": "Bach", "Piece": "Foo", "Year": "1723"}]

    with pytest.raises(ValueError, match="Nope"):
        em.build_annotation_index(header, rows, ["Nope", "Piece"])


# ---------------------------------------------------------------------------
# enrich_rows
# ---------------------------------------------------------------------------


def _base_fixture() -> tuple[list[str], list[dict[str, str]]]:
    header = ["canonical_composer", "canonical_title", "midi_filename"]
    rows = [
        {
            "canonical_composer": "Bach",
            "canonical_title": "Foo",
            "midi_filename": "a.midi",
        },
        {
            "canonical_composer": "Nobody",
            "canonical_title": "Missing",
            "midi_filename": "b.midi",
        },
    ]
    return header, rows


def _annotation_index(em: ModuleType) -> dict[tuple[str, ...], dict[str, str]]:
    header = ["Composer", "Piece", "Year"]
    rows = [{"Composer": "Bach", "Piece": "Foo", "Year": "1723"}]
    index, _ = em.build_annotation_index(header, rows, ["Composer", "Piece"])
    return index


def test_enrich_rows_adds_columns_blank_not_na_on_unmatched(em: ModuleType) -> None:
    base_header, base_rows = _base_fixture()
    index = _annotation_index(em)

    out_header, out_rows, _ = em.enrich_rows(
        base_header,
        base_rows,
        [("canonical_composer", "Composer"), ("canonical_title", "Piece")],
        [("Year", "composition_year")],
        index,
    )

    assert out_header == base_header + ["composition_year"]
    assert out_rows[0]["composition_year"] == "1723"
    assert out_rows[1]["composition_year"] == ""
    assert out_rows[1]["composition_year"] != "NA"


def test_enrich_rows_coverage_report_with_unmatched_sample(em: ModuleType) -> None:
    base_header, base_rows = _base_fixture()
    index = _annotation_index(em)

    _, _, coverage = em.enrich_rows(
        base_header,
        base_rows,
        [("canonical_composer", "Composer"), ("canonical_title", "Piece")],
        [("Year", "composition_year")],
        index,
    )

    assert coverage.n_matched == 1
    assert coverage.n_total == 2
    assert coverage.n_unmatched_keys == 1
    assert ("Nobody", "Missing") in coverage.unmatched_sample


def test_enrich_rows_refuses_output_column_already_in_base(em: ModuleType) -> None:
    base_header = ["canonical_composer", "canonical_title", "composition_year"]
    base_rows = [
        {
            "canonical_composer": "Bach",
            "canonical_title": "Foo",
            "composition_year": "keep",
        }
    ]
    index = _annotation_index(em)

    with pytest.raises(ValueError, match="composition_year"):
        em.enrich_rows(
            base_header,
            base_rows,
            [("canonical_composer", "Composer"), ("canonical_title", "Piece")],
            [("Year", "composition_year")],
            index,
        )


def test_case_insensitive_matches_across_case(em: ModuleType) -> None:
    base_header = ["canonical_composer", "canonical_title"]
    base_rows = [{"canonical_composer": "bach", "canonical_title": "foo"}]
    ann_header = ["Composer", "Piece", "Year"]
    ann_rows = [{"Composer": "BACH", "Piece": "FOO", "Year": "1723"}]

    strict_index, _ = em.build_annotation_index(ann_header, ann_rows, ["Composer", "Piece"])
    _, strict_rows, strict_cov = em.enrich_rows(
        base_header,
        base_rows,
        [("canonical_composer", "Composer"), ("canonical_title", "Piece")],
        [("Year", "composition_year")],
        strict_index,
    )
    assert strict_cov.n_matched == 0
    assert strict_rows[0]["composition_year"] == ""

    folded_index, _ = em.build_annotation_index(
        ann_header, ann_rows, ["Composer", "Piece"], case_insensitive=True
    )
    _, folded_rows, folded_cov = em.enrich_rows(
        base_header,
        base_rows,
        [("canonical_composer", "Composer"), ("canonical_title", "Piece")],
        [("Year", "composition_year")],
        folded_index,
        case_insensitive=True,
    )
    assert folded_cov.n_matched == 1
    assert folded_rows[0]["composition_year"] == "1723"


# ---------------------------------------------------------------------------
# main end-to-end
# ---------------------------------------------------------------------------


def _write_base_metadata(path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["canonical_composer", "canonical_title", "midi_filename"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "canonical_composer": "Alexander Scriabin",
                "canonical_title": "24 Preludes Op. 11, No. 13-24",
                "midi_filename": "a.midi",
            }
        )
        writer.writerow(
            {
                "canonical_composer": "Nobody",
                "canonical_title": "Missing Piece",
                "midi_filename": "b.midi",
            }
        )
    return path


def _write_small_annotations(path: Path) -> Path:
    return _write_annotations(
        path,
        "# comment\n"
        "Composer|Piece|Year\n"
        "Alexander Scriabin|24 Preludes Op. 11, No. 13-24|1896\n",
    )


def _run_main(
    em: ModuleType,
    tmp_path: Path,
    *extra_args: str,
) -> tuple[int, Path, Path, Path]:
    meta_path = _write_base_metadata(tmp_path / "meta.csv")
    ann_path = _write_small_annotations(tmp_path / "ann.txt")
    out_path = tmp_path / "enriched.csv"
    code = em.main(
        [
            "--metadata",
            str(meta_path),
            "--annotations",
            str(ann_path),
            "--annotations-delimiter",
            "|",
            "--on",
            "canonical_composer=Composer",
            "--on",
            "canonical_title=Piece",
            "--add",
            "Year=composition_year",
            "--output",
            str(out_path),
            *extra_args,
        ]
    )
    return code, meta_path, ann_path, out_path


def test_main_end_to_end_rfc4180_provenance_and_partial_warning(
    em: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, meta_path, ann_path, out_path = _run_main(em, tmp_path)

    assert code == 0
    # Correct RFC4180 quoting for the comma-bearing title.
    raw = out_path.read_text(encoding="utf-8")
    assert '"24 Preludes Op. 11, No. 13-24"' in raw
    with out_path.open(newline="", encoding="utf-8") as handle:
        enriched = list(csv.DictReader(handle))
    assert len(enriched) == 2
    assert enriched[0]["composition_year"] == "1896"
    assert enriched[1]["composition_year"] == ""

    # Provenance sidecar.
    provenance_path = Path(str(out_path) + ".provenance.json")
    assert provenance_path.exists()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["metadata"] == str(meta_path)
    assert provenance["annotations"] == str(ann_path)
    assert provenance["metadata_sha256"] == em.sha256_of(meta_path)
    assert provenance["annotations_sha256"] == em.sha256_of(ann_path)
    assert provenance["n_matched"] == 1
    assert provenance["n_total"] == 2

    # Partial coverage warns on stderr.
    stderr = capsys.readouterr().err
    assert "matched 1/2 rows" in stderr
    assert "warning" in stderr.lower()


def test_main_require_full_coverage_nonzero_when_incomplete(
    em: ModuleType, tmp_path: Path
) -> None:
    code, _, _, _ = _run_main(em, tmp_path, "--require-full-coverage")
    assert code != 0


def test_main_require_full_coverage_zero_when_complete(
    em: ModuleType, tmp_path: Path
) -> None:
    meta_path = tmp_path / "meta.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["canonical_composer", "canonical_title"]
        )
        writer.writeheader()
        writer.writerow({"canonical_composer": "Bach", "canonical_title": "Foo"})
    ann_path = _write_annotations(tmp_path / "ann.txt", "Composer|Piece|Year\nBach|Foo|1723\n")
    out_path = tmp_path / "enriched.csv"

    code = em.main(
        [
            "--metadata",
            str(meta_path),
            "--annotations",
            str(ann_path),
            "--annotations-delimiter",
            "|",
            "--on",
            "canonical_composer=Composer",
            "--on",
            "canonical_title=Piece",
            "--add",
            "Year=composition_year",
            "--output",
            str(out_path),
            "--require-full-coverage",
        ]
    )
    assert code == 0


def test_main_refuses_to_clobber_metadata(
    em: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    meta_path = _write_base_metadata(tmp_path / "meta.csv")
    ann_path = _write_small_annotations(tmp_path / "ann.txt")

    code = em.main(
        [
            "--metadata",
            str(meta_path),
            "--annotations",
            str(ann_path),
            "--annotations-delimiter",
            "|",
            "--on",
            "canonical_composer=Composer",
            "--output",
            str(meta_path),
        ]
    )

    assert code != 0
    assert "refusing" in capsys.readouterr().err.lower()


def test_main_missing_key_column_is_nonzero_and_names_column(
    em: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    meta_path = _write_base_metadata(tmp_path / "meta.csv")
    ann_path = _write_small_annotations(tmp_path / "ann.txt")

    code = em.main(
        [
            "--metadata",
            str(meta_path),
            "--annotations",
            str(ann_path),
            "--annotations-delimiter",
            "|",
            "--on",
            "canonical_composer=Nope",
            "--output",
            str(tmp_path / "enriched.csv"),
        ]
    )

    assert code != 0
    assert "Nope" in capsys.readouterr().err


def test_main_duplicate_output_column_is_nonzero_and_names_column(
    em: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    meta_path = tmp_path / "meta.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["canonical_composer", "canonical_title", "Year"]
        )
        writer.writeheader()
        writer.writerow(
            {"canonical_composer": "Bach", "canonical_title": "Foo", "Year": "x"}
        )
    ann_path = _write_annotations(tmp_path / "ann.txt", "Composer|Piece|Year\nBach|Foo|1723\n")

    code = em.main(
        [
            "--metadata",
            str(meta_path),
            "--annotations",
            str(ann_path),
            "--annotations-delimiter",
            "|",
            "--on",
            "canonical_composer=Composer",
            "--on",
            "canonical_title=Piece",
            "--output",
            str(tmp_path / "enriched.csv"),
        ]
    )

    assert code != 0
    assert "Year" in capsys.readouterr().err


@pytest.mark.requires_r
def test_enriched_csv_round_trips_through_r(
    em: ModuleType, tmp_path: Path
) -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript not available")
    meta_path = tmp_path / "meta.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["canonical_composer", "canonical_title"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "canonical_composer": "Frédéric Chopin",
                "canonical_title": 'Variations and Fugue in E-flat Major, Op. 35, "Eroica"',
            }
        )
        writer.writerow(
            {"canonical_composer": "Leoš Janáček", "canonical_title": "Sonata 1.X.1905"}
        )
    ann_path = _write_annotations(
        tmp_path / "ann.txt",
        "Composer|Piece|Year\n"
        'Frédéric Chopin|Variations and Fugue in E-flat Major, Op. 35, "Eroica"|1802\n'
        "Leoš Janáček|Sonata 1.X.1905|1905\n",
    )
    out_path = tmp_path / "enriched.csv"
    code = em.main(
        [
            "--metadata",
            str(meta_path),
            "--annotations",
            str(ann_path),
            "--annotations-delimiter",
            "|",
            "--on",
            "canonical_composer=Composer",
            "--on",
            "canonical_title=Piece",
            "--add",
            "Year=composition_year",
            "--output",
            str(out_path),
        ]
    )
    assert code == 0

    composers_path = tmp_path / "composers.txt"
    titles_path = tmp_path / "titles.txt"
    # Pin a UTF-8 C locale so R parses decimals as C numerics yet keeps
    # non-ASCII composer names intact (same rationale as r_locale_env in
    # scripts/run_mixed_effects_analysis.py).
    r_env = dict(os.environ)
    r_env["LC_ALL"] = "C.UTF-8"
    r_env["LANG"] = "C.UTF-8"
    r_code = (
        f"d <- read.csv('{out_path}', stringsAsFactors = FALSE, fileEncoding = 'UTF-8');"
        f"writeLines(d$canonical_composer, '{composers_path}', useBytes = TRUE);"
        f"writeLines(as.character(d$composition_year), '{titles_path}', useBytes = TRUE);"
        f"writeLines(d$canonical_title, '{tmp_path / 'rtitles.txt'}', useBytes = TRUE)"
    )
    result = subprocess.run(
        [rscript, "--vanilla", "-e", r_code],
        capture_output=True,
        text=True,
        timeout=120,
        env=r_env,
    )
    assert result.returncode == 0, result.stderr
    assert composers_path.read_text(encoding="utf-8").splitlines() == [
        "Frédéric Chopin",
        "Leoš Janáček",
    ]
    assert (tmp_path / "rtitles.txt").read_text(encoding="utf-8").splitlines() == [
        'Variations and Fugue in E-flat Major, Op. 35, "Eroica"',
        "Sonata 1.X.1905",
    ]
