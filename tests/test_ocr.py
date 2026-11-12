"""Smoke and regression tests. Run from the repository root: pytest"""

import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC)

import edit  # noqa: E402
import OCR  # noqa: E402
import utilities  # noqa: E402
from character_segmentation import vertical_transitions  # noqa: E402
from segmentation import projection_segmentation  # noqa: E402
from train import prepare_char  # noqa: E402

TRUTH_DIR = os.path.join(SRC, 'truth')
# Measured accuracy on the bundled samples is 95.7%; leave room for
# floating-point / library-version noise but catch real regressions.
MIN_ACCURACY = 0.93


def test_last_segment_touching_the_edge_is_kept():
    img = np.zeros((20, 60), np.uint8)
    img[:, 5:15] = 255
    img[:, 40:60] = 255          # touches the right edge
    assert len(projection_segmentation(img, 'vertical', cut=3)) == 2

    img = np.zeros((60, 20), np.uint8)
    img[5:15, :] = 255
    img[40:60, :] = 255          # touches the bottom edge
    assert len(projection_segmentation(img, 'horizontal', cut=3)) == 2


@pytest.mark.parametrize('crop', [
    None,
    np.zeros((10, 0), np.uint8),   # empty
    np.zeros((10, 6), np.uint8),   # no ink
])
def test_prepare_char_returns_none_for_unusable_images(crop):
    assert prepare_char(crop) is None


def test_prepare_char_resizes_to_25x25():
    crop = np.zeros((12, 8), np.uint8)
    crop[2:10, 2:6] = 255
    assert prepare_char(crop).shape == (25, 25)


@pytest.mark.parametrize('column, expected', [
    ([0, 0, 0], 0),
    ([1, 1, 1], 1),            # a run that reaches the bottom still counts once
    ([0, 1, 1, 0], 2),
    ([1, 0, 1, 0, 1], 5),
    ([0, 1, 0, 1, 1, 0, 1], 5),
])
def test_vertical_transitions(column, expected):
    img = np.array(column, np.uint8).reshape(-1, 1)
    assert vertical_transitions(img, 0) == expected
    assert vertical_transitions(img.astype(bool), 0) == expected


@pytest.mark.parametrize('name', ['LinearSVM', '1L_NN', '2L_NN', 'Gaussian_Naive_Bayes'])
def test_every_saved_model_loads_and_predicts(name):
    # OCR.py only uses 2L_NN, but a stale pickle in another file (the Naive
    # Bayes one once could not predict at all) should not go unnoticed.
    import pickle
    with open(os.path.join(SRC, 'models', f'{name}.sav'), 'rb') as fin:
        model = pickle.load(fin)
    prediction = model.predict(np.zeros((1, 625), np.uint8))
    assert len(prediction) == 1


def test_projection_rejects_unknown_axis():
    with pytest.raises(ValueError):
        utilities.projection(np.zeros((2, 2)), 'diagonal')


def test_edit_accuracy_handles_empty_truth():
    assert edit.accuracy(0, 0) == 1.0
    assert edit.accuracy(3, 0) == 0.0
    assert edit.accuracy(5, 10) == 0.5


def test_image_roundtrip_with_non_ascii_path(tmp_path):
    path = str(tmp_path / 'ا.png')
    utilities.write_image(path, np.full((5, 5), 255, np.uint8))
    assert utilities.read_image(path, 0).shape == (5, 5)


@pytest.mark.parametrize('workers', [[], ['--workers', '1'], ['--workers', '2']],
                         ids=['default', 'in-process', 'two-workers'])
def test_end_to_end_on_bundled_samples(tmp_path, workers):
    exit_code = OCR.main(['--output', str(tmp_path)] + workers)
    assert exit_code == 0

    results, average = edit.evaluate(str(tmp_path / 'text'), TRUTH_DIR)
    assert {name for name, _ in results} == {'capr12.txt', 'capr14.txt'}
    assert average >= MIN_ACCURACY, f'accuracy dropped to {average:.2%}'


@pytest.mark.parametrize('value', ['0', '-1'])
def test_workers_must_be_positive(value):
    with pytest.raises(SystemExit):
        OCR.parse_args(['--workers', value])


def test_blas_is_limited_to_one_thread_per_process():
    """Many cores x many worker processes x one OpenBLAS thread per core used
    to exhaust memory ("OpenBLAS: memory allocation failed") on Windows."""

    code = (
        'import sys; sys.path.insert(0, sys.argv[1]); import OCR; '
        'from threadpoolctl import threadpool_info; '
        'print(sorted({i["num_threads"] for i in threadpool_info() '
        'if i["internal_api"] == "openblas"}))'
    )
    env = {k: v for k, v in os.environ.items()
           if k not in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}
    out = subprocess.run([sys.executable, '-c', code, SRC], env=env,
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out in ('[1]', '[]'), f'OpenBLAS thread counts: {out}'   # [] = no OpenBLAS in use


# ---- PDF input -------------------------------------------------------------

PDF_MIN_ACCURACY = 0.90     # about 94% measured; the test PDF is lossy (JPEG)
PAGE_TRUTH = {1: 'capr12.txt', 2: 'capr14.txt'}


@pytest.fixture
def sample_pdf_folder(tmp_path):
    """A folder with book.pdf: page 1 is capr12.png, page 2 is capr14.jpg."""

    from PIL import Image

    folder = tmp_path / 'in'
    folder.mkdir()
    pages = [Image.open(os.path.join(SRC, 'test', name)).convert('RGB')
             for name in ('capr12.png', 'capr14.jpg')]
    pages[0].save(folder / 'book.pdf', save_all=True, append_images=pages[1:], resolution=300)
    return folder


def split_pages(text):
    parts = re.split(r'^=== Page (\d+) ===$', text, flags=re.M)
    return {int(parts[i]): ''.join(parts[i + 1].split()) for i in range(1, len(parts), 2)}


def test_pdf_gives_one_file_with_a_marker_and_the_right_text_per_page(sample_pdf_folder, tmp_path):
    from rapidfuzz.distance import Levenshtein

    out = tmp_path / 'out'
    assert OCR.main(['--input', str(sample_pdf_folder), '--output', str(out), '--workers', '1']) == 0
    assert os.listdir(out / 'text') == ['book.txt']

    text = (out / 'text' / 'book.txt').read_text(encoding='utf8')
    assert text.startswith('=== Page 1 ===\n')
    pages = split_pages(text)
    assert sorted(pages) == [1, 2]
    for number, truth_name in PAGE_TRUTH.items():
        with open(os.path.join(TRUTH_DIR, truth_name), encoding='utf8') as f:
            truth = ''.join(f.read().split())
        accuracy = 1 - Levenshtein.distance(pages[number], truth) / len(truth)
        assert accuracy >= PDF_MIN_ACCURACY, f'page {number}: {accuracy:.2%}'


def test_pdf_pages_are_numbered_from_one_and_are_three_channel_images(sample_pdf_folder):
    pages = list(OCR.pdf_pages(str(sample_pdf_folder / 'book.pdf'), 300))
    assert [number for number, _ in pages] == [1, 2]
    for _, image in pages:
        assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8


def test_a_broken_pdf_is_reported_but_other_files_are_still_read(tmp_path):
    folder = tmp_path / 'in'
    folder.mkdir()
    (folder / 'broken.pdf').write_bytes(b'this is not a pdf')
    shutil.copy(os.path.join(SRC, 'test', 'capr14.jpg'), folder / 'capr14.jpg')

    out = tmp_path / 'out'
    assert OCR.main(['--input', str(folder), '--output', str(out), '--workers', '1']) == 1
    assert os.listdir(out / 'text') == ['capr14.txt']


def test_book_pdf_and_book_png_do_not_overwrite_each_other():
    names = OCR.output_names(['x/book.pdf', 'x/book.png', 'x/other.png'])
    assert names == {'x/book.pdf': 'book', 'x/book.png': 'book.png', 'x/other.png': 'other'}


@pytest.mark.parametrize('dpi', ['5', '5000'])
def test_dpi_must_be_in_range(dpi):
    with pytest.raises(SystemExit):
        OCR.parse_args(['--dpi', dpi])
