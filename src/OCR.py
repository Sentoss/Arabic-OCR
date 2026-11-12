import os

# One BLAS thread per process. The recognition work is already spread over a
# pool of worker processes, and by default every one of them would start an
# OpenBLAS thread per CPU core, each with its own memory buffer. On a machine
# with many cores that exhausts memory ("OpenBLAS: memory allocation failed
# after 10 tries") before any text is read. This has to be set before numpy is
# imported; a value the user set themselves is respected.
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import argparse
import logging
import multiprocessing as mp
import pickle
import sys
import time
from glob import glob

import numpy as np
from tqdm import tqdm

from character_segmentation import segment
from segmentation import extract_words
from train import prepare_char, featurizer
from utilities import read_image

log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')
PDF_EXTENSION = '.pdf'

# Scanned PDF pages are rendered to images at this resolution. The recognition
# was tuned on a fixed font size, so this may need adjusting for your scans.
DEFAULT_DPI = 300
DPI_RANGE = (50, 1200)

# Written before the text of every PDF page, e.g. "=== Page 3 ===".
PAGE_MARKER = '=== Page {} ==='

model_name = '2L_NN.sav'

# Every worker process holds its own copy of numpy, OpenCV, scikit-learn and
# the model (a few hundred MB), so the default is deliberately modest.
DEFAULT_MAX_WORKERS = 4


def load_model():
    path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model file not found: {path}')
    with open(path, 'rb') as fin:
        return pickle.load(fin)


_model = None


def get_model():
    """The classifier, loaded once per process instead of once per word."""

    global _model
    if _model is None:
        _model = load_model()
    return _model


def default_workers():
    return max(1, min(os.cpu_count() or 1, DEFAULT_MAX_WORKERS))


def init_worker(log_level):
    # Windows starts workers from scratch (no inherited logging setup).
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    get_model()


def run2(obj):
    word, line = obj
    model = get_model()

    # For each word in the image
    try:
        char_imgs = segment(line, word)
    except Exception as exc:
        # One bad word must not abort the whole image.
        log.warning('Skipping a word that failed to segment: %s: %s', type(exc).__name__, exc)
        return ''

    feature_vectors = []
    skipped = 0
    # For each character in the word
    for char_img in char_imgs:
        ready_char = prepare_char(char_img)
        if ready_char is None:
            skipped += 1
            continue
        feature_vectors.append(featurizer(ready_char))

    if skipped:
        log.debug('Dropped %d unusable character image(s) in a word', skipped)
    if not feature_vectors:
        return ''

    # One predict call for the whole word instead of one per character.
    return ''.join(model.predict(np.array(feature_vectors)))


def recognize(full_image, pool=None):
    """The text of one page image, with the lines separated by newlines."""

    words = extract_words(full_image)       # [ (word, its line),(word, its line),..  ]
    if pool is None:
        predicted_words = [run2(word) for word in words]
    else:
        predicted_words = pool.map(run2, words)

    # Rebuild the text line by line. Words of the same line share the same
    # `line` array object, which is how a new line is detected.
    lines_text = []
    current = []
    previous_line = None
    for (_, line), word_text in zip(words, predicted_words):
        if previous_line is not None and line is not previous_line:
            lines_text.append(' '.join(current))
            current = []
        current.append(word_text)
        previous_line = line
    if current:
        lines_text.append(' '.join(current))
    return '\n'.join(lines_text)


def is_pdf(path):
    return os.path.splitext(path)[1].lower() == PDF_EXTENSION


def pdf_pages(pdf_path, dpi):
    """Yield (page number, image) for each page of a PDF, one page at a time.

    Page numbers count from 1, in the order of the PDF. The image is None for
    a page that could not be rendered (the error is logged)."""

    # Imported here: only PDFs need it, and the worker processes never do.
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    try:
        for index in range(len(pdf)):
            image = None
            try:
                page = pdf[index]
                try:
                    image = np.array(page.render(scale=dpi / 72).to_numpy())
                finally:
                    page.close()
                if image.ndim == 3 and image.shape[2] == 4:     # drop an alpha channel
                    image = np.ascontiguousarray(image[:, :, :3])
            except Exception as exc:
                log.error('%s, page %d could not be rendered: %s: %s',
                          os.path.basename(pdf_path), index + 1, type(exc).__name__, exc)
            yield index + 1, image
    finally:
        pdf.close()


def run(path, output_dir, pool=None, dpi=DEFAULT_DPI, name=None):
    """Recognize one image or PDF and write its text file.

    A PDF gives one text file with a PAGE_MARKER line before every page.
    Returns (name, seconds)."""

    name = name or os.path.splitext(os.path.basename(path))[0]
    failed_pages = 0

    if is_pdf(path):
        before = time.time()
        parts = []
        for number, page_image in pdf_pages(path, dpi):
            text = ''
            if page_image is None:
                failed_pages += 1
            else:
                try:
                    text = recognize(page_image, pool)
                except Exception as exc:
                    log.error('%s, page %d failed: %s: %s',
                              os.path.basename(path), number, type(exc).__name__, exc)
                    failed_pages += 1
            parts.append(f'{PAGE_MARKER.format(number)}\n{text}')
        if not parts:
            raise ValueError(f'No pages found in {path}')
        predicted_text = '\n\n'.join(parts)
    else:
        # Read test image
        full_image = read_image(path)
        if full_image is None:
            raise ValueError(f'Could not read image: {path}')
        before = time.time()
        predicted_text = recognize(full_image, pool)

    exc_time = time.time() - before

    with open(os.path.join(output_dir, 'text', f'{name}.txt'), 'w', encoding='utf8') as fo:
        fo.write(predicted_text)

    if failed_pages:
        raise RuntimeError(f'{failed_pages} page(s) could not be read (the text file was still written)')
    return (name, exc_time)


def find_inputs(input_dir):
    extensions = IMAGE_EXTENSIONS + (PDF_EXTENSION,)
    return sorted(path for path in glob(os.path.join(input_dir, '*'))
                  if os.path.splitext(path)[1].lower() in extensions)


def output_names(paths):
    """Name (without .txt) of each input's result file. book.pdf and book.png
    would both give book.txt, so the second one keeps its extension."""

    names, used = {}, set()
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        if name in used:
            name = os.path.basename(path)
        used.add(name)
        names[path] = name
    return names


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Arabic OCR: images and scanned PDFs of typed text -> text files')
    parser.add_argument('--input', default=os.path.join(BASE_DIR, 'test'),
                        help='folder with the images (png, jpg, jpeg, bmp) and PDFs; default: src/test')
    parser.add_argument('--output', default=os.path.join(BASE_DIR, 'output'),
                        help='folder for the results; default: src/output')
    parser.add_argument('--workers', type=int, default=None, metavar='N',
                        help='worker processes (default: number of CPU cores, at most %d); '
                             '1 runs everything in this process. Use a lower number '
                             'if the computer runs out of memory' % DEFAULT_MAX_WORKERS)
    parser.add_argument('--dpi', type=int, default=DEFAULT_DPI,
                        help='resolution used to render PDF pages (default: %d)' % DEFAULT_DPI)
    parser.add_argument('-v', '--verbose', action='store_true', help='also log dropped characters')
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error('--workers must be at least 1')
    if not DPI_RANGE[0] <= args.dpi <= DPI_RANGE[1]:
        parser.error('--dpi must be between %d and %d' % DPI_RANGE)
    return args


def main(argv=None):
    args = parse_args(argv)
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    output_dir = args.output
    os.makedirs(os.path.join(output_dir, 'text'), exist_ok=True)

    images_paths = find_inputs(args.input)
    if not images_paths:
        log.error('No images or PDFs found in %s', args.input)
        return 1
    names = output_names(images_paths)

    before = time.time()

    running_time = []
    failed = []

    # One pool for all images (starting workers is slow, especially on Windows).
    workers = args.workers or default_workers()
    pool = mp.Pool(workers, initializer=init_worker, initargs=(log_level,)) if workers > 1 else None
    try:
        for images_path in tqdm(images_paths, total=len(images_paths)):
            try:
                running_time.append(run(images_path, output_dir, pool, args.dpi, names[images_path]))
            except Exception as exc:
                log.error('Failed on %s: %s: %s', images_path, type(exc).__name__, exc)
                failed.append(images_path)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    running_time.sort()
    with open(os.path.join(output_dir, 'running_time.txt'), 'w') as r:
        for t in running_time:
            r.write(f'image#{t[0]}: {t[1]}\n')       # if no need for printing 'image#id'.

    after = time.time()
    print(f'total time to finish {len(running_time)} images:')
    print(after - before)
    if failed:
        print(f'{len(failed)} image(s) failed: {", ".join(failed)}')
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
