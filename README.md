# **Arabic OCR**

  ![GitHub contributors](https://img.shields.io/badge/contributions-open-green)
  [![GitHub forks](https://img.shields.io/github/forks/HusseinYoussef/Arabic-OCR)](https://github.com/HusseinYoussef/Arabic-OCR/network)
  [![GitHub stars](https://img.shields.io/github/stars/HusseinYoussef/Arabic-OCR)](https://github.com/HusseinYoussef/Arabic-OCR/stargazers)
  [![GitHub license](https://img.shields.io/github/license/HusseinYoussef/Arabic-OCR)](https://github.com/HusseinYoussef/Arabic-OCR/blob/master/LICENSE)
  
- OCR system for Arabic language that converts images of typed text to machine-encoded text.<br>
- The system aims to solve a simpler problem of OCR with images that contain only Arabic characters (check the dataset link below to see a sample of the images).

## Important Note
> The system currently supports only letters: the 28 letters ا-ي and the لا ligature (no numbers, special symbols, ة, ى or hamza forms).
## Setup

Install 64-bit Python 3.11 to 3.14 (3.11 and 3.13 are tested on Linux and Windows in CI), then run this command:

```shell
pip install -r requirements.txt
```

> **Python 3.15 and 32-bit Python do not work yet.** Some dependencies (scikit-image for 3.15; scipy and scikit-learn for 32-bit) publish no pre-built packages for them, so pip tries to compile them from source and fails with errors about missing compilers (`meson`, `cl`, `gcc`...). Use a supported Python in a virtual environment, for example on Windows: `py -3.13 -m venv .venv`, then `.venv\Scripts\activate`.

> The bundled models in `src/models` are pickles saved with scikit-learn 1.9.1, which needs Python 3.11 or newer. Pickles load reliably only in the same or a newer scikit-learn version, and loading one can run arbitrary code, so only load pickles you trust. If a future scikit-learn stops loading them, retrain with `python train.py` (see [Training](#training)).

## Run

1. Put the images and/or scanned PDFs in the src/test directory (or choose another folder with `--input`)
2. Run the following command (from any directory)
   ```shell
   python src/OCR.py
   ```
   Options: `--input DIR` (default `src/test`), `--output DIR` (default `src/output`), `--workers N` (processes used; default: the number of CPU cores, at most 4; `1` uses no extra processes), `--dpi N` (resolution used to render PDF pages; default 300), `-v` (log characters that could not be read).
3. The output folder will be created with:
   - text folder which has text files corresponding to the images and PDFs.
   - running_time file which has the time taken to process each image or PDF.

### PDFs

A PDF gives one text file (`name.txt`) with a marker line before every page, in the PDF's page order starting from 1 (these are not the page numbers printed on the pages):

```
=== Page 1 ===
text of the first page...

=== Page 2 ===
text of the second page...
```

If a page cannot be read, its marker is still written (with no text under it) and the failure is reported. If a folder holds both `book.pdf` and `book.png`, the second result is saved as `book.png.txt` so nothing is overwritten.

Only **scanned** PDFs (pages that are pictures) need this; for a PDF with real, selectable text, copy the text out instead. Each page is turned into an image first, and **the recognition is very sensitive to the size of the text**: it was tuned on a fixed font size. For a test PDF built from the sample images, accuracy was about 94% at 300 DPI (the resolution where the page's pixels match the original images) but only about 44% at 200 DPI and about 82-86% at 400 DPI. When some words cannot be split into letters (text that is too small or too large, pictures, lines, borders) they are left out, and one warning per page says how many, e.g. `book.pdf, page 3: 41 of 174 words could not be split into letters and were skipped`; `-v` gives the details. A page with many skipped words usually means the `--dpi` does not suit the text size. If a PDF reads badly, try other `--dpi` values (for example 100, 150 or 200); the right value depends on how large the text is in your PDF.

   The output folder is generated, so it is not tracked by git.

> **`OpenBLAS: memory allocation failed after 10 tries`** (or the computer running out of memory): every worker process loads numpy, OpenCV, scikit-learn and the model. Update to the latest version, which limits each worker to one math thread, and if it still happens use fewer workers, for example `--workers 2`, or `--workers 1` to use no extra processes at all.

## Pipeline

![Pipeline](./Figures/pipeline.PNG)

## Dataset

- Link to dataset of images and the corresponding text: [here](https://drive.google.com/open?id=1Nbp9ZXLlWV3n8yRMwj2gjs_rE6qGZU01).
- We used 1000 images to generate character dataset that we used for training.
- `Dataset/char_4K_sample.rar` in this repository is a sample of the character dataset.

## Training

All commands are run from the `src` folder.

1. Make the character images available as `Dataset/char_sample/<character>/<n>.png` (`train.py` reads that layout; check that it matches what you extract from the archive). `python dataset.py` can generate such images from the full dataset (`Dataset/scanned/*.png` with `Dataset/text/*.txt`); it writes them to `Dataset/chars`.
2. Train: `python train.py` trains all four classifiers, or name the ones you want: `python train.py 2L_NN`. Models are saved to `src/models` and their scores to `src/models/report.txt`.
3. `OCR.py` uses `src/models/2L_NN.sav`.

# Examples

## Line Segmentation

![Line](./Figures/line.png)

## Word Segmentation

![Word](./Figures/word.png)

## Character Segmentation

![Word](./Figures/char4.png)
![Word](./Figures/char3.png)
![Word](./Figures/char2.png)
![Word](./Figures/char1.png)

# Testing

**NOTE**: Make sure you have a folder with the truth output with same file names to compare it with the predicted text.

From within `src` folder run:

`python edit.py 'output/text' 'truth'`

![Test](./Figures/test.png)

## Automated tests

```shell
pip install -r requirements-dev.txt
pytest
```

The tests include an end-to-end run on the two sample images in `src/test` that fails if the average accuracy drops below 93% (it is currently 95.7%). They run on every push through GitHub Actions.

# Performance

- Average accuracy: 95%.
- Average time per image: 16 seconds.
- For reference, the two small sample images in `src/test` (about 340 words in total) take 0.5-0.7 seconds each on a 2-core Linux machine.

---

**NOTE**

We achieved these results when we used only the flatten image as feature.

---

# References

1. [An Efficient, Font Independent Word and Character Segmentation Algorithm for Printed Arabic Text](https://www.researchgate.net/publication/335562626_An_Efficient_Font_Independent_Word_and_Character_Segmentation_Algorithm_for_Printed_Arabic_Text).

2. [A Robust Line Segmentation Algorithm for Arabic Printed Text with Diacritics](https://www.researchgate.net/publication/317876029_A_Robust_Line_Segmentation_Algorithm_for_Arabic_Printed_Text_with_Diacritics).

3. [Arabic Character Segmentation Using Projection Based Approach with Profile's Amplitude Filter
   ](https://www.researchgate.net/publication/318205989_Arabic_Character_Segmentation_Using_Projection_Based_Approach_with_Profile's_Amplitude_Filter).
