# Backend Research Paper reader powered by Nougat and deployed on modal platform.
import modal

stub = modal.Stub("streamlit-hack")


def get_model():
    from nougat.utils.checkpoint import get_checkpoint
    CHECKPOINT = get_checkpoint('nougat')


streamlit_image = modal.Image.debian_slim().apt_install("ffmpeg", "git").pip_install("requests",
                                                                                     "pypdf>=3.1.0",
                                                                                     "nougat-ocr==0.1.8",
                                                                                     "tqdm",
                                                                                     "torch==2.0.0").run_function(get_model)


@stub.function(image=streamlit_image)
def is_arxiv_url(url: str) -> bool:
    import requests
    import re
    import urllib
    from typing import Optional, Union
    arxiv_pattern = r'https?://arxiv\.org/abs/.+'
    return bool(re.match(arxiv_pattern, url))


@stub.function(image=streamlit_image)
def is_acl_anthology_url(url: str) -> bool:
    import requests
    import re
    import urllib
    from typing import Optional, Union
    acl_anthology_pattern = r'https://aclanthology\.org/.*?/'
    return bool(re.match(acl_anthology_pattern, url))


@stub.function(image=streamlit_image, gpu="any", timeout=1000)
def nougat_paper_pdf(url: str) -> str:
    import requests
    import re
    import urllib
    from typing import Optional, Union
    from nougat import NougatModel
    from nougat.utils.dataset import LazyDataset
    from nougat.utils.checkpoint import get_checkpoint
    from nougat.postprocessing import markdown_compatible
    import pypdf
    import sys
    from pathlib import Path
    import logging
    import re
    import argparse
    import re
    from functools import partial
    import torch
    from torch.utils.data import ConcatDataset
    from tqdm import tqdm

    class InvalidURLException(Exception):
        pass

    class DownloadError(Exception):
        pass

    paper_url = url

    if is_arxiv_url.call(paper_url):
        pdf_url = paper_url.replace('/abs/', '/pdf/') + '.pdf'
    elif is_acl_anthology_url.call(paper_url):
        pdf_url = paper_url.rstrip('/') + '.pdf'
    else:
        raise InvalidURLException('Invalid URL. Please provide a valid ArXiv or ACL Anthology URL.')

    try:
        response = requests.get(pdf_url)
        response.raise_for_status()  # Raise an exception if there's an HTTP error

        pdf_filename = pdf_url.split('/')[-1]

        with open(pdf_filename, 'wb') as pdf_file:
            pdf_file.write(response.content)

    except requests.exceptions.RequestException as e:
        raise DownloadError(f'Failed to download the PDF: {e}')

    filename = pdf_filename

    batchsize = 4
    checkpoint = "nougat"
    model = NougatModel.from_pretrained(checkpoint)

    def move_to_device(model):
        if torch.cuda.is_available():
            return model.to("cuda").to(torch.bfloat16)
        try:
            if torch.backends.mps.is_available():
                return model.to("mps")
        except AttributeError:
            pass

        return model.to(torch.bfloat16)

    model = move_to_device(model)
    model.eval()
    datasets = []
    try:
        dataset = LazyDataset(
            filename, partial(model.encoder.prepare_input, random_padding=False)
        )

    except pypdf.errors.PdfStreamError:
        logging.info(f"Could not load file {str(filename)}.")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=False,
        collate_fn=LazyDataset.ignore_none_collate,
    )

    predictions = []
    file_index = 0
    page_num = 0
    combined_output = ""
    for i, (sample, is_last_page) in enumerate(tqdm(dataloader)):
        model_output = model.inference(image_tensors=sample)
        # check if model output is faulty
        for j, output in enumerate(model_output["predictions"]):
            if page_num == 0:
                logging.info(
                    "Processing file"

                )
            page_num += 1
            if output.strip() == "[MISSING_PAGE_POST]":
                # uncaught repetitions -- most likely empty page
                predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")

            else:
                output = markdown_compatible(output)
                predictions.append(output)
            if is_last_page[j]:
                out = "".join(predictions).strip()
                out = re.sub(r"\n{3,}", "\n\n", out).strip()
                combined_output += out

                predictions = []
                page_num = 0
                file_index += 1

    return combined_output


@stub.function(image=streamlit_image, timeout=1200)
def main(url: str) -> str:
    return nougat_paper_pdf.call(url)


@stub.local_entrypoint()
def test_method(url):
    main.call(url)